import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
import scipy as sp
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from transformers.trainer import logger
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from DMPO_config import DMPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from utils import print_prompt_completions_sample
import wandb
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fast_samplers.fast_dllm.generate import generate_llada, generate_pd, generate_with_prefix_cache, generate_with_dual_cache
from fast_samplers.wino.generate import generate_wino

if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class DMPOTrainer(GRPOTrainer):
    """
    Distribution Matching Policy Optimization (DMPO) Trainer for Diffusion Language Models.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[DMPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        # check parameter compatibility for (P)CE
        assert args.alpha == -1 or args.alpha >= 0.0, "Invalid alpha value"
        assert 0.0 <= args.coeff <= 1.0, "Invalid coeff value"
        assert 0.0 <= args.ada_coeff_ess_threshold <= 1.0, "Invalid ada_coeff_ess_threshold value"
        if args.loss_antithetic and (args.num_replicates % 2 == 1 or self.args.compute_ref_log_prob_elbo_size % 2 == 1):
            logger.warning("num_replicates and compute_ref_log_prob_elbo_size should be even")

        assert args.use_fast_sampler in ["fast_dllm", "wino", "no"], "Invalid fast sampler"
        assert args.sampler in ["roar", "llada", "pd", "pd_cache_prefix", "pd_cache_dual", "wino"], "Invalid sampler"
        if args.sampler in ["pd_cache_prefix", "pd_cache_dual"]:
            assert args.use_fast_sampler == "fast_dllm", \
                "Samplers `pd_cache_prefix` and `pd_cache_dual` can only be used for Fast-dLLM"
        if args.sampler == "wino":
            assert args.use_fast_sampler == "wino", "Sampler `wino` can only be used for WINO"

        if args.sampler != "roar":
            if not self.args.compute_ref_log_prob_elbo:
                self.args.compute_ref_log_prob_elbo = True
                warnings.warn("`self.args.compute_ref_log_prob_elbo` set to True! "
                              "Require ELBO to approximate sequence log probability.")
            self.generate = {"llada": generate_llada,
                             "pd": generate_pd,
                             "pd_cache_prefix": generate_with_prefix_cache,
                             "pd_cache_dual": generate_with_dual_cache,
                             "wino": generate_wino,
                             }[args.sampler]

    #################### loss computation and buffer preparation ####################

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Rewrite GRPOTrainer._prepare_inputs() to avoid getting None when restarting training from a checkpoint.
        """
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if (self.state.global_step % self.num_iterations == 0 
                or self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] is None):
                # Originally, `if self.state.global_step % self.num_iterations == 0:`
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs


    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids = inputs["prompt_ids"] # [bs, prompt_length]
        batch_size = prompt_ids.shape[0]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"] # both [bs, gen_length]
        advantages, negative_advantages = inputs["advantages"], inputs["negative_advantages"] # both [bs]
        log_prob_cur = inputs["log_prob_cur"] # [bs]
        coeff = inputs["coeff"] # [1]
        
        prompt_ids, completion_ids, completion_mask = [
            x.repeat([self.args.num_replicates, 1]) for x in [prompt_ids, completion_ids, completion_mask]]
        # [bs * num_replicates, prompt_length or gen_length]
        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1) # [bs * num_replicates, seq_len = prompt_length + gen_length]
        gen_length = completion_ids.size(1)  # only compute logits for completion tokens
        del inputs, prompt_ids, completion_ids

        if not self.args.loss_antithetic:
            lamda = torch.rand(input_ids.shape[0], device=model.device) # [bs * num_replicates]
            if self.args.loss_mask_prob_clamp: lamda = 0.1 + 0.9 * lamda # now in [0.1, 0.9]
            masked_index = torch.rand(input_ids.shape[0], gen_length, device=model.device) < lamda.unsqueeze(1) # [bs * num_replicates, gen_length]
        else:
            lamda = torch.rand(input_ids.shape[0] // 2, device=model.device) # [bs * num_replicates / 2]
            if self.args.loss_mask_prob_clamp: lamda = 0.1 + 0.9 * lamda # now in [0.1, 0.9]
            masked_index = torch.rand(input_ids.shape[0] // 2, gen_length, device=model.device) < lamda.unsqueeze(1) # [bs * num_replicates / 2, gen_length]
            masked_index = torch.cat([masked_index, ~masked_index], dim=0) # [bs * num_replicates, gen_length]
        if self.args.loss_mask_non_eos: # only apply mask on non EOS positions
            masked_index[~completion_mask] = False
            # recall that in completion_mask, everything after the first EOS token is False 
        full_masked_index = torch.full(input_ids.shape, False, device=model.device)
        full_masked_index[:, -gen_length:] = masked_index
        # [bs * num_replicates, seq_len], do not mask the prompt
        m = masked_index.sum(dim=-1).clamp(min=1) # [bs * num_replicates], number of masks in each row, clamped to >= 1
        del masked_index

        perturbed_input_ids = torch.where(full_masked_index, self.args.mask_id, input_ids)
        logits = model(perturbed_input_ids).logits # [bs * num_replicates, seq_len, vocab_size]

        # the following implementation using F.cross_entropy is equivalent and more efficient
        losses = F.cross_entropy(input=logits.view(-1, logits.shape[-1]), target=input_ids.view(-1), reduction='none').view(logits.shape[:-1])
        # [N := bs * num_replicates * seq_len, vocab_size], [N] -> [N] -> [bs * num_replicates, seq_len], don't require logits to be log-softmaxed
        losses[~full_masked_index] = 0
        
        if self.args.loss == "wdce":
            advantages, negative_advantages = [
                x.repeat(self.args.num_replicates) for x in [advantages, negative_advantages]] # [bs * num_replicates]

            if self.args.advantage_centering:
                if self.args.advantage_centering_unbias:
                    # first compute log_prob_theta(x)
                    log_prob_theta = (-losses).clone().detach()
                    log_prob_theta = log_prob_theta.view(self.args.num_replicates, batch_size, -1).transpose(0, 1) # [bs, num_replicates, seq_len]
                    t_weights = (gen_length / m).view(self.args.num_replicates, batch_size, 1).transpose(0, 1) # [bs, num_replicates, 1]
                    log_prob_theta = (t_weights * log_prob_theta).mean(dim = 1).sum(dim = 1) # [bs]

                    centering_factor = (log_prob_theta - log_prob_cur).softmax(dim = -1)
                    centering_factor = centering_factor.repeat(self.args.num_replicates)
                elif self.args.advantage_centering_neg:
                    centering_factor = negative_advantages
                else:
                    centering_factor = advantages.mean(dim=-1, keepdim=True)
                advantages -= self.args.centering_strength * centering_factor
                
            
            loss = (losses.sum(dim=-1) / m * advantages).sum() / self.args.num_replicates
            # theoretically should be gen_length / m, we remove gen_length (fixed throughout) for smaller loss scales
            return loss
        
        elif self.args.loss == "ddo":
            log_prob_theta = (-losses).view(self.args.num_replicates, batch_size, -1).transpose(0, 1) # [bs, num_replicates, seq_len]
            t_weights = (gen_length / m).view(self.args.num_replicates, batch_size, 1).transpose(0, 1) # [bs, num_replicates, 1]
            
            if self.args.ddo_indep_set:
                log_prob_theta = (t_weights * log_prob_theta).reshape(batch_size, 2, self.args.num_replicates // 2, -1).mean(dim = 2).sum(dim = -1)
                log_prob_theta_real = log_prob_theta[:, 0]
                log_prob_theta_fake = log_prob_theta[:, 1]
                log_prob_gap_real = log_prob_theta_real - log_prob_cur
                log_prob_gap_fake = log_prob_theta_fake - log_prob_cur
            else:
                log_prob_theta = (t_weights * log_prob_theta).sum(dim = 1).sum(dim = 1) # [bs]
                log_prob_gap_real = log_prob_theta - log_prob_cur
                log_prob_gap_fake = log_prob_theta - log_prob_cur
            
            ## In DDO paper, the loss is .mean(), and log_prob gap is computed with two independent set of samples
            loss = - (advantages * F.logsigmoid(self.args.ddo_beta * log_prob_gap_real)).mean() - self.args.ddo_alpha * F.logsigmoid(-self.args.ddo_beta * log_prob_gap_fake).mean()
            
            return loss
        else:
            raise NotImplementedError(self.args.loss)

    @torch.no_grad()
    def compute_log_prob_elbo(self, model, prompt_ids, completion_ids, completion_mask, repeated_size):
        """
        Compute the ELBO of a batch of samples
        Args:
            prompt_ids: [bs, prompt_length]
            completion_ids: [bs, gen_length]
            completion_mask: [bs, gen_length]
            repeated_size: int
        
        Return:
            log_prob_ref, log_prob_cur: [bs, seq_len], per-token log probability

        # TODO: maybe also consider EUBO as in https://arxiv.org/abs/2510.09541 and use the average?
        """
        batch_size = prompt_ids.shape[0]
        gen_length = completion_ids.shape[1]

        prompt_ids, completion_ids, completion_mask = [
            x.repeat([repeated_size, 1]) for x in [prompt_ids, completion_ids, completion_mask]]
        # [bs * num_replicates, prompt_length or gen_length]
        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1) # [bs * num_replicates, seq_len = prompt_length + gen_length]
        del prompt_ids, completion_ids

        if not self.args.loss_antithetic:
            lamda = torch.rand(input_ids.shape[0], device=model.device) # [bs * num_replicates]
            if self.args.loss_mask_prob_clamp: lamda = 0.1 + 0.9 * lamda # now in [0.1, 0.9]
            masked_index = torch.rand(input_ids.shape[0], gen_length, device=model.device) < lamda.unsqueeze(1) # [bs * num_replicates, gen_length]
        else:
            lamda = torch.rand(input_ids.shape[0] // 2, device=model.device) # [bs * num_replicates / 2]
            if self.args.loss_mask_prob_clamp: lamda = 0.1 + 0.9 * lamda # now in [0.1, 0.9]
            masked_index = torch.rand(input_ids.shape[0] // 2, gen_length, device=model.device) < lamda.unsqueeze(1) # [bs * num_replicates / 2, gen_length]
            masked_index = torch.cat([masked_index, ~masked_index], dim=0) # [bs * num_replicates, gen_length]
            
        full_masked_index = torch.full(input_ids.shape, False, device=model.device)
        full_masked_index[:, -gen_length:] = masked_index
        # [bs * num_replicates, seq_len], do not mask the prompt
        m = masked_index.sum(dim=-1).clamp(min=1) # [bs * num_replicates], number of masks in each row, clamped to >= 1
        del masked_index

        perturbed_input_ids = torch.where(full_masked_index, self.args.mask_id, input_ids)
        logits_cur = model(perturbed_input_ids).logits # [bs * num_replicates, seq_len, vocab_size]
        
        with self.accelerator.unwrap_model(model).disable_adapter():
            logits_ref = model(perturbed_input_ids).logits # [bs * num_replicates, seq_len, vocab_size]
        
        losses_cur = F.cross_entropy(input=logits_cur.view(-1, logits_cur.shape[-1]), target=input_ids.view(-1), reduction='none').view(logits_cur.shape[:-1])
        # [N := bs * num_replicates * seq_len, vocab_size], [N] -> [N] -> [bs * num_replicates, seq_len], don't require logits to be log-softmaxed
        losses_cur[~full_masked_index] = 0
        log_prob_cur = (-losses_cur).view(repeated_size, batch_size, -1).transpose(0, 1) # [bs, repeated_size, seq_len]
        t_weights = (gen_length / m).view(-1, batch_size, 1).transpose(0, 1) # [bs, repeated_size, 1]
        log_prob_cur = (t_weights * log_prob_cur).mean(dim = 1) # [bs, seq_len]
        
        losses_ref = F.cross_entropy(input=logits_ref.view(-1, logits_ref.shape[-1]), target=input_ids.view(-1), reduction='none').view(logits_ref.shape[:-1])
        losses_ref[~full_masked_index] = 0
        log_prob_ref = (-losses_ref).view(repeated_size, batch_size, -1).transpose(0, 1) # [bs, repeated_size, seq_len]
        t_weights = (gen_length / m).view(-1, batch_size, 1).transpose(0, 1) # [bs, repeated_size, 1]
        log_prob_ref = (t_weights * log_prob_ref).mean(dim = 1) # [bs, seq_len]
        
        return log_prob_ref, log_prob_cur


    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        r"""
        inputs has length = per_dev_train_batch_size (=: bs), which contains bs duplicates of a same question
        e.g., [{'question': 'xxx', 'answer': 'xxx', 'prompt': [{'content': '\nRespond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n\n\nxxx', 'role': 'user'}]}] * bs
        """
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        # self.processing_class is the tokenizer
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        # for all tensors, move to device and convert to dtype
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # generate roll-outs from the current model and compute the log rnd
        generation_batch_size = self.args.generation_batch_size
        prompt_completion_ids = []; log_prob_pre = []; log_prob_cur = []
        for i in range(0, prompt_ids.size(0), generation_batch_size):
            end_idx = min(i + generation_batch_size, prompt_ids.size(0))
            if self.args.sampler == 'roar':
                batch_prompt_completion_ids, batch_log_prob_pre, batch_log_prob_cur = self.generate_and_compute_log_rnd(
                    prompt=prompt_ids[i:end_idx],
                    gen_length=self.args.max_completion_length,
                    block_length=self.args.block_length,
                    temperature=self.args.temperature,
                    cfg_scale=self.args.cfg_scale,
                    mask_id=self.args.mask_id,
                ) # [mbs (= end_idx - i), seq_len (= prompt_length + gen_length)]; both [mbs, gen_length]
            else:
                batch_prompt_completion_ids = self.generate(
                    model=self.model,
                    prompt=prompt_ids[i:end_idx],
                    steps=self.args.sampler_steps,
                    gen_length=self.args.max_completion_length,
                    block_length=self.args.block_length,
                    temperature=self.args.temperature,
                    cfg_scale=self.args.cfg_scale,
                    remasking=self.args.sampler_remasking,
                    mask_id=self.args.mask_id,

                    threshold_pd=self.args.sampler_threshold_pd,
                    factor=self.args.sampler_factor,
                    
                    threshold_wino=self.args.sampler_threshold_wino,
                    threshold_wino_back=self.args.sampler_threshold_wino_back,
                ) # [mbs, seq_len]

            if self.args.compute_ref_log_prob_elbo:
                # approximate sequence log probabilities under both p_theta and p_pre by ELBO, used for computing advantage
                # note that we cannot do so when computing loss as the parameters may have been changed
                prompt_length_ref = prompt_ids.size(1)
                is_eos = batch_prompt_completion_ids[:, prompt_length_ref:] == self.processing_class.eos_token_id
                eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device) # [bs], default value gen_length
                eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
                # [bs], i.e., length before the first EOS token
                sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
                batch_completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
                # [bs, gen_length], everything after the first EOS token is False
            
                batch_log_prob_pre, batch_log_prob_cur = self.compute_log_prob_elbo(
                        model=self.model,
                        prompt_ids=batch_prompt_completion_ids[:, :prompt_length_ref],
                        completion_ids=batch_prompt_completion_ids[:, prompt_length_ref:],
                        completion_mask=batch_completion_mask,
                        repeated_size=self.args.compute_ref_log_prob_elbo_size)
                # both [mbs, gen_length]
                
            prompt_completion_ids.append(batch_prompt_completion_ids); log_prob_pre.append(batch_log_prob_pre); log_prob_cur.append(batch_log_prob_cur)
            del batch_prompt_completion_ids, batch_log_prob_pre, batch_log_prob_cur
            
        prompt_completion_ids = torch.cat(prompt_completion_ids, dim=0) # [bs, seq_len]
        log_prob_pre = torch.cat(log_prob_pre, dim=0) # [bs, gen_length]
        log_prob_cur = torch.cat(log_prob_cur, dim=0) # [bs, gen_length]
        torch.cuda.empty_cache()

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        del prompt_completion_ids

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device) # [bs], default value gen_length
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # [bs], i.e., length before the first EOS token
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        # [bs, gen_length], everything after the first EOS token is False

        if self.args.log_rnd_omit_eos:
            # do not count the log probabilities after the first EOS token
            log_prob_pre *= completion_mask; log_prob_cur *= completion_mask
        log_prob_pre = log_prob_pre.sum(dim=1); log_prob_cur = log_prob_cur.sum(dim=1) # [bs]

        ##### the following lines are copied from the original implementation #####
        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        # [bs, num_reward_funcs] -> [num_processes * bs, num_reward_funcs]
        ##### the above lines are copied from the original implementation #####

        log_prob_pre = gather(log_prob_pre); log_prob_cur = gather(log_prob_cur) # [bs] -> [num_processes * bs]
        log_rnds = log_prob_pre - log_prob_cur # [num_processes * bs]

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)  # [num_processes * bs]

        # Compute grouped-wise rewards, log_rnds, and advantages
        grouped_rewards = rewards.view(-1, len(prompts)) # [num_processes, bs]
        grouped_log_prob_pre = log_prob_pre.view(-1, len(prompts)) # [num_processes, bs]
        grouped_log_prob_cur = log_prob_cur.view(-1, len(prompts)) # [num_processes, bs]
        
        grouped_log_rnds = grouped_log_prob_pre - grouped_log_prob_cur
        
        if self.args.alpha != -1: # compute advantage as in (P)CE

            # TODO: also adaptive alpha?
            # choose coeff (fixed or adaptive based on ESS)
            coeffs = torch.full((grouped_rewards.shape[0], 1), self.args.coeff, device=device, dtype=grouped_rewards.dtype)
            if self.args.ada_coeff:
                # for each process (i.e., each prompt), compute coeff according to the ESS threshold
                for i in range(grouped_rewards.shape[0]):
                    coeffs[i, 0] = self.find_coeff(grouped_log_rnds[i], grouped_rewards[i], 
                                                   self.args.alpha, self.args.ada_coeff_ess_threshold)

            if self.args.alpha > 0.0:
                grouped_advantages = (coeffs * (grouped_log_rnds + grouped_rewards / self.args.alpha)).reshape(-1, self.args.num_generations).softmax(dim=-1).reshape(grouped_rewards.shape)
                negative_grouped_advantages = (coeffs * (grouped_log_rnds - grouped_rewards / self.args.alpha)).reshape(-1, self.args.num_generations).softmax(dim=-1).reshape(grouped_rewards.shape)
            else:
                grouped_advantages = (coeffs * grouped_rewards).reshape(-1, self.args.num_generations).softmax(dim=-1).reshape(grouped_rewards.shape) # [num_processes, bs]
                negative_grouped_advantages = (-coeffs * grouped_rewards).reshape(-1, self.args.num_generations).softmax(dim=-1).reshape(grouped_rewards.shape)
            ess = 1 / grouped_advantages.square().sum(dim=-1) / grouped_advantages.shape[-1] # [num_processes]
        else: # use the reward as the advantage
            grouped_advantages = grouped_rewards.clone() # [num_processes, bs]
            negative_grouped_advantages = -grouped_rewards.clone() # [num_processes, bs]
        
        
        advantages = grouped_advantages[self.accelerator.process_index] # [bs]
        negative_advantages = negative_grouped_advantages[self.accelerator.process_index] # [bs]
        log_prob_pre = grouped_log_prob_pre[self.accelerator.process_index] # [bs]
        log_prob_cur = grouped_log_prob_cur[self.accelerator.process_index] # [bs]
        coeff = coeffs[self.accelerator.process_index] # [1]

        # compute metrics for evaluation, all [num_processes]
        mean_grouped_rewards = grouped_rewards.mean(dim=1); std_grouped_rewards = grouped_rewards.std(dim=1)
        mean_grouped_log_rnds = grouped_log_rnds.mean(dim=1); std_grouped_log_rnds = grouped_log_rnds.std(dim=1)
        mean_grouped_advantages = grouped_advantages.mean(dim=1); std_grouped_advantages = grouped_advantages.std(dim=1)

        # Count prompts with zero std deviation (only for logging)
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["log_rnd"].append(mean_grouped_log_rnds.mean().item())
        self._metrics[mode]["log_rnd_std"].append(std_grouped_log_rnds.mean().item())
        self._metrics[mode]["advantage"].append(mean_grouped_advantages.mean().item())
        self._metrics[mode]["advantage_std"].append(std_grouped_advantages.mean().item())
        if self.args.alpha != -1:
            self._metrics[mode]["ess"].append(ess.mean().item())
            self._metrics[mode]["ess_std"].append(ess.std().item())
            if self.args.ada_coeff:
                self._metrics[mode]["coeff"].append(coeffs.mean().item())
                self._metrics[mode]["coeff_std"].append(coeffs.std().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()
            log_rnds_to_log = log_rnds.tolist()
            advantages_to_log = grouped_advantages.view(-1).tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompt=prompts_to_log,
                        completion=completions_to_log,
                        step=self.state.global_step,
                        reward=rewards_to_log,
                        log_rnd=log_rnds_to_log,
                        advantage=advantages_to_log,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    # For logging
                    table = {
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "step": [str(self.state.global_step)] * len(rewards),
                        "reward": rewards_to_log,
                        "log_rnd": log_rnds_to_log,
                        "advantage": advantages_to_log,
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids, # [bs, prompt_length]
            "prompt_mask": prompt_mask, # [bs, prompt_length]
            "completion_ids": completion_ids, # [bs, gen_length]
            "completion_mask": completion_mask, # [bs, gen_length]
            "advantages": advantages, # [bs]
            "negative_advantages": negative_advantages, # [bs]
            "log_prob_cur": log_prob_cur, # [bs]
            "coeff": coeff, # [1]
        }

    @torch.no_grad()
    def generate_and_compute_log_rnd(self, prompt, gen_length=256, block_length=32, temperature=1.0, cfg_scale=0., 
                                     mask_id=126336):
        r'''
        Use the current model \sim p_{\theta}, run block random-order autoregressive sampling,
        and compute log rnd: log p_{\theta_{pre}} / p_{\theta} (x) for x \sim p_{\theta}.

        Split the whole generation length `gen_length` into blocks with length block_length.
        In each block, use random-order autoregressive sampling to fill in the tokens.
        NFE = gen_length. block_length = 1 <=> AR; block_length = gen_length <=> random order AR

        Args:
            prompt: A tensor of shape [batch_size, prompt_length]
            gen_length: Generated answer length.
            block_length: Block length, must be divisible by gen_length.
            temperature: Categorical distribution sampling temperature.
            cfg_scale: Unsupervised classifier-free guidance scale.
            mask_id: The toke id of [MASK] is 126336.
        
        Return:
            x: final full samples generated under the current model
                shape [batch_size, prompt_length + gen_length]
            log_prob_pre, log_prob_cur: per-token log probabilities: log p_{\theta_{pre}} (x), log p_\theta (x)
                shape [batch_size, gen_length]
        '''
        batch_size = prompt.shape[0]; batch_arange = torch.arange(batch_size, device=self.model.device)
        x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()
        prompt_index = x != mask_id

        assert gen_length % block_length == 0; num_blocks = gen_length // block_length

        def get_cfg_logits(x, model):
            with torch.amp.autocast('cuda', enabled=self.args.fp16):
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits
                return logits.log_softmax(dim=-1)

        log_prob_pre_arr = torch.zeros((batch_size, gen_length), device=self.model.device, dtype=torch.bfloat16)
        log_prob_cur_arr = torch.zeros((batch_size, gen_length), device=self.model.device, dtype=torch.bfloat16)
        # store per-token log probs
        for blk in range(num_blocks):
            order = torch.rand((batch_size, block_length), device=self.model.device).argsort(
                dim=-1) + prompt.shape[1] + blk * block_length
            for d in range(block_length):
                logits = get_cfg_logits(x, self.model)
                logits_jump_pos = logits[batch_arange, order[:, d]] # [batch_size, vocab_size]
                update = self.sample_categorical_logits(logits_jump_pos, temperature=temperature) # [batch_size]
                log_prob_cur = logits_jump_pos[batch_arange, update] # [batch_size]
                log_prob_cur_arr[batch_arange, order[:, d] - prompt.shape[1]] = log_prob_cur
                if not self.args.compute_ref_log_prob_elbo:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        logits_pre = get_cfg_logits(x, self.model)
                    log_prob_pre = logits_pre[batch_arange, order[:, d], update] # [batch_size]
                    log_prob_pre_arr[batch_arange, order[:, d] - prompt.shape[1]] = log_prob_pre
                
                x[batch_arange, order[:, d]] = update

        torch.cuda.empty_cache()
        return x, log_prob_pre_arr, log_prob_cur_arr

    @staticmethod
    def sample_categorical_logits(logits, temperature=1.0, dtype=torch.float64):
        # do not require logits to be log-softmaxed
        if temperature == 0.0:
            return logits.argmax(dim=-1)
        gumbel_noise = -(1e-10 - (torch.rand_like(logits, dtype=dtype) + 1e-10).log()).log()
        return (logits + temperature * gumbel_noise).argmax(dim=-1)

    @staticmethod
    def find_coeff(log_rnds: torch.tensor, rewards: torch.tensor, alpha: float, ess_threshold: float, 
                   default_coeff: float = 0.01, coeff_range: tuple[float] = (0.01, 0.99)) -> float:
        r"""
        Args:
            log_rnds: log_rnd(x) = log p^{\theta_{pre}}/p^{\theta} (x), [bs]
            rewards: r(x), [bs]
            alpha: temperature, float in [0, \infty)
            ess_threshold: make sure ESS >= ess_threshold, float in (0, 1)
            default_coeff: if the optimizer fails (very unlikely), return this value, float in [0, 1]
        
        Return:
            coeff: float \in [0, 1] such that the ESS of the weights
            w(x_i) = 
                If alpha > 0: softmax(coeff * (log_rnd(x_j) + r(x_j) / alpha), 1<=j<=bs)_i
                If alpha = 0: softmax(coeff * r(x_j), 1<=j<=bs)_i
            is equal to ess_threshold.

            When alpha > 0, coeff = eta * alpha / (eta * alpha + 1) => eta = coeff / (1 - coeff) / alpha
            When alpha = 0, coeff = eta.

        Remark: optimizing eta seems to be less stable than optimizing coeff, though can unify both alpha > and = 0.
        """
        log_rnds = log_rnds.float().cpu().numpy(); rewards = rewards.float().cpu().numpy()

        def loss(coeff):
            if alpha > 0.0:
                weights = sp.special.softmax(coeff * (log_rnds + rewards / alpha), axis=0)
            else:
                weights = sp.special.softmax(coeff * rewards, axis=0)
            ess = 1 / np.sum(weights ** 2) / log_rnds.shape[0]
            return (ess_threshold - ess) ** 2
        result = sp.optimize.minimize(loss, x0=0.5, bounds=[coeff_range])

        if not result.success:
            logger.warning("Optimizer in `find_coeff` returned FAIL!\n"
                           f"Optimizer message: \n{result.message}\n"
                           f"log_rnds: {log_rnds}\n"
                           f"rewards: {rewards}\n"
                           f"Use default coeff {default_coeff} instead.")
            return default_coeff
        else:
            if result.x.item() < coeff_range[0] + 1e-8 or result.x.item() > coeff_range[1] - 1e-8:
                logger.warning(f"Optimizer in `find_coeff` returned coeff {result.x.item()} near the boundary of {coeff_range}, "
                               "which may require further investigation.\n"
                               f"log_rnds: {log_rnds}\n"
                               f"rewards: {rewards}\n")
            return result.x.item()
