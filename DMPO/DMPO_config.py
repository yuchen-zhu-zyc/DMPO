from dataclasses import dataclass, field
from typing import Optional, Union
from trl.trainer.grpo_config import GRPOConfig

@dataclass
class DMPOConfig(GRPOConfig):
    r"""
    Configuration class for the [`DMPOTrainer`].

    Only the parameters specific to DMPO training or have different default values are listed here.
    For details on other parameters, refer to the [`~trl.trainer.grpo_config.GRPOConfig`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.
    """

    # Parameters that control generation
    generation_batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "Batch size for generation. If not set, the batch size will be equal to the number of generations."},
    )
    mask_id: int = field(
        default=126336,
        metadata={"help": "Mask token id. Default is from Llada"},
    )
    block_length: Optional[int] = field(
        default=32, # following d1
        metadata={"help": "diffusion block length"},
    )
    temperature: float = field(
        default=0.2, # GRPOConfig default 0.9
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    cfg_scale: Optional[float] = field(
        default=0.0,
        metadata={"help": "Classifier-free guidance scale. 0.0 means no guidance."},
    )

    use_fast_sampler: str = field(
        default="fast_dllm",
        metadata={"help": "Whether to use fast samplers with KV cache for training. "
                          "Can be 'fast_dllm', 'wino', or 'no' (use the default sampler in the model)."},
    )
    sampler: str = field(
        default="pd_cache_prefix",
        metadata={"help": "The sampler to use for generating roll-outs. Can be:\n"
                          "roar: (blockwise) random order autoregressive (default).\n"
                          "llada: the default sampler in LLaDA.\n"
                          "pd: confidence-aware parallel decoding as in Fast-dLLM (without KV cache).\n"
                          "pd_cache_prefix, pd_cache_dual: confidence-aware parallel decoding as in Fast-dLLM (with KV cache).\n"
                          "wino: Wide-In, Narrow-Out (with KV cache).\n"
                          },
    )

    sampler_steps: int = field(
        default=128,
        metadata={"help": "Number of steps for sampling in non-roar samplers, "
                          "including llada, pd, pd_cache_prefix, pd_cache_dual. "
                          "Note that max_completion_length = 256."}
    )
    sampler_remasking: str = field(
        default="low_confidence",
        metadata={"help": "Remasking strategy for non-roar samplers, "
                          "including llada, pd, pd_cache_prefix, pd_cache_dual. "
                          "Can be 'low_confidence' or 'random'."},
    )

    sampler_threshold_pd: Optional[float] = field(
        default=None,
        metadata={"help": "Confidence threshold, only for pd, pd_cache_prefix, pd_cache_dual."}
    )
    sampler_factor: Optional[int] = field(
        default=None,
        metadata={"help": "Factor, only for pd, pd_cache_prefix, pd_cache_dual."}
    )

    sampler_threshold_wino: float = field(
        default=0.6,
        metadata={"help": "Confidence threshold, only for wino."}
    )
    sampler_threshold_wino_back: float = field(
        default=0.9,
        metadata={"help": "Confidence threshold, only for wino."}
    )


    # Parameters that control the data preprocessing
    max_prompt_length: Optional[int] = field(
        default=200, # following d1
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    pretrained_model_path: Optional[str] = field(
        default="GSAI-ML/LLaDA-8B-Instruct",
        metadata={
            "help": "Path to the model to be used for training. This can be a local path or a Hugging Face Hub model ID."
        },
    )

    # Parameters that control the training
    num_iterations: int = field(
        default=8,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm). "
                  "Refresh the buffer every num_iterations gradient updates."},
    )
    dataset: Optional[str] = field(
        default="gsm8k",
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=True,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )

    #################### new params for DMPO ####################
    alpha: float = field(
        default=0.04,
        metadata={"help": r"If >= 0.0, temperature \alpha for CE loss, p_* \propto p_{pre} e^{r / \alpha}; "
                  "If = -1, this means use reward as the advantage"}
    )
    
    centering_strength: float = field(
        default=1,
        metadata={"help": "Strength of centering the advantage"}
    )

    # For compute_loss
    num_replicates: int = field(
        default=4,
        metadata={"help": "Number of replicates of each roll-out for CE loss"}
    )
    loss_antithetic: bool = field(
        default=True,
        metadata={"help": "Whether to use antithetic masking when computing the CE loss. "
                  "If True, for each completion, sample num_replicates // 2 pairs of masked completions that are complementary."}
    )
    loss_mask_prob_clamp: bool = field(
        default=True,
        metadata={"help": "Whether to clamp the mask probabilities to the range [0.1, 0.9] when computing loss"}
    )
    loss_mask_non_eos: bool = field(
        default= False, # True,
        metadata={"help": "If True, only apply mask on non EOS positions"}
    )

    # For _generate_and_score_completions
    ## computing log RND
    log_rnd_omit_eos: bool = field(
        default=False,
        metadata={"help": "Whether to omit the log RND after the first EOS token when computing advantage"}
    )
    ## proximal step
    coeff: float = field(
        default=1.0,
        metadata={"help": "Default value of coeff when ada_coeff is False (1 means infinite stepsize eta)"}
    )
    ada_coeff: bool = field(
        default=False,
        metadata={"help": "If True, use adaptive coeff (related with the step size \eta) based on ESS instead of the default one above"}
    )
    ada_coeff_ess_threshold: float = field(
        default=0.8,
        metadata={"help": "Choose adaptive coeff such that the ESS is equal to this value "
                  "Solve by minimizing L2 loss"}
    )

    ## computing advantage
    advantage_centering: bool = field(
        default=True,
        metadata={"help": "Whether to center the advantage for the completion of the same prompt "
                  "by subtracting the mean"}
    )
    advantage_centering_unbias: bool = field(
        default=False,
        metadata={"help": "Whether to unbias the advantage for the completion of the same prompt "
                  "by subtracting the mean of the advantages"}
    )
    
    advantage_centering_neg: bool = field(
        default=True,
        metadata={"help": "Whether to use the negative advantage for the completion of the same prompt "
                  "by subtracting the mean"}
    )

    
    compute_ref_log_prob_elbo: bool = field(
        default=True,
        metadata={"help": "Whether to compute the reference log probability for the ELBO loss"}
    )
    
    compute_ref_log_prob_elbo_size: int = field(
        default=4,
        metadata={"help": "Batch size of the reference log probability for the ELBO loss"}
    )
    
    use_sft_model: bool = field(
        default=False,
        metadata={"help": "Whether to use the SFT model as the reference model"}
    )
    sft_model_path: str = field(
        default=None,
        metadata={"help": "Path to the SFT model to be used as the reference model"}
    )
    
    ## computing loss
    loss: str = field(
        default="wdce",
        metadata={"help": "Loss function to use. Can be 'wdce' or 'ddo'"}
    )
    ddo_alpha: float = field(
        default=1.0,
        metadata={"help": "Alpha parameter for DDO loss"}
    )
    ddo_beta: float = field(
        default=1.0,
        metadata={"help": "Beta parameter for DDO loss"}
    )
    ddo_indep_set: bool = field(
        default=False,
        metadata={"help": "Whether to use two independent sets of samples for DDO loss"}
    )
    