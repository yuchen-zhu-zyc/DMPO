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
    temperature: float = field(
        default=0.9, # GRPOConfig default 0.9
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
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
        default=10,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm). "
                  "Refresh the buffer every num_iterations gradient updates."},
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=True,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )
    generation_batch_size: Optional[int] = field(
        default=4,
        metadata={
            "help": "Batch size for generation. If not set, the batch size will be equal to the number of generations."
        },
    )
    block_length: Optional[int] = field(
        default=32, # following d1
        metadata={"help": "diffusion block length"},
    )
    cfg_scale: Optional[float] = field(
        default=0.0,
    )
    remasking: Optional["str"] = field(
        default="low_confidence",
    )
    dataset: Optional[str] = field(
        default="gsm8k",
    )
    mask_id: int = field(
        default=126336,
        metadata={"help": "Mask token id. Default is from Llada"},
    )

    # Parameters that control uploading checkpoints
    upload_to_google_drive: bool = field(
        default=False,
        metadata={"help": "If True, upload the saved checkpoints to Google Drive"}
    )
    google_drive_parent_folder_id: str = field(
        default=None,
        metadata={"help": "Google Drive parent folder ID to upload checkpoints. "
                  "Default is None, which uploads to sub directories in the root directory. "
                  "Folder ID can be obtained from the URL when opening the folder in browser, not the complete URL. "
                  "Example: https://drive.google.com/drive/u/0/folders/google_drive_folder_id"}
    )

    #################### new params for DMPO ####################
    alpha: float = field(
        default=0.1,
        metadata={"help": r"If >= 0.0, temperature \alpha for CE loss, p_* \propto p_{pre} e^{r / \alpha}; "
                  "If = -1, this means use reward as the advantage"}
    )
    
    centering_strength: float = field(
        default=1,
        metadata={"help": "Strength of centering the advantage"}
    )

    # For compute_loss
    num_replicates: int = field(
        default=2,
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
        default= True, # True,
        metadata={"help": "If True, only apply mask on non EOS positions"}
    )

    # For _generate_and_score_completions
    ## computing log RND
    log_rnd_omit_eos: bool = field(
        default=False,
        metadata={"help": "Whether to omit the log RND after the first EOS token when computing advantage"}
    )
    log_rnd_normalize_by_length: bool = field(
        default=False,
        metadata={"help": "Whether to normalize the log RND by the true generation length, "
                  "i.e., the length of the generated sequence until the first EOS token"}
    )
    log_rnd_normalize_power: float = field(
        default=1.0,
        metadata={"help": "When log_rnd_normalize_by_length is True, the power to raise the true generation length to"}
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
        default=False,
        metadata={"help": "Whether to center the advantage for the completion of the same prompt "
                  "by subtracting the mean"}
    )
    advantage_centering_unbias: bool = field(
        default=False,
        metadata={"help": "Whether to unbias the advantage for the completion of the same prompt "
                  "by subtracting the mean of the advantages"}
    )
    
    advantage_centering_neg: bool = field(
        default=False,
        metadata={"help": "Whether to use the negative advantage for the completion of the same prompt "
                  "by subtracting the mean"}
    )
    
    advantage_centering_warmup: int = field(
        default=100,
        metadata={"help": "Number of steps to warm up the advantage centering before using the unbias version"}
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
    
    advantage_div_std: bool = field(
        default=False,
        metadata={"help": "Whether to normalize the advantage for the completion of the same prompt "
                  "by dividing by the standard deviation"}
    )
    advantage_div_eta: bool = field(
        default=False,
        metadata={"help": "Whether to normalize the advantage for the completion of the same prompt "
                  "by dividing by the eta. Only applies when ada_coeff and advantage_centering are True"}
    )
    advantage_div_eta_threshold: float = field(
        default=0.01,
        metadata={"help": "When advantage_div_eta is True, and when eta is below this threshold, "
                  "divide the loss by this value to maintain the reasonable range of the loss"}
    )