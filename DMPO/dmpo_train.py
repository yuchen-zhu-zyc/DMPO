import os
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from peft import LoraConfig

# Custom imports
from dmpo_trainer import DMPOTrainer
from DMPO_config import DMPOConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
)
from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    set_random_seed,
    get_math_questions,
)

def main(dmpo_config, model_config):

    # Set seed for reproducibility
    set_random_seed(dmpo_config.seed)

    # Load dataset based on configuration
    if dmpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func, # [0, 0.5], can be smaller due to penalty
            soft_format_reward_func, # {0, 0.5}
            strict_format_reward_func, # {0, 0.5}
            int_reward_func, # {0, 0.5}
            correctness_reward_func, # {0, 2}
        ] # this implies rewards should typically between [0, 4] (smaller than 0 is possible)
    elif dmpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    elif dmpo_config.dataset == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
    elif dmpo_config.dataset == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=dmpo_config.seed)

    # Split dataset if needed
    if dmpo_config.dataset in ["countdown", "sudoku"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
        train_set = dataset

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        dmpo_config.pretrained_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        cache_dir=os.environ['HF_HOME']
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        dmpo_config.pretrained_model_path,
        trust_remote_code=True,
        cache_dir=os.environ['HF_HOME']
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )

    # Configure Google Drive uploader callback if required
    if dmpo_config.upload_to_google_drive:
        from drive_utils import GoogleDriveUploaderCallback
        callback = [GoogleDriveUploaderCallback(dmpo_config)]
    else:
        callback = None

    # Initialize and run trainer
    trainer = DMPOTrainer(
        args=dmpo_config,
        model=model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
        callbacks=callback,
    )

    trainer.train(resume_from_checkpoint=dmpo_config.resume_from_checkpoint)


if __name__ == "__main__":
    parser = TrlParser((DMPOConfig, ModelConfig))
    dmpo_config, model_config = parser.parse_args_and_config()
    main(dmpo_config=dmpo_config, model_config=model_config)
