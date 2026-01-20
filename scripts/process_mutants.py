import sys
import os
import argparse
import time
import random
import multiprocessing as mp
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login

# Environment / HF login

load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print(
        "[WARNING] HF_TOKEN not found in environment. "
        "Private / gated models may fail to load."
    )

# Add project root to Python path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from BiasEvaluation.utils.data_loader import process_mutant_directory_multi_gpu  # noqa: E402

# Reproducibility

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run bias evaluation on a directory of mutant files using "
            "BERT or LLaMA models with optional word-level explanations."
        )
    )

    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["bert", "llama"],
        required=True,
        help="Type of model to use: 'bert' (sequence classifier) or 'llama' (causal LM).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help=(
            "Hugging Face model ID. For 'bert', this is the classifier model "
            "(e.g., 'ProsusAI/finbert'). For 'llama', this is the causal LM "
            "(e.g., 'oliverwang15/FinGPT_v32_Llama2_Sentiment_Instruction_LoRA_FT')."
        ),
    )
    parser.add_argument(
        "--base_tokenizer_id",
        type=str,
        default=None,
        help=(
            "Base tokenizer ID for LLaMA models "
            "(e.g., 'meta-llama/Llama-2-7b-chat-hf'). "
            "Required if --model_type=llama."
        ),
    )

    # Label IDs for LLaMA sentiment decoding
    parser.add_argument(
        "--positive_ids",
        type=str,
        default=None,
        help=(
            "Comma-separated token IDs that decode to 'positive' "
            "(required for --model_type=llama; e.g., '6374')."
        ),
    )
    parser.add_argument(
        "--negative_ids",
        type=str,
        default=None,
        help=(
            "Comma-separated token IDs that decode to 'negative' "
            "(required for --model_type=llama; e.g., '8178,22198')."
        ),
    )
    parser.add_argument(
        "--neutral_ids",
        type=str,
        default=None,
        help=(
            "Comma-separated token IDs that decode to 'neutral' "
            "(required for --model_type=llama; e.g., '21104')."
        ),
    )

    # Data / IO
    parser.add_argument(
        "--mutants_dir",
        type=str,
        required=True,
        help="Directory containing mutant .pkl files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where result .xlsx files will be written.",
    )

    # SHAP / sampling / batching / GPUs
    parser.add_argument(
        "--sampling_ratio",
        type=float,
        default=0.3,
        help="Fraction of SHAP combinations to sample (used when --shap is enabled).",
    )
    parser.add_argument(
        "--max_combinations",
        type=int,
        default=1000,
        help="Maximum number of SHAP combinations per sentence.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for batched inference.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for multi-process evaluation.",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Enable TokenSHAP explanations (if set, SHAP attributions will be computed).",
    )

    return parser.parse_args()


def _parse_id_list(s: str):
    """Parse comma-separated token IDs into a list of ints."""
    return [int(x) for x in s.split(",") if x.strip()] if s is not None else []


def main():
    args = parse_args()

    # Build model_config depending on model_type
    if args.model_type == "bert":
        model_config = args.model_name
    else:  # llama
        if not args.base_tokenizer_id:
            raise ValueError(
                "For --model_type=llama, --base_tokenizer_id must be provided "
                "(e.g., 'meta-llama/Llama-2-7b-chat-hf')."
            )

        pos_ids = _parse_id_list(args.positive_ids)
        neg_ids = _parse_id_list(args.negative_ids)
        neu_ids = _parse_id_list(args.neutral_ids)

        if not pos_ids or not neg_ids or not neu_ids:
            raise ValueError(
                "For --model_type=llama, --positive_ids, --negative_ids and "
                "--neutral_ids must all be provided. "
                "Use utils/find_label_ids.py to discover them."
            )

        label_ids = {
            "Positive": pos_ids,
            "Negative": neg_ids,
            "Neutral": neu_ids,
        }

        model_config = {
            "base_tokenizer_id": args.base_tokenizer_id,
            "model_id": args.model_name,
            "label_ids": label_ids,
        }

    mutants_dir = args.mutants_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("\n=== BiasEvaluation: process_mutants ===")
    print(f"Model type      : {args.model_type}")
    print(f"Model name      : {args.model_name}")
    if args.model_type == "llama":
        print(f"Tokenizer       : {args.base_tokenizer_id}")
        print(f"Label IDs       : {model_config['label_ids']}")
    print(f"Mutants dir     : {mutants_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Sampling ratio  : {args.sampling_ratio}")
    print(f"Max combinations: {args.max_combinations}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Num GPUs        : {args.num_gpus}")
    print(f"SHAP enabled    : {args.shap}")
    print("=======================================\n")

    mp.set_start_method("spawn", force=True)

    start_time = time.time()
    process_mutant_directory_multi_gpu(
        directory=mutants_dir,
        model_config=model_config,
        output_dir=output_dir,
        sampling_ratio=args.sampling_ratio,
        max_combinations=args.max_combinations,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        shap=args.shap,
        model_type=args.model_type,
    )
    duration = time.time() - start_time
    print(f"\nExecution time: {duration/60:.2f} minutes")


if __name__ == "__main__":
    main()