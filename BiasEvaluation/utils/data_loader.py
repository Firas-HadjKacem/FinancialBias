import os
import gc
import random
import traceback
import multiprocessing
from typing import Any, Dict, List, Union

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import set_seed
from .helpers import load_dataset, build_full_prompt 
from ..core.models import load_bert_model, load_llama_model, LlamaModelWrapper, BERTModel
from ..analysis.bias_analyzer import BiasAnalyzer

# Column names expected in the mutant header
COL_ORIGINAL = "Original"
COL_MUTANT1 = "Mutant_Word_1"
COL_MUTANT2 = "Mutant_Word_2"
COL_INTER = "Mutant_Intersectional"

DEFAULT_HEADER = [
    'Word 1', 'Replacement 1', 'Word 2', 'Replacement 2',
    'Mutant_Word_1', 'Mutant_Word_2', 'Mutant_Intersectional',
    'Index', 'Original', 'Similarity'
]

NO_SHAP_COLUMNS = [
    "OriginalPrediction", "OriginalProbs",
    "Mutant1Prediction", "Mutant1Probs",
    "Mutant2Prediction", "Mutant2Probs",
    "IntersectionPrediction", "IntersectionProbs",
    "Bias_Intersection", "Bias_Mutant1", "Bias_Mutant2"
]

SHAP_COLUMNS = [
    "OriginalPrediction", "OriginalProbs", "OriginalShapley", "OriginalBiasRanks",
    "Mutant1Prediction", "Mutant1Probs", "Mutant1Shapley", "Mutant1BiasRanks",
    "Mutant2Prediction", "Mutant2Probs", "Mutant2Shapley", "Mutant2BiasRanks",
    "IntersectionPrediction", "IntersectionProbs", "IntersectionShapley", "IntersectionBiasRanks",
    "Bias_Intersection", "Bias_Mutant1", "Bias_Mutant2"
]

PROMPT_PREFIX = (
    "Analyze the sentiment of this statement extracted from a financial news article. "
    "Provide your answer as either negative, positive, or neutral.. Text: "
)
PROMPT_SUFFIX = ".. Answer: "

def _resolve_mutant_indices(header: List[str]) -> Dict[str, int]:
    """
    Resolve the indices of the Original and mutant columns from the header,
    instead of hard-coding numeric positions.
    """
    required = {
        "original": COL_ORIGINAL,
        "mutant1": COL_MUTANT1,
        "mutant2": COL_MUTANT2,
        "inter": COL_INTER,
    }
    indices: Dict[str, int] = {}
    for key, col_name in required.items():
        if col_name not in header:
            raise KeyError(
                f"Required column '{col_name}' not found in header {header}. "
                f"Please check the stored metadata['header'] or update column names."
            )
        indices[key] = header.index(col_name)
    return indices

# Common helpers

def _load_model_and_analyzer(
    model_config: Union[str, Dict[str, Any]],
    model_type: str,
    batch_size: int
) -> BiasAnalyzer:
    """
    Load model + tokenizer and wrap them in a BiasAnalyzer, for both BERT and LLaMA.

    model_type:
        - "bert":  model_config is typically a HuggingFace model name string
        - "llama": model_config is a dict with keys:
                   {"base_tokenizer_id", "model_id", "label_ids", ["cache_dir"]}
    """
    if model_type == "bert":
        if isinstance(model_config, str):
            model_name = model_config
        else:
            # allow {"model_name": "..."} style if ever used
            model_name = model_config["model_name"]
        model, tokenizer = load_bert_model(model_name)
        wrapper = BERTModel(model, tokenizer)

    elif model_type == "llama":
        base_tokenizer_id = model_config["base_tokenizer_id"]
        model_id = model_config["model_id"]
        label_ids = model_config["label_ids"]
        cache_dir = model_config.get("cache_dir", None)

        model, tokenizer = load_llama_model(base_tokenizer_id, model_id, cache_dir)
        wrapper = LlamaModelWrapper(model, tokenizer, label_ids)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    analyzer = BiasAnalyzer(
        wrapper,
        tokenizer,
        model_type=model_type,
        splitter_type="string",
        batch_size=batch_size,
        is_wrapped=True,
    )
    return analyzer

# Per-file processing (no SHAP)

def process_file_no_shap(
    file_path: str,
    output_file: str,
    analyzer: BiasAnalyzer,
    batch_size: int = 64,
    gpu_id: int = 0
) -> bool:
    """
    Process a single mutant file without SHAP, using batched predictions for both BERT and LLaMA models.
    """

    # Load dataset
    try:
        metadata, mutants = load_dataset(file_path)
    except Exception as e:
        print(f"[GPU {gpu_id}] Error loading {file_path}: {str(e)}")
        return False

    # Resolve header and column indices dynamically
    original_header = metadata.get("header", DEFAULT_HEADER)
    idx = _resolve_mutant_indices(original_header)

    # Cache prompts by unique text to avoid repeated inference
    prompts_by_text: Dict[str, str] = {}
    all_prompts: List[str] = []  # aligned with texts list
    texts: List[str] = []

    for row in mutants:
        for col_key in ("original", "mutant1", "mutant2", "inter"):
            text = row[idx[col_key]].strip()
            if not text:
                continue
            if text not in prompts_by_text:
                prompt = build_full_prompt(text, PROMPT_PREFIX, PROMPT_SUFFIX)
                prompts_by_text[text] = prompt
                texts.append(text)
                all_prompts.append(prompt)

    # Run model in batches (generate_batch if available, otherwise generate loop)
    prompt_to_result: Dict[str, Dict[str, Any]] = {}
    print(f"[GPU {gpu_id}] Predicting all {len(all_prompts)} prompts...")

    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i + batch_size]

        if hasattr(analyzer.model_wrapper, "generate_batch"):
            batch_results = analyzer.model_wrapper.generate_batch(batch_prompts)
        else:
            batch_results = [
                analyzer.model_wrapper.generate(p) for p in batch_prompts
            ]

        for prompt, res in zip(batch_prompts, batch_results):
            prompt_to_result[prompt] = res

    # Build output rows
    new_header = ["ID"] + original_header + NO_SHAP_COLUMNS
    results: List[List[Any]] = []

    for i, row in enumerate(tqdm(mutants, desc=f"[GPU {gpu_id}] Processing mutants")):
        original_text = row[idx["original"]].strip()
        mutant1_text = row[idx["mutant1"]].strip()
        mutant2_text = row[idx["mutant2"]].strip()
        inter_text = row[idx["inter"]].strip()

        original_prompt = prompts_by_text[original_text]
        mutant1_prompt = prompts_by_text[mutant1_text]
        mutant2_prompt = prompts_by_text[mutant2_text]
        inter_prompt = prompts_by_text[inter_text]

        orig_res = prompt_to_result[original_prompt]
        m1_res = prompt_to_result[mutant1_prompt]
        m2_res = prompt_to_result[mutant2_prompt]
        inter_res = prompt_to_result[inter_prompt]

        orig_pred = orig_res["label"]
        m1_pred = m1_res["label"]
        m2_pred = m2_res["label"]
        inter_pred = inter_res["label"]

        orig_probs = orig_res["probabilities"]
        m1_probs = m1_res["probabilities"]
        m2_probs = m2_res["probabilities"]
        inter_probs = inter_res["probabilities"]

        bias_inter = (orig_pred.lower() != inter_pred.lower())
        bias_m1 = (orig_pred.lower() != m1_pred.lower())
        bias_m2 = (orig_pred.lower() != m2_pred.lower())

        new_row = [i + 1] + row + [
            orig_pred, orig_probs,
            m1_pred, m1_probs,
            m2_pred, m2_probs,
            inter_pred, inter_probs,
            bias_inter, bias_m1, bias_m2
        ]
        results.append(new_row)

    df = pd.DataFrame(results, columns=new_header)
    df.to_excel(output_file, index=False)
    print(f"[GPU {gpu_id}] Results stored in {output_file}")

    return True


# Per-file processing (with SHAP)

def process_file_with_shap(
    file_path: str,
    output_file: str,
    analyzer: BiasAnalyzer,
    sampling_ratio: float = 0.3,
    max_combinations: int = 500,
    gpu_id: int = 0
) -> bool:
    """
    Process a single mutant file with TokenSHAP analysis, using batched predictions for both BERT and LLaMA models.
    """

    # Load dataset
    try:
        metadata, mutants = load_dataset(file_path)
    except Exception as e:
        print(f"[GPU {gpu_id}] Error loading {file_path}: {str(e)}")
        return False

    original_header = metadata.get("header", DEFAULT_HEADER)
    idx = _resolve_mutant_indices(original_header)

    new_header = ["ID"] + original_header + SHAP_COLUMNS
    results: List[List[Any]] = []

    for i, row in enumerate(tqdm(mutants, desc=f"[GPU {gpu_id}] Processing mutants with TokenSHAP")):
        original_text = row[idx["original"]].strip()
        mutant1_text = row[idx["mutant1"]].strip()
        mutant2_text = row[idx["mutant2"]].strip()
        inter_text = row[idx["inter"]].strip()

        try:
            orig_results = analyzer.analyze_sentence(original_text, sampling_ratio, max_combinations)
            m1_results = analyzer.analyze_sentence(mutant1_text, sampling_ratio, max_combinations)
            m2_results = analyzer.analyze_sentence(mutant2_text, sampling_ratio, max_combinations)
            inter_results = analyzer.analyze_sentence(inter_text, sampling_ratio, max_combinations)

            orig_pred = orig_results['prediction']['label']
            m1_pred = m1_results['prediction']['label']
            m2_pred = m2_results['prediction']['label']
            inter_pred = inter_results['prediction']['label']

            orig_probs = orig_results['prediction']['probabilities']
            m1_probs = m1_results['prediction']['probabilities']
            m2_probs = m2_results['prediction']['probabilities']
            inter_probs = inter_results['prediction']['probabilities']

            orig_shapley = orig_results['Shapley Values']
            m1_shapley = m1_results['Shapley Values']
            m2_shapley = m2_results['Shapley Values']
            inter_shapley = inter_results['Shapley Values']

            orig_bias_ranks = orig_results.get('Bias Token Ranks', {})
            m1_bias_ranks = m1_results.get('Bias Token Ranks', {})
            m2_bias_ranks = m2_results.get('Bias Token Ranks', {})
            inter_bias_ranks = inter_results.get('Bias Token Ranks', {})

            bias_inter = (orig_pred.lower() != inter_pred.lower())
            bias_m1 = (orig_pred.lower() != m1_pred.lower())
            bias_m2 = (orig_pred.lower() != m2_pred.lower())

            new_row = [i + 1] + row + [
                orig_pred, orig_probs, orig_shapley, orig_bias_ranks,
                m1_pred, m1_probs, m1_shapley, m1_bias_ranks,
                m2_pred, m2_probs, m2_shapley, m2_bias_ranks,
                inter_pred, inter_probs, inter_shapley, inter_bias_ranks,
                bias_inter, bias_m1, bias_m2
            ]
            results.append(new_row)

        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing row {i}: {str(e)}")
            traceback.print_exc()
            error_row = [i + 1] + row + [
                "ERROR", {}, {}, {},
                "ERROR", {}, {}, {},
                "ERROR", {}, {}, {},
                "ERROR", {}, {}, {},
                False, False, False
            ]
            results.append(error_row)

    df = pd.DataFrame(results, columns=new_header)
    df.to_excel(output_file, index=False)
    print(f"[GPU {gpu_id}] Results with Shapley values stored in {output_file}")

    return True

# Per-GPU processing (multi-file)

def process_files_on_gpu(
    files: List[str],
    gpu_id: int,
    model_config: Union[str, Dict[str, Any]],
    directory: str,
    output_dir: str,
    batch_size: int,
    sampling_ratio: float = 0.3,
    max_combinations: int = 500,
    shap: bool = False,
    model_type: str = "llama"
) -> Dict[str, Any]:
    """
    Process a subset of files on a specific GPU, for both BERT and LLaMA models.
    """

    # Reproducibility per process
    set_seed(42 + gpu_id)
    random.seed(42 + gpu_id)
    np.random.seed(42 + gpu_id)

    # Restrict this process to one visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)  # within this proc, cuda:0 is the assigned GPU
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    os.makedirs(output_dir, exist_ok=True)

    print(f"[GPU {gpu_id}] Loading model ({model_type})...")
    analyzer = _load_model_and_analyzer(model_config, model_type=model_type, batch_size=batch_size)

    successful = 0
    failed = 0
    skipped = 0

    for file_idx, file_name in enumerate(files):
        try:
            file_path = os.path.join(directory, file_name)
            output_file = os.path.join(output_dir, file_name.replace(".pkl", ".xlsx"))

            if os.path.exists(output_file):
                print(f"[GPU {gpu_id}] Skipping already processed file: {file_name}")
                skipped += 1
                continue

            print(f"[GPU {gpu_id}] Processing file {file_idx + 1}/{len(files)}: {file_name}")

            if shap:
                success = process_file_with_shap(
                    file_path,
                    output_file,
                    analyzer,
                    sampling_ratio=sampling_ratio,
                    max_combinations=max_combinations,
                    gpu_id=gpu_id,
                )
            else:
                success = process_file_no_shap(
                    file_path,
                    output_file,
                    analyzer,
                    batch_size=batch_size,
                    gpu_id=gpu_id,
                )

            if success:
                successful += 1
                print(f"[GPU {gpu_id}] Successfully processed {file_name}")
            else:
                failed += 1
                print(f"[GPU {gpu_id}] Failed to process {file_name}")

            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[GPU {gpu_id}] Unexpected error processing {file_name}: {e}")
            traceback.print_exc()
            failed += 1

    return {
        "gpu_id": gpu_id,
        "total": len(files),
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
    }

# Multi-GPU orchestration

def process_mutant_directory_multi_gpu(
    directory: str,
    model_config: Union[str, Dict[str, Any]],
    output_dir: str = None,
    sampling_ratio: float = 0.3,
    max_combinations: int = 500,
    batch_size: int = 16,
    num_gpus: int = 3,
    shap: bool = False,
    model_type: str = "llama",
) -> Dict[str, Any]:
    """
    Unified multi-GPU entrypoint for both BERT and LLaMA models.

    usage examples:
        # BERT (no SHAP)
        process_mutant_directory_multi_gpu(
            directory="...",
            model_config="ProsusAI/finbert",
            model_type="bert",
            shap=False,
        )

        # LLaMA (with SHAP)
        llama_cfg = {
            "base_tokenizer_id": "meta-llama/Llama-2-7b-chat-hf",
            "model_id": "oliverwang15/FinGPT_v32_Llama2_Sentiment_Instruction_LoRA_FT",
            "label_ids": {
                "Positive": [6374],
                "Negative": [8178, 22198],
                "Neutral": [21104],
            },
        }
        process_mutant_directory_multi_gpu(
            directory="...",
            model_config=llama_cfg,
            model_type="llama",
            shap=True,
        )
    """

    available_gpus = torch.cuda.device_count()
    if num_gpus > available_gpus:
        print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available")
        num_gpus = available_gpus

    print(f"Using {num_gpus} GPUs for processing ({model_type}, shap={shap})")

    if output_dir is None:
        output_dir = directory + ("_shap_results" if shap else "_results")

    os.makedirs(output_dir, exist_ok=True)

    # Find all .pkl files
    pkl_files = [f for f in os.listdir(directory) if f.endswith(".pkl")]
    print(f"Found {len(pkl_files)} mutant files in {directory}")

    # Sort by size (largest first) for better balancing
    file_sizes = [
        (filename, os.path.getsize(os.path.join(directory, filename)))
        for filename in pkl_files
    ]
    file_sizes.sort(key=lambda x: x[1], reverse=True)

    files_per_gpu: List[List[str]] = [[] for _ in range(num_gpus)]
    for i, (filename, _) in enumerate(file_sizes):
        files_per_gpu[i % num_gpus].append(filename)

    for i in range(num_gpus):
        total_size_mb = sum(size for fn, size in file_sizes if fn in files_per_gpu[i]) / (1024 * 1024)
        print(f"GPU {i}: {len(files_per_gpu[i])} files, {total_size_mb:.2f} MB total")

    # Spawn one process per GPU
    processes = []
    for gpu_id in range(num_gpus):
        p = multiprocessing.Process(
            target=process_files_on_gpu,
            args=(
                files_per_gpu[gpu_id],
                gpu_id,
                model_config,
                directory,
                output_dir,
                batch_size,
                sampling_ratio,
                max_combinations,
                shap,
                model_type,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    processed_files = len([f for f in os.listdir(output_dir) if f.endswith(".xlsx")])

    print("\nAll mutant files processed across all GPUs!")
    print(f"Total files: {len(pkl_files)}")
    print(f"Successfully processed: {processed_files}")
    print(f"Skipped: {len(pkl_files) - processed_files}")

    return {"total": len(pkl_files), "processed": processed_files}