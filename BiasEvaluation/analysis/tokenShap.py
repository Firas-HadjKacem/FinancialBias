# SPDX-FileCopyrightText: 2023-2024 The TokenSHAP Authors
#
# Modifications for this project:
# - Adapted the value function to use Jensen–Shannon distance (JSD) between the
#   baseline and combination probability distributions instead of cosine or
#   logit-based similarity.
# - Added a second metric based on the probability of the baseline-predicted class.
# - Normalized Shapley values separately per metric and exposed a
#   `get_sim_shapley_values()` helper tailored to our bias-analysis pipeline.
# - Integrated with financial sentiment prompts and wrappers (`BERTModel`,
#   `LlamaModelWrapper`) via `build_full_prompt` and the unified model interface.

import pandas as pd
import matplotlib.pyplot as plt
import re
from typing import List, Dict, Optional, Tuple, Any
from tqdm.auto import tqdm
from collections import defaultdict
from BiasEvaluation.core.base import ModelBase, BaseSHAP
from BiasEvaluation.core.splitters import Splitter
from BiasEvaluation.utils.helpers import build_full_prompt, jensen_shannon_distance

class TokenSHAP(BaseSHAP):
    """
    Analyzes token importance in text prompts using SHAP values.

    This variant uses probability-distribution similarity based on
    Jensen–Shannon distance (JSD) between the baseline and combination
    predictions, plus a baseline-class probability metric. Both metrics
    are used to derive normalized Shapley values per token.
    """
    def __init__(self,
                 model: ModelBase,
                 splitter: Splitter,
                 debug: bool = False,
                 batch_size=16):
        """
        Initialize TokenSHAP
       
        Args:
            model: Model to analyze
            splitter: Text splitter implementation
            debug: Enable debug output
        """
        super().__init__(model, debug)
        self.splitter = splitter
        self.prompt_prefix = ""  
        self.prompt_suffix = ""  
        self.batch_size = batch_size

    def _get_samples(self, content: str) -> List[str]:
        """Get tokens from prompt"""
        return self.splitter.split(content)
   
    def _get_combination_key(self, combination: List[str], indexes: Tuple[int, ...]) -> str:
        return self.splitter.join(combination)

    def _prepare_combination_args(self, combination: List[str], original_content: str) -> Dict:
        prompt = f"{self.prompt_prefix}{self.splitter.join(combination)}{self.prompt_suffix}"
        return {"prompt": prompt}

    def _get_result_per_combination(self, content, sampling_ratio=0.0, max_combinations=None):
        """
        Get model responses for combinations with batch processing
        
        Args:
            content: Original content
            sampling_ratio: Ratio of combinations to sample
            max_combinations: Maximum number of combinations
            
        Returns:
            Dictionary mapping combination keys to response data
        """
        samples = self._get_samples(content)
        combinations = self._get_all_combinations(samples, sampling_ratio, max_combinations)
        
        # Prepare prompts for batch processing
        prompts = []
        comb_keys = []
        comb_indices = []
        
        for key, (combination, indices) in combinations.items():
            #Call with both parameters and extract prompt from returned dict
            comb_args = self._prepare_combination_args(combination, content)
            prompt = comb_args["prompt"]  # Extract prompt from dict
            
            prompts.append(prompt)
            comb_keys.append(key)
            comb_indices.append(indices)
        
        # Batching with error handling
        all_results = []
        for batch_start in range(0, len(prompts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            try:
                batch_results = self.model.generate_batch(batch_prompts)
                all_results.extend(batch_results)
            except RuntimeError as e:
                if "stack expects each tensor to be equal size" in str(e):
                    print(f"Error in batch {batch_start//self.batch_size}: {str(e)}")
                    print("Falling back to individual processing for this batch")
                    # Fall back to individual processing with generate
                    for prompt in batch_prompts:
                        try:
                            single_result = self.model.generate(prompt)
                            all_results.append(single_result)
                        except Exception as inner_e:
                            print(f"Individual processing also failed: {str(inner_e)}")
                            # Provide fallback result with default values
                            all_results.append({
                                "label": "NA",  
                                "probabilities": {"Positive": 0.33, "Negative": 0.33, "Neutral": 0.34}
                            })
                else:
                    # Re-raise other RuntimeErrors
                    raise
            except Exception as other_e:
                # Handle any other exceptions during batch processing
                print(f"Unexpected error in batch {batch_start//self.batch_size}: {str(other_e)}")
                # Fall back to individual processing
                for prompt in batch_prompts:
                    try:
                        single_result = self.model.generate(prompt)
                        all_results.append(single_result)
                    except Exception:
                        # Provide fallback result
                        all_results.append({
                            "label": "NA",
                            "probabilities": {"Positive": 0.33, "Negative": 0.33, "Neutral": 0.34}
                        })
        
        # Attach back to combination keys
        results = {}
        for i, key in enumerate(comb_keys):
            results[key] = {
                "combination": combinations[key][0],
                "indices": comb_indices[i],
                "response": all_results[i]
            }
        
        return results 

    def _get_df_per_combination(self, responses: Dict[str, Dict[str, Any]], baseline_response: Dict[str, Any]) -> pd.DataFrame:
        """
        Create DataFrame with combination results using probability-based similarity
       
        Args:
            responses: Dictionary of combination responses
            baseline_response: Baseline model response
           
        Returns:
            DataFrame with results
        """
        # Prepare data for DataFrame
        data = []
       
        baseline_probs = baseline_response["probabilities"]
        baseline_label = baseline_response["label"]
       
        # Process each combination response
        for key, res in responses.items():
            combination = res["combination"]
            indices = res["indices"]
            response_data = res["response"]
            response_probs = response_data["probabilities"]
            response_label = response_data["label"]
           
            # Calculate probability-based similarity (lower = more similar)
            prob_similarity = 1.0 - jensen_shannon_distance(baseline_probs, response_probs)
           
            # Track the probability of the baseline's predicted class
            baseline_class_prob = response_probs.get(baseline_label, 0.0)
           
            # Add to data
            data.append({
                "key": key,
                "combination": combination,
                "indices": indices,
                "response_label": response_label,
                "similarity": prob_similarity,
                "baseline_class_prob": baseline_class_prob,
                "probabilities": response_probs
            })
       
        # Create DataFrame
        df = pd.DataFrame(data)
        return df
   
    def _calculate_shapley_values(self, df: pd.DataFrame, content: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate Shapley values for each sample using probability distributions
       
        Args:
            df: DataFrame with combination results
            content: Original content
           
        Returns:
            Dictionary mapping sample names to various Shapley values
        """
        samples = self._get_samples(content)
        n = len(samples)
       
        # Initialize counters for each sample
        with_count = defaultdict(int)
        without_count = defaultdict(int)
        with_similarity_sum = defaultdict(float)
        without_similarity_sum = defaultdict(float)
        with_baseline_prob_sum = defaultdict(float)
        without_baseline_prob_sum = defaultdict(float)
       
        # Process each combination
        for _, row in df.iterrows():
            indices = row["indices"]  
            similarity = row["similarity"]
            baseline_class_prob = row["baseline_class_prob"]
           
            # Update counters for each sample
            for i in range(n):
                if i in indices:
                    with_similarity_sum[i] += similarity
                    with_baseline_prob_sum[i] += baseline_class_prob
                    with_count[i] += 1
                else:
                    without_similarity_sum[i] += similarity
                    without_baseline_prob_sum[i] += baseline_class_prob
                    without_count[i] += 1
        
        # Calculate Shapley values for different metrics
        shapley_values = {}
        for i in range(n):
            # Similarity-based Shapley (distribution similarity)
            with_avg = with_similarity_sum[i] / with_count[i] if with_count[i] > 0 else 0
            without_avg = without_similarity_sum[i] / without_count[i] if without_count[i] > 0 else 0
            similarity_shapley = with_avg - without_avg
           
            # Baseline class probability-based Shapley
            with_prob_avg = with_baseline_prob_sum[i] / with_count[i] if with_count[i] > 0 else 0
            without_prob_avg = without_baseline_prob_sum[i] / without_count[i] if without_count[i] > 0 else 0
            prob_shapley = with_prob_avg - without_prob_avg
           
            shapley_values[f"{samples[i]}_{i}"] = {
                "similarity_shapley": similarity_shapley,
                "prob_shapley": prob_shapley
            }
        
        # Normalize each type of Shapley value separately
        norm_shapley = self._normalize_shapley_dict(shapley_values)
       
        return norm_shapley
   
    def _normalize_shapley_dict(self, shapley_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Normalize each type of Shapley value separately"""
        # Get all metric types
        if not shapley_dict:
            return {}
           
        metrics = list(next(iter(shapley_dict.values())).keys())
        normalized = {k: {} for k in shapley_dict}
       
        # Normalize each metric separately
        for metric in metrics:
            values = [v[metric] for v in shapley_dict.values()]
            min_val = min(values)
            max_val = max(values)
            value_range = max_val - min_val
           
            if value_range > 0:
                for k, v in shapley_dict.items():
                    normalized[k][metric] = (v[metric] - min_val) / value_range
            else:
                for k, v in shapley_dict.items():
                    normalized[k][metric] = 0.5  # Default to middle when no variance
                   
        return normalized
   
    def get_tokens_shapley_values(self) -> Dict[str, Dict[str, float]]:
        """
         Returns a dictionary mapping each token to all its Shapley metrics.

        Returns:
            Dict[token, Dict[metric_name, value]] — e.g.
            {
                "bank": {"similarity_shapley": 0.82, "prob_shapley": 0.76},
                ...
            }
        """
        if not hasattr(self, 'shapley_values'):
            raise ValueError("Must run analyze() before getting Shapley values")

        # Extract token texts without indices
        tokens = {}
        for key, value in self.shapley_values.items():
            token = key.rsplit('_', 1)[0]  # Remove index suffix
            tokens[token] = value
       
        return tokens

    # A method to get the Similarity-based Shapley values specifically
    def get_sim_shapley_values(self) -> Dict[str, float]:
        """
        Returns a dictionary mapping each token to its similarity-based Shapley value
       
        Returns:
            Dictionary with token text as keys and similarity-based Shapley values as values
        """
        if not hasattr(self, 'shapley_values'):
            raise ValueError("Must run analyze() before getting Shapley values")

        # Extract token texts without indices and get the similarity-based metric
        tokens = {}
        for key, value_dict in self.shapley_values.items():
            token = key.rsplit('_', 1)[0]  # Remove index suffix
            tokens[token] = value_dict["similarity_shapley"]
       
        return tokens
   
    def analyze(self, prompt: str,
        sampling_ratio: float = 0.0,
        max_combinations: Optional[int] = 1000) -> pd.DataFrame:
        """
        Analyze token importance in a financial statement
       
        Args:
            prompt: Financial statement to analyze (without instructions)
            sampling_ratio: Ratio of combinations to sample (0-1)
            max_combinations: Maximum number of combinations to generate
           
        Returns:
            DataFrame with analysis results
        """
        # Clean prompt
        prompt = prompt.strip()
        prompt = re.sub(r'\s+', ' ', prompt)

        # Get baseline using full prompt with instructions
        prefix = "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.. Text: "
        suffix = ".. Answer: "
        full_prompt = build_full_prompt(prompt, prefix, suffix)
        self.baseline_response = self._calculate_baseline(full_prompt)
        self.baseline_text = self.baseline_response["label"]
       
        # Process combinations (this function will add instructions to each combination)
        responses = self._get_result_per_combination(
            prompt,
            sampling_ratio=sampling_ratio,
            max_combinations=max_combinations
        )
       
        # Create results DataFrame
        self.results_df = self._get_df_per_combination(responses, self.baseline_response)
       
        # Calculate Shapley values
        self.shapley_values = self._calculate_shapley_values(self.results_df, prompt)

        return self.results_df