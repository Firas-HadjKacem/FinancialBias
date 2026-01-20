#Base classes and utilities for TokenSHAP
# SPDX-FileCopyrightText: 2023-2024 The TokenSHAP Authors
import pandas as pd
import random
from typing import List, Dict, Optional, Tuple, Any
from tqdm.auto import tqdm

class ModelBase:
    """Base model interface"""
   
    def generate(self, **kwargs) -> str:
        """Generate a response for the given input"""
        raise NotImplementedError


class BaseSHAP:
    """Base class for SHAP value calculation with Monte Carlo sampling"""
   
    def __init__(self, model: ModelBase, debug: bool = False):
        """
        Initialize BaseSHAP
       
        Args:
            model: Model to analyze
            debug: Enable debug output
        """
        self.model = model  
        self._cache = {}  # Cache for model responses
        self.debug = debug
   
    def _calculate_baseline(self, content: str) -> Dict[str, Any]:
        """Calculate baseline model response for full content"""
        # Content here should already have the prefix/suffix if needed
        baseline = self.model.generate(prompt=content)
        if self.debug:
            print(f"Baseline prediction: {baseline['label']}")
        return baseline
   
    def _prepare_generate_args(self, content: str, **kwargs) -> Dict:
        """Prepare arguments for model.generate()"""
        raise NotImplementedError
   
    def _get_samples(self, content: str) -> List[str]:
        """Get samples from content"""
        raise NotImplementedError
   
    def _prepare_combination_args(self, combination: List[str], original_content: str) -> Dict:
        """Prepare model arguments for a combination"""
        raise NotImplementedError
   
    def _get_combination_key(self, combination: List[str], indexes: Tuple[int, ...]) -> str:
        """Get unique key for combination"""
        raise NotImplementedError
   
    def _get_all_combinations(self, samples: List[str], sampling_ratio: float = 0.0,
                             max_combinations: Optional[int] = None) -> Dict[str, Tuple[List[str], Tuple[int, ...]]]:
        """
        Get all possible combinations of samples with their indices
       
        Args:
            samples: List of samples (e.g., tokens)
            sampling_ratio: Ratio of combinations to sample (0-1)
            max_combinations: Maximum number of combinations to generate
           
        Returns:
            Dictionary mapping combination keys to (combination, indices) tuples
        """
        n = len(samples)
        # Always include combinations that exclude exactly one token
        essential_combinations = {}      
        for i in range(n):
            combination = samples.copy()
            del combination[i]
            indices = tuple(j for j in range(n) if j != i)
            key = f"omit_{i}"
            essential_combinations[key] = (combination, indices)
       
        # Calculate total possible combinations and sampling count
        if sampling_ratio <= 0:
            # Just return essential combinations
            return essential_combinations
           
        total_combinations = 2**n - 1  # All non-empty combinations
        sample_count = int(total_combinations * sampling_ratio)
       
        if max_combinations is not None:
            sample_count = min(sample_count, max_combinations)
       
        if sample_count <= len(essential_combinations):
            return essential_combinations
        
        # Randomly sample additional combinations
        all_combinations = essential_combinations.copy()
        additional_needed = sample_count - len(essential_combinations)       
        # Generate random combinations
        combinations_added = 0
        max_attempts = additional_needed * 10  # Limit attempts to avoid infinite loop
        attempts = 0
       
        while combinations_added < additional_needed and attempts < max_attempts:
            # Decide how many tokens to include
            subset_size = random.randint(1, n-1)  # At least 1, at most n-1
           
            # Randomly select indices
            indices = tuple(sorted(random.sample(range(n), subset_size)))
           
            # Create combination
            combination = [samples[i] for i in indices]
            key = f"random_{','.join(str(i) for i in indices)}"
           
            # Only add if not already present
            if key not in all_combinations:
                all_combinations[key] = (combination, indices)
                combinations_added += 1
           
            attempts += 1
       
        if self.debug and attempts >= max_attempts:
            print(f"Warning: Reached max attempts ({max_attempts}) when generating combinations")
           
        return all_combinations
   
    def _get_result_per_combination(self, content: str, sampling_ratio: float = 0.0,
                                   max_combinations: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get model responses for combinations of content
       
        Args:
            content: Original content
            sampling_ratio: Ratio of combinations to sample
            max_combinations: Maximum number of combinations
           
        Returns:
            Dictionary mapping combination keys to response data
        """
        samples = self._get_samples(content)
        if self.debug:
            print(f"Found {len(samples)} samples in content")
       
        combinations = self._get_all_combinations(samples, sampling_ratio, max_combinations)
        if self.debug:
            print(f"Generated {len(combinations)} combinations")
        
        results = {}
        # Process each combination
        for key, (combination, indices) in tqdm(combinations.items(), desc="Processing combinations"):
            comb_args = self._prepare_combination_args(combination, content)
            comb_key = self._get_combination_key(combination, indices)
           
            # Check cache first
            if comb_key in self._cache:
                response = self._cache[comb_key]
            else:
                response = self.model.generate(**comb_args)
                self._cache[comb_key] = response
           
            # Store results
            results[key] = {
                "combination": combination,
                "indices": indices,
                "response": response
            }        
        return results

    def analyze(self, content: str, sampling_ratio: float = 0.0, max_combinations: Optional[int] = None) -> pd.DataFrame:
        """
        Analyze importance in content
       
        Args:
            content: Content to analyze
            sampling_ratio: Ratio of combinations to sample
            max_combinations: Maximum number of combinations
           
        Returns:
            DataFrame with analysis results
        """
        raise NotImplementedError