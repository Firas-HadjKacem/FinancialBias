# SPDX-FileCopyrightText: 2023-2024 The TokenSHAP Authors

from typing import Dict
from ..utils.helpers import build_full_prompt
import os
import csv
from ..core.models import BERTModel
from ..core.splitters import StringSplitter, TokenizerSplitter
from ..analysis.tokenShap import TokenSHAP
from ..utils.data_loader import PROMPT_PREFIX, PROMPT_SUFFIX  # or move them to helpers

class BiasAnalyzer:
    """
    Analyzes bias in financial language models using TokenSHAP.
    
    This class combines model inference with TokenSHAP analysis to identify
    bias-inducing tokens in financial sentiment analysis. It uses Jensen-Shannon
    distance to measure how token presence affects prediction distributions.
    
    Attributes:
        model_wrapper: Wrapped model with generate() and generate_batch() methods
        splitter: Text splitter for tokenization (string-based or tokenizer-based)
        token_shap: TokenSHAP analyzer instance for computing Shapley values
        
    Example:
        >>> analyzer = BiasAnalyzer(model, tokenizer, model_type="bert")
        >>> result = analyzer.analyze_sentence("Apple stock rises sharply")
        >>> print(result['prediction']['label'])  # "Positive"
        >>> print(result['Bias Token Ranks'])  # Token importance rankings
    """
    
    def __init__(self, model, tokenizer, model_type, splitter_type='string', batch_size = 16, is_wrapped=False):
        """
        Initialize the bias analyzer.

        Args:
            model:
                Either:
                  - a pre-wrapped model implementing `generate()` and
                    `generate_batch()` (e.g., `BERTModel`, `LlamaModelWrapper`),
                    in which case you should pass `is_wrapped=True`, or
                  - a raw Hugging Face model, in which case the analyzer may
                    wrap it internally (only supported for BERT).
            tokenizer:
                Tokenizer associated with the underlying Hugging Face model.
            model_type:
                String indicating model family, e.g. "bert" or "llama".
            splitter_type:
                How to split the input for TokenSHAP: "string" or "tokenizer".
            batch_size:
                Batch size for SHAP sampling; passed to `TokenSHAP` and used
                when constructing prompts for SHAP combinations.
            is_wrapped:
                If True, `model` is assumed to already be a wrapper and will
                be used as-is without further checks.
        """
        # Check if model is already a wrapper
        if is_wrapped or hasattr(model, 'generate') and hasattr(model, 'generate_batch'):
            print("Using pre-wrapped model")
            self.model_wrapper = model
        else:
            # Check for bert or llama based model
            if model_type == 'bert':
                self.model_wrapper = BERTModel(model, tokenizer)
            elif model_type == 'llama':
                # Assuming label_ids is passed separately or handled elsewhere
                raise ValueError("For LLaMA models, please pass a LlamaModelWrapper and set is_wrapped=True when instantiating BiasAnalyzer.")
            else:
                raise ValueError(f"Unknown model type: {type(model)}. Only BERT and Llama models are supported.")
       
        # Create appropriate splitter
        if splitter_type == 'string':
            self.splitter = StringSplitter()
        elif splitter_type == 'tokenizer':
            self.splitter = TokenizerSplitter(tokenizer)
        else:
            raise ValueError(f"Unknown splitter type: {splitter_type}")
            
        # Initialize token SHAP
        self.token_shap = TokenSHAP(self.model_wrapper, self.splitter, batch_size=batch_size) 
    
    def compare_sentences(self, original: str, mutated: str, sampling_ratio: float = 0.1, max_combinations: int = 100):
        """
        Compare original and mutated sentences
       
        Args:
            original: Original financial sentence
            mutated: Mutated version of the sentence
            sampling_ratio: Ratio of combinations to sample
            max_combinations: Maximum number of combinations
           
        Returns:
            Comparison results
        """
        # Analyze both sentences
        original_result = self.analyze_sentence(original, sampling_ratio, max_combinations)
        mutated_result = self.analyze_sentence(mutated, sampling_ratio, max_combinations)
       
        # Get prediction changes
        prediction_change = mutated_result['prediction']['label'] != original_result['prediction']['label']
       
        # Find common bias tokens
        common_bias_tokens = set(original_result['Bias Token Ranks'].keys()) & set(mutated_result['Bias Token Ranks'].keys())
       
        # Compare ranks for common bias tokens
        bias_rank_changes = {}
        for token in common_bias_tokens:
            orig_rank = original_result['Bias Token Ranks'][token]['rank']
            mut_rank = mutated_result['Bias Token Ranks'][token]['rank']
            bias_rank_changes[token] = {
                'original_rank': orig_rank,
                'mutated_rank': mut_rank,
                'rank_changed': orig_rank != mut_rank,
                'rank_difference': mut_rank - orig_rank
            }
       
        return {
            'original': original_result,
            'mutated': mutated_result,
            'prediction_changed': prediction_change,
            'common_bias_tokens': list(common_bias_tokens),
            'bias_rank_changes': bias_rank_changes
        }
   
    def analyze_sentence(self, financial_statement: str, sampling_ratio: float = 0.5, max_combinations: int = 1000):
        """
        Analyze a single financial statement
        This uses our JSD-based TokenSHAP variant to compute token-level
        importance scores and then ranks bias-related tokens accordingly.
       
        Args:
            financial_statement: Plain financial statement to analyze (without instructions)
            sampling_ratio: Ratio of combinations to sample
            max_combinations: Maximum number of combinations
           
        Returns:
            Prediction and analysis results
        """
        # Create the full prompt with instructions
        full_prompt = build_full_prompt(financial_statement, PROMPT_PREFIX, PROMPT_SUFFIX)
       
        # Get baseline prediction using the FULL prompt
        prediction = self.model_wrapper.generate(prompt=full_prompt)
       
        # Store the prefix and suffix in TokenSHAP for use in combinations
        self.token_shap.prompt_prefix = PROMPT_PREFIX
        self.token_shap.prompt_suffix = PROMPT_SUFFIX
       
        # Store the original statement for multi-word bias detection
        self.token_shap.original_statement = financial_statement
       
        # Run TokenSHAP analysis on ONLY the financial statement
        self.token_shap.analyze(financial_statement, sampling_ratio, max_combinations)
       
        # Get token importance values
        shapley_values = self.token_shap.get_tokens_shapley_values()
        shapley_values_similarity = self.token_shap.get_sim_shapley_values()
        bias_tokens_ranks = self.analyze_bias_tokens_importance('../data/bias', original_text=financial_statement)
       
        return {
            'sentence': financial_statement,
            'prediction': prediction,
            'Shapley Values': shapley_values_similarity,
            'Bias Token Ranks': bias_tokens_ranks
        }
    
    def analyze_bias_tokens_importance(self, bias_files_dir: str, original_text: str = None):
        """
        Analyze the importance of bias tokens in a financial statement
       
        Args:
            bias_files_dir: Directory containing files with bias terms            
        Returns:
            Dictionary with bias analysis results including rankings
        """
        # Load bias terms from files
        single_word_terms, multi_word_terms = self._load_bias_terms(bias_files_dir)
       
        # Get the original sentence and token importance values
        shapley_values_similarity = self.token_shap.get_sim_shapley_values()
       
        # Rank ALL tokens by importance (highest to lowest)
        all_tokens_ranked = sorted(shapley_values_similarity.items(), key=lambda x: x[1], reverse=True)
       
        # Create rankings dictionary with positions
        total_tokens = len(all_tokens_ranked)
        token_rankings = {token: {'value': value, 'rank': idx + 1}
                        for idx, (token, value) in enumerate(all_tokens_ranked)}
       
        # Get the original text - use parameter if provided, otherwise try to get from object
        if original_text is None:
            original_text = getattr(self.token_shap, 'original_statement', '')
   
        # Original content in lowercase for case-insensitive matching
        original_text_lower = original_text.lower()
       
        # Identify bias tokens and their rankings
        bias_tokens_with_rank = {}
       
        # 1. Process single-word terms
        for token, token_data in token_rankings.items():
            if token.lower() in single_word_terms:
                rank = token_data['rank']
                value = token_data['value']
                bias_tokens_with_rank[token] = {
                    'shapley_value': value,
                    'rank': rank,
                    'total_tokens': total_tokens,
                    'percentile': round((1 - (rank - 1) / total_tokens) * 100, 1),
                    'type': 'single_word'
                }
       
        # 2. Process multi-word terms by checking the original sentence
        for multi_word_term in multi_word_terms:
           
            # Case insensitive check if the term exists in the original content
            if multi_word_term.lower() in original_text_lower:
               
                # Split the multi-word term into individual words
                term_words = multi_word_term.lower().split()
               
                # Find matching tokens in our token rankings
                matched_tokens = []
                matched_values = []
               
                # Look for each word in the tokenized tokens
                for word in term_words:
                    for token, data in token_rankings.items():
                        # Case insensitive comparison
                        if word == token.lower():
                            matched_tokens.append(token)
                            matched_values.append(data['value'])
                            break
               
                # If we found at least one token, calculate an aggregate score
                if matched_tokens:
                    avg_value = sum(matched_values) / len(matched_values)
                   
                    # Find equivalent rank based on value
                    equivalent_rank = 1
                    for idx, (_, value) in enumerate(all_tokens_ranked):
                        if avg_value >= value:
                            equivalent_rank = idx + 1
                            break
                        equivalent_rank = idx + 2  # If lower than all, put at the end
                   
                    # Add the multi-word term to results
                    bias_tokens_with_rank[multi_word_term] = {
                        'shapley_value': avg_value,
                        'rank': equivalent_rank,
                        'total_tokens': total_tokens,
                        'percentile': round((1 - (equivalent_rank - 1) / total_tokens) * 100, 1),
                        'type': 'multi_word',
                        'constituent_tokens': matched_tokens,
                        'individual_values': dict(zip(matched_tokens, matched_values))
                    }
               
        return bias_tokens_with_rank

    def _load_bias_terms(self, bias_files_dir: str) -> tuple:
        """
        Load bias terms from files in the specified directory
       
        Args:
            bias_files_dir: Directory containing files with bias terms
               
        Returns:
            Tuple of (single_word_terms, multi_word_terms)
        """
        single_word_terms = set()
        multi_word_terms = set()
       
        # Check if the directory exists
        if not os.path.exists(bias_files_dir):
            raise ValueError(f"Bias files directory {bias_files_dir} does not exist")
       
        # Load terms from each file
        for bias_folder in os.listdir(bias_files_dir):
            folder_path = os.path.join(bias_files_dir, bias_folder)
            if not os.path.isdir(folder_path):
                continue
               
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        csv_reader = csv.reader(f, delimiter=';')
                        for row in csv_reader:
                            for term in row:
                                term = term.strip().lower()
                                if term:
                                    if ' ' in term:
                                        multi_word_terms.add(term)
                                    else:
                                        single_word_terms.add(term)
       
        return single_word_terms, multi_word_terms
