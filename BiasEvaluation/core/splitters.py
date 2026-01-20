import re
from typing import List

class Splitter:
    """Base class for text splitting"""
    def split(self, text: str) -> List[str]:
        raise NotImplementedError
       
    def join(self, tokens: List[str]) -> str:
        raise NotImplementedError

class StringSplitter(Splitter):
    """Split text by pattern (default: space)"""
    def __init__(self, split_pattern: str = ' '):
        self.split_pattern = split_pattern
   
    def split(self, prompt: str) -> List[str]:
        return re.split(self.split_pattern, prompt.strip())
   
    def join(self, tokens: List[str]) -> str:
        return ' '.join(tokens)

class TokenizerSplitter(Splitter):
    """Split text using HuggingFace tokenizer"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def split(self, prompt: str) -> List[str]:
        return self.tokenizer.tokenize(prompt)

    def join(self, tokens: List[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)