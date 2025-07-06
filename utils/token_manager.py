"""
Advanced token management system for AI models.
Provides accurate token counting and content truncation.
"""

import logging
import os
from typing import Dict
from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv()


class TokenManager:
    """
    Advanced token management for different AI models.
    Uses actual tokenizers for accurate token counting.
    """
    
    def __init__(self):
        """Initialize the token manager."""
        self.logger = logging.getLogger(__name__)
        self.tokenizers = {}
        self._load_tokenizers()
    
    def _load_tokenizers(self):
        """Load tokenizers for different models."""
        try:
            # For Groq (Gemma2 model)
            self.tokenizers['groq'] = AutoTokenizer.from_pretrained(
                "google/gemma-2-9b-it",
                token=os.getenv('HUGGINGFACE_API_KEY'),
                trust_remote_code=True
            )
            self.logger.info("Loaded Groq/Gemma2 tokenizer")
        except Exception as e:
            self.logger.warning(f"Failed to load Groq tokenizer: {e}")
            self.tokenizers['groq'] = None
        
        try:
            # For Together AI (Llama model)  
            self.tokenizers['together'] = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.3-70B-Instruct",
                token=os.getenv('HUGGINGFACE_API_KEY'),
                trust_remote_code=True
            )
            self.logger.info("Loaded Together/Llama tokenizer")
        except Exception as e:
            self.logger.warning(f"Failed to load Together tokenizer: {e}")
            self.tokenizers['together'] = None
    
    def count_tokens(self, text: str, model_provider: str = 'groq') -> int:
        """
        Count tokens using the appropriate tokenizer.
        
        Args:
            text: Text to count tokens for
            model_provider: 'groq' or 'together'
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
            
        tokenizer = self.tokenizers.get(model_provider)
        if tokenizer is None:
            # Fallback to character-based estimation
            return len(text) // 4
        
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            self.logger.warning(f"Token counting failed: {e}, using fallback")
            return len(text) // 4
    
    def truncate_to_tokens(self, text: str, max_tokens: int, 
                          model_provider: str = 'groq',
                          preserve_sentences: bool = True) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            model_provider: Model provider for tokenizer
            preserve_sentences: Try to preserve sentence boundaries
            
        Returns:
            Truncated text
        """
        if not text:
            return text
            
        current_tokens = self.count_tokens(text, model_provider)
        if current_tokens <= max_tokens:
            return text
        
        # Binary search for optimal length
        left, right = 0, len(text)
        best_text = text[:max_tokens * 4]  # Initial guess
        
        while left < right:
            mid = (left + right + 1) // 2
            candidate = text[:mid]
            
            if self.count_tokens(candidate, model_provider) <= max_tokens:
                best_text = candidate
                left = mid
            else:
                right = mid - 1
        
        if preserve_sentences and len(best_text) > 100:
            # Try to end at sentence boundary
            sentences_ends = ['.', '!', '?', '\n']
            last_sentence = -1
            
            for i in range(len(best_text) - 1, max(0, len(best_text) - 200), -1):
                if best_text[i] in sentences_ends and i < len(best_text) - 1:
                    last_sentence = i + 1
                    break
            
            if last_sentence > len(best_text) * 0.8:
                best_text = best_text[:last_sentence]
        
        return best_text + "...[truncated]" if len(best_text) < len(text) else best_text
    
    def prepare_content_for_classification(self, content: Dict[str, str], 
                                         max_tokens: int = 5000) -> Dict[str, str]:
        """
        Prepare content for classification with optimal token usage.
        
        Args:
            content: Content dictionary with title, description, url
            max_tokens: Maximum tokens for all content combined
            
        Returns:
            Optimized content dictionary
        """
        title = content.get('title', '')
        description = content.get('description', '')
        url = content.get('url', '')
        
        # Reserve tokens for title and URL (these are critical)
        title_tokens = self.count_tokens(title, 'groq')
        url_tokens = self.count_tokens(url, 'groq')
        
        # Reserve 500 tokens for prompt and response
        available_for_description = max_tokens - title_tokens - url_tokens - 500
        
        if available_for_description > 0:
            truncated_description = self.truncate_to_tokens(
                description, available_for_description, 'groq', True
            )
        else:
            # If even title + URL is too long, truncate title
            available_for_title = max_tokens - url_tokens - 500
            title = self.truncate_to_tokens(title, available_for_title, 'groq', False)
            truncated_description = ""
        
        return {
            'title': title,
            'description': truncated_description,
            'url': url
        }
    
    def prepare_content_for_summarization(self, content: str, 
                                        max_tokens: int = 3000,
                                        model_provider: str = 'together') -> str:
        """
        Prepare content for summarization with intelligent truncation.
        
        Args:
            content: Full content text
            max_tokens: Maximum tokens for content
            model_provider: Model provider for tokenizer
            
        Returns:
            Optimized content for summarization
        """
        if not content:
            return content
            
        current_tokens = self.count_tokens(content, model_provider)
        if current_tokens <= max_tokens:
            return content
        
        # For long content, take beginning and end to preserve context
        if current_tokens > max_tokens * 2:
            # Take first 60% and last 40% of the available tokens
            first_part_tokens = int(max_tokens * 0.6)
            last_part_tokens = int(max_tokens * 0.4)
            
            # Find the midpoint to split
            midpoint = len(content) // 2
            first_part = self.truncate_to_tokens(
                content[:midpoint], first_part_tokens, model_provider
            )
            
            # For the last part, work backwards from the end
            last_part = content[midpoint:]
            last_part = self.truncate_to_tokens(
                last_part, last_part_tokens, model_provider
            )
            
            return first_part + "\n\n[... content truncated ...]\n\n" + last_part
        else:
            # Simple truncation for moderately long content
            return self.truncate_to_tokens(content, max_tokens, model_provider)


# Global instance
token_manager = TokenManager()
