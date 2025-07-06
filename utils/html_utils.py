"""
Utility functions for HTML processing
"""
from bs4 import BeautifulSoup
import re

def clean_html(text: str) -> str:
    """
    Clean HTML tags and entities from text.
    
    Args:
        text: Input text containing HTML
        
    Returns:
        Cleaned text without HTML tags
    """
    if not text:
        return ""
    
    # Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text(separator=' ')
    
    # Replace multiple spaces with single space
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text
