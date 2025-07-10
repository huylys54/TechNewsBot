"""
Content parser for extracting full content from news articles.
"""

import requests
import logging
import os
import json
import re
from typing import List, Dict, Optional
from datetime import datetime
import time
from bs4 import BeautifulSoup
from pathlib import Path


class ContentParser:
    """
    Parses and extracts full content from news article URLs.
    
    This class handles fetching the full text content from news articles
    using intelligent HTML parsing.
    """
    
    def __init__(self):
        """Initialize the NewsContentParser."""
        self.logger = self._setup_logging()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def extract_content(self, articles: List[Dict[str, str]], delay: float = 1.0) -> List[Dict[str, str]]:
        """
        Extract full content from a list of news articles.
        
        Args:
            articles: List of articles with 'link' field.
            delay: Delay between requests in seconds to be respectful.
            
        Returns:
            List of articles with added 'content' field containing extracted text.
        """
        enriched_articles = []
        
        for i, article in enumerate(articles):
            try:
                content = self._fetch_article_content(article.get('link', ''))
                enriched_article = article.copy()
                enriched_article['content'] = content
                enriched_article['content_fetch_time'] = datetime.now().isoformat()
                enriched_articles.append(enriched_article)
                
                self.logger.info(f"Extracted content from {article.get('source', 'Unknown')} article: {article.get('title', 'No title')[:50]}...")
                
                # Add delay between requests
                if i < len(articles) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                self.logger.error(f"Failed to extract content from {article.get('link', '')}: {e}")
                # Add article without content
                enriched_article = article.copy()
                enriched_article['content'] = article.get('description', '')  # Use description as fallback
                enriched_article['content_error'] = str(e)
                enriched_articles.append(enriched_article)
        
        self.logger.info(f"Successfully extracted content from {len([a for a in enriched_articles if a.get('content')])} out of {len(articles)} articles")
        return enriched_articles
    
    def _fetch_article_content(self, url: str) -> str:
        """
        Fetch and extract clean text content from a single article URL.
        
        Args:
            url: URL of the article to fetch.
            
        Returns:
            Clean text content of the article.
        """
        if not url:
            raise ValueError("Empty URL provided")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML and extract text content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Try to find main content area
            content_text = self._extract_main_content(soup)
            
            # Clean up the text
            content_text = re.sub(r'\s+', ' ', content_text).strip()
            
            return content_text
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch article content: {e}")
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content from HTML soup using common content selectors.
        
        Args:
            soup: BeautifulSoup object of the HTML.
            
        Returns:
            Extracted text content.
        """
        # Common content selectors (ordered by priority)
        content_selectors = [
            'article',
            '[role="main"]',
            'main',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            '#content',
            '.story-body',
            '.article-body'
        ]
        
        # Try each selector
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Extract text from paragraphs and headings
                text_elements = content_elem.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if text_elements:
                    return ' '.join([elem.get_text().strip() for elem in text_elements if elem.get_text().strip()])
        
        # Fallback: extract all paragraphs
        paragraphs = soup.find_all('p')
        if paragraphs:
            return ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        # Last resort: get all text
        return soup.get_text()

    def save_content_to_file(self, articles: List[Dict[str, str]], filename: Optional[str] = None) -> str:
        """
        Save articles with content to a JSON file.
        
        Args:
            articles: Articles with content to save.
            filename: Custom filename. If None, uses timestamp.
            
        Returns:
            Path to the saved file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"articles_with_content_{timestamp}.json"
        
        # Ensure data directory exists
        os.makedirs("data/content", exist_ok=True)
        filepath = os.path.join("data/content", filename)
        
        p = Path(filepath)
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(articles, file, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(articles)} articles with content to {filepath}")
        return filepath
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the content parser."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
