"""
Content classification system for tech news articles and arXiv papers.
Uses AI to categorize content into predefined categories.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import yaml
from langchain.schema import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from utils.rate_limiter import groq_rate_limiter
from utils.token_manager import token_manager
from utils.robust_parser import robust_parser
from utils.html_utils import clean_html
from dotenv import load_dotenv

load_dotenv()

class ContentCategory(BaseModel):
    """Pydantic model for content classification results."""
    category: str = Field(description="Primary category of the content")
    subcategory: Optional[str] = Field(description="More specific subcategory")
    confidence: float = Field(description="Confidence score between 0 and 1")
    tags: List[str] = Field(description="Relevant tags for the content")
    reasoning: str = Field(description="Brief explanation for the classification")


class TechNewsClassifier:
    """
    AI-powered classifier for tech news articles and research papers.
    Categorizes content into predefined tech categories.
    """
    
    def __init__(self, config_path: str = "config/classification.yml"):
        """        Initialize the classifier.
        
        Args:
            config_path: Path to classification configuration file.
        """
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.categories = self._load_categories()
        self.llm = self._setup_llm()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the classifier."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_categories(self) -> Dict[str, Any]:
        """Load classification categories from config file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                # Return default categories if config file doesn't exist
                return self._get_default_categories()
        except Exception as e:
            self.logger.warning(f"Failed to load categories from {self.config_path}: {e}")
            return self._get_default_categories()
    
    def _get_default_categories(self) -> Dict[str, Any]:
        """Get default tech news categories."""
        return {
            "categories": {
                "Artificial Intelligence": {
                    "subcategories": ["Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Robotics"],
                    "keywords": ["AI", "artificial intelligence", "machine learning", "neural networks", "deep learning"]
                },
                "Software Development": {
                    "subcategories": ["Programming Languages", "Frameworks", "DevOps", "Web Development", "Mobile Development"],
                    "keywords": ["programming", "software", "development", "framework", "library", "coding"]
                },
                "Hardware": {
                    "subcategories": ["Processors", "Graphics Cards", "Mobile Devices", "Computing Hardware"],
                    "keywords": ["CPU", "GPU", "processor", "chip", "hardware", "smartphone", "laptop"]
                },
                "Cybersecurity": {
                    "subcategories": ["Data Breaches", "Vulnerabilities", "Security Tools", "Privacy"],
                    "keywords": ["security", "hack", "breach", "vulnerability", "privacy", "encryption"]
                },
                "Cloud Computing": {
                    "subcategories": ["AWS", "Azure", "Google Cloud", "Infrastructure", "SaaS"],
                    "keywords": ["cloud", "AWS", "Azure", "Google Cloud", "SaaS", "infrastructure"]
                },
                "Startups & Business": {
                    "subcategories": ["Funding", "IPO", "Acquisitions", "Corporate News"],
                    "keywords": ["startup", "funding", "investment", "IPO", "acquisition", "merger"]
                },
                "Social Media & Platforms": {
                    "subcategories": ["Meta", "Twitter/X", "TikTok", "YouTube", "Platform Updates"],
                    "keywords": ["social media", "Facebook", "Twitter", "TikTok", "YouTube", "platform"]
                },
                "Gaming": {
                    "subcategories": ["Game Releases", "Gaming Hardware", "Esports", "Game Development"],
                    "keywords": ["gaming", "games", "PlayStation", "Xbox", "Nintendo", "esports"]
                },                "Other": {
                    "subcategories": ["General Tech", "Research", "Industry News"],
                    "keywords": []                }
            }
        }

    def _setup_llm(self) -> Optional[ChatGroq]:
        """Set up the Groq language model for classification."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            self.logger.warning("GROQ_API_KEY not found. Classification will use rule-based fallback.")
            return None
        
        return ChatGroq(
            model="gemma2-9b-it",            temperature=0.0
        )
    
    def classify_content(self, content: Dict[str, str]) -> ContentCategory:
        """
        Classify a single piece of content.
        
        Args:
            content: Dictionary with title, description, and url keys.
            
        Returns:
            ContentCategory object with classification results.
        """
        try:
            if self.llm:
                return self._classify_with_ai(content)
            else:
                return self._classify_with_rules(content)
        except Exception as e:
            self.logger.error(f"Classification failed for content '{content.get('title', 'Unknown')}': {e}")
            return ContentCategory(
                category="Other",
                subcategory="General Tech",
                confidence=0.1,
                tags=[],                reasoning="Classification failed due to error"
            )
    def _classify_with_ai(self, content: Dict[str, str]) -> ContentCategory:
        """Classify content using AI model."""
        # Clean HTML content before processing
        cleaned_content = {
            'title': clean_html(content.get('title', '')),
            'description': clean_html(content.get('description', '')),
            'url': content.get('url', '')  # URL doesn't need HTML cleaning
        }
        
        # Use improved token management with cleaned content
        truncated_content = token_manager.prepare_content_for_classification(cleaned_content, max_tokens=5000)
        
        categories_text = "\n".join([
            f"- {category}: {', '.join(details['subcategories'])} (keywords: {', '.join(details['keywords'])})"
            for category, details in self.categories['categories'].items()
        ])
        
        prompt_text = f"""You are an expert tech news classifier. Classify the given content into one of the predefined categories.

Available Categories:
{categories_text}

Content to classify:
Title: {truncated_content.get('title', '')}
Description: {truncated_content.get('description', '')}
URL: {truncated_content.get('url', '')}

Analyze the content and provide a classification with high confidence. Consider:
1. The main topic and focus of the content
2. Technical keywords and terminology used
3. The target audience and context
4. Relevance to specific tech domains

Respond with valid JSON in this exact format:
{{
    "category": "category name from the list above",
    "subcategory": "relevant subcategory or General",
    "confidence": 0.85,
    "tags": ["tag1", "tag2", "tag3"],
    "reasoning": "brief explanation of why this category was chosen"
}}

Classification:"""
        
        messages = [
            SystemMessage(content="You are an expert tech news classifier. Always respond with valid JSON only."),
            HumanMessage(content=prompt_text)
        ]
        
        # Apply rate limiting before making the API call
        groq_rate_limiter.wait_if_needed()
        
        if self.llm is None:
            self.logger.warning("LLM not available, falling back to rule-based classification")
            return self._classify_with_rules(content)
        
        try:
            response = self.llm.invoke(messages)
            content_text = response.content if isinstance(response.content, str) else str(response.content)
            
            if not content_text or content_text.strip() == "":
                self.logger.warning("Empty response from LLM, falling back to rule-based classification")
                return self._classify_with_rules(content)
            
            # Use robust JSON parser
            parsed_result = robust_parser.parse_classification_response(
                content_text, 
                fallback_category="Other"
            )
            
            # Convert to ContentCategory object
            return ContentCategory(
                category=parsed_result['category'],
                subcategory=parsed_result['subcategory'],
                confidence=parsed_result['confidence'],
                tags=parsed_result['tags'],
                reasoning=parsed_result['reasoning']
            )
            
        except Exception as e:
            self.logger.warning(f"LLM classification failed: {e}, falling back to rule-based classification")
            return self._classify_with_rules(content)
    
    def _classify_with_rules(self, content: Dict[str, str]) -> ContentCategory:
        """Fallback rule-based classification."""
        text_to_analyze = f"{content.get('title', '')} {content.get('description', '')}".lower()
        
        best_category = "Other"
        best_subcategory = "General Tech"
        best_score = 0
        matched_keywords = []
        
        for category, details in self.categories['categories'].items():
            keywords = details.get('keywords', [])
            score = sum(1 for keyword in keywords if keyword.lower() in text_to_analyze)
            
            if score > best_score:
                best_score = score
                best_category = category
                best_subcategory = details['subcategories'][0] if details['subcategories'] else "General"
                matched_keywords = [kw for kw in keywords if kw.lower() in text_to_analyze]
        
        confidence = min(0.9, best_score * 0.2) if best_score > 0 else 0.3
        
        return ContentCategory(
            category=best_category,
            subcategory=best_subcategory,
            confidence=confidence,
            tags=matched_keywords[:5],
            reasoning=f"Rule-based classification based on {len(matched_keywords)} keyword matches"
        )
    
    def classify_batch(self, content_list: List[Dict[str, str]]) -> List[ContentCategory]:
        """
        Classify multiple pieces of content.
        
        Args:
            content_list: List of content dictionaries.
            
        Returns:
            List of ContentCategory objects.
        """
        results = []
        
        for i, content in enumerate(content_list):
            try:
                classification = self.classify_content(content)
                results.append(classification)
                self.logger.info(f"Classified content {i+1}/{len(content_list)}: {classification.category}")
            except Exception as e:
                self.logger.error(f"Failed to classify content {i+1}: {e}")
                results.append(ContentCategory(
                    category="Other",
                    subcategory="General Tech",
                    confidence=0.1,
                    tags=[],
                    reasoning="Classification failed due to error"                ))
        
        return results
    
    def save_classifications(self, classifications: List[ContentCategory], 
                           content_list: List[Dict[str, Any]], 
                           filename: Optional[str] = None) -> str:
        """
        Save classifications to a JSON file.
        
        Args:
            classifications: List of ContentCategory objects.
            content_list: Original content list.
            filename: Custom filename. If None, uses timestamp.
            
        Returns:
            Path to the saved file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"classifications_{timestamp}.json"
        
        # Ensure data directory exists
        os.makedirs("data/classifications", exist_ok=True)
        filepath = os.path.join("data/classifications", filename)
        
        # Combine content with classifications
        results = []
        for content, classification in zip(content_list, classifications):
            result = content.copy()
            result['classification'] = classification.model_dump()
            results.append(result)
        
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(classifications)} classifications to {filepath}")
        return filepath

    def _is_tech_content(self, content: Dict[str, str], classification: ContentCategory) -> bool:
        """
        Determine if content is tech-related (IT and Computer Science focus).
        
        Args:
            content: Content dictionary with title, description, etc.
            classification: Classification result from LLM or rule-based fallback.
            
        Returns:
            True if content is tech-related, False otherwise.
        """
        # Core tech categories (IT and Computer Science focused)
        core_tech_categories = {cat for cat in self.categories['categories'].keys() 
                       if cat not in {"Startups & Business", "Social Media & Platforms", "Gaming", "Other"}}
        
        # tech-related content
        broader_categories = {
            "Startups & Business",
            "Social Media & Platforms",
            "Gaming",
            "Other"
        }
        
        # tech keywords list
        tech_keywords = [
            "programming", "software", "algorithm", "computer", "technology",
            "tech", "it", "coding", "development", "engineering", "system",
            "network", "database", "server", "api", "framework", "library",
            "quantum", "blockchain", "cryptocurrency", "robotics", "autonomous",
            "virtual reality", "augmented reality", "machine learning", "ai",
            "deep learning", "neural network", "data science", "cloud", "security",
            "hardware", "processor", "gpu", "cpu", "smartphone", "laptop",
            "internet", "web", "mobile", "app", "cybersecurity", "encryption",
            "iot", "big data", "analytics", "devops", "saas", "infrastructure"
        ]
        
        # Step 1: Check if category is in core tech categories
        if classification.category in core_tech_categories:
            return True
        
        # Step 2: Check broader categories with keyword evidence
        if classification.category in broader_categories:
            # Combine title, description, and tags for a comprehensive check
            text_to_check = f"{content.get('title', '')} {content.get('description', '')} {' '.join(classification.tags)}".lower()
            # Accept if at least one tech keyword is present
            if any(keyword in text_to_check for keyword in tech_keywords):
                return True
        
        # Step 3: Default rejection for non-matching content
        return False

    def classify_content_tech_only(self, content: Dict[str, str]) -> Optional[ContentCategory]:
        """
        Classify content and return only if it's tech-related.
        
        Args:
            content: Dictionary with title, description, and url keys.
            
        Returns:
            ContentCategory object if tech-related, None otherwise.
        """
        classification = self.classify_content(content)
        
        if self._is_tech_content(content, classification):
            return classification
        else:
            self.logger.info(f"Filtered out non-tech content: {content.get('title', 'Unknown')[:50]}...")
            return None


