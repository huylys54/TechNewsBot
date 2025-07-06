"""
Robust JSON parser for AI model outputs with fallback strategies.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List


class RobustJsonParser:
    """
    Enhanced JSON parser for AI model outputs with multiple fallback strategies.
    """
    
    def __init__(self):
        """Initialize the robust JSON parser."""
        self.logger = logging.getLogger(__name__)
    
    def extract_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from AI response with multiple fallback strategies.
        
        Args:
            response_text: Raw response text from AI model
            
        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        if not response_text or not response_text.strip():
            self.logger.warning("Empty response text provided")
            return None
        
        # Strategy 1: Try direct JSON parsing
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Look for JSON blocks wrapped in code fences
        json_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        for block in json_blocks:
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue
        
        # Strategy 3: Look for JSON objects (anything between { and })
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Single level nesting
            r'\{.*?\}(?=\s*$|\s*\n|\s*[^\w\{])',  # JSON at end of line
            r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'    # Multi-level nesting
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    cleaned = self._clean_json_string(match)
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
        
        # Strategy 4: Try to construct JSON from structured text
        return self._construct_json_from_structured_text(response_text)
    
    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean common issues in JSON strings from AI responses.
        
        Args:
            json_str: Raw JSON string
            
        Returns:
            Cleaned JSON string
        """
        # Remove leading/trailing whitespace
        cleaned = json_str.strip()
        
        # Remove trailing commas
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # Fix unescaped quotes in strings
        cleaned = re.sub(r'(?<!\\)"(?![,:}\]\s])', '\\"', cleaned)
        
        # Remove control characters that might interfere
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
        
        return cleaned
    
    def _construct_json_from_structured_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to construct JSON from structured text output.
        
        Args:
            text: Structured text that might contain JSON-like information
            
        Returns:
            Constructed JSON dictionary or None
        """
        result = {}
        
        # Common patterns for classification outputs
        patterns = {
            'category': [
                r'category[:\s]+([^\n\r]+)',
                r'primary category[:\s]+([^\n\r]+)',
                r'classification[:\s]+([^\n\r]+)'
            ],
            'subcategory': [
                r'subcategory[:\s]+([^\n\r]+)',
                r'sub-category[:\s]+([^\n\r]+)'
            ],
            'confidence': [
                r'confidence[:\s]+([0-9.]+)',
                r'score[:\s]+([0-9.]+)'
            ],
            'reasoning': [
                r'reasoning[:\s]+([^\n\r]+)',
                r'explanation[:\s]+([^\n\r]+)',
                r'rationale[:\s]+([^\n\r]+)'
            ],
            'tags': [
                r'tags[:\s]+\[([^\]]+)\]',
                r'keywords[:\s]+\[([^\]]+)\]',
                r'tags[:\s]+([^\n\r]+)'
            ]
        }
        
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    
                    # Special handling for different field types
                    if field == 'confidence':
                        try:
                            result[field] = float(value)
                        except ValueError:
                            result[field] = 0.5  # Default confidence
                    elif field == 'tags':
                        if value.startswith('[') and value.endswith(']'):
                            # Try to parse as list
                            try:
                                result[field] = json.loads(value)
                            except:
                                # Split by comma
                                result[field] = [tag.strip().strip('"\'') for tag in value.strip('[]').split(',')]
                        else:
                            # Split by comma or space
                            result[field] = [tag.strip().strip('"\'') for tag in re.split(r'[,;]', value)]
                    else:
                        # Clean quotes from string values
                        result[field] = value.strip('"\'')
                    break
        
        # Return result only if we found some useful information
        if len(result) >= 2:  # At least category and one other field
            # Fill in defaults for missing required fields
            result.setdefault('category', 'Other')
            result.setdefault('subcategory', 'General')
            result.setdefault('confidence', 0.5)
            result.setdefault('tags', [])
            result.setdefault('reasoning', 'Extracted from structured text')
            
            return result
        
        return None
    
    def parse_classification_response(self, response_text: str, fallback_category: str = "Other") -> Dict[str, Any]:
        """
        Parse classification response with guaranteed return of valid structure.
        
        Args:
            response_text: AI model response
            fallback_category: Category to use if parsing fails
            
        Returns:
            Valid classification dictionary
        """
        parsed = self.extract_json_from_response(response_text)
        
        if parsed:
            # Validate and clean the parsed result
            result = {
                'category': str(parsed.get('category', fallback_category)).strip(),
                'subcategory': str(parsed.get('subcategory', 'General')).strip(),
                'confidence': float(parsed.get('confidence', 0.5)),
                'tags': self._ensure_list(parsed.get('tags', [])),
                'reasoning': str(parsed.get('reasoning', 'AI classification')).strip()
            }
            
            # Validate confidence is between 0 and 1
            result['confidence'] = max(0.0, min(1.0, result['confidence']))
            
            return result
          # Fallback: create default classification
        self.logger.warning(f"Failed to parse classification response, using fallback: {fallback_category}")
        
        # Log the problematic response for debugging (truncated)
        response_preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
        self.logger.debug(f"Problematic response: {response_preview}")
        return {
            'category': fallback_category,
            'subcategory': 'General',
            'confidence': 0.3,
            'tags': [],
            'reasoning': 'Parsing failed, using fallback classification'
        }
    
    def parse_summarization_response(self, response_text: str, original_title: str = "") -> Dict[str, Any]:
        """
        Parse summarization response with guaranteed return of valid structure.
        
        Args:
            response_text: AI model response
            original_title: Original content title for fallback
            
        Returns:
            Valid summarization dictionary
        """
        parsed = self.extract_json_from_response(response_text)
        
        if parsed:
            # Handle both news and paper summaries
            result = {
                'title': original_title,
                'summary': str(parsed.get('summary', 'Summary not available')).strip(),
                'key_points': self._ensure_list(parsed.get('key_points', [])),
                'impact': str(parsed.get('impact', 'Impact analysis not available')).strip(),
                'technical_level': str(parsed.get('technical_level', 'intermediate')).strip(),
                'confidence': float(parsed.get('confidence', 0.5))
            }
            
            # Add paper-specific fields if present
            if 'problem' in parsed:
                result['problem'] = str(parsed['problem']).strip()
            if 'method' in parsed:
                result['method'] = str(parsed['method']).strip()
            if 'results' in parsed:
                result['results'] = str(parsed['results']).strip()
            if 'applications' in parsed:
                result['applications'] = str(parsed['applications']).strip()
            
            # Validate confidence
            result['confidence'] = max(0.0, min(1.0, result['confidence']))
            
            return result
        
        # Fallback: create basic summary from response text
        self.logger.warning("Failed to parse summarization response, creating basic summary")
        
        # Try to extract useful information from the raw text
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        summary_text = ' '.join(lines[:3]) if lines else "Summary not available"
        
        return {
            'title': original_title or 'Unknown Title',
            'summary': summary_text[:500] + ('...' if len(summary_text) > 500 else ''),
            'key_points': self._extract_bullet_points(response_text),
            'impact': 'Impact analysis not available',
            'technical_level': 'intermediate',
            'confidence': 0.2
        }
    
    def _ensure_list(self, value: Any) -> List[str]:
        """
        Ensure a value is a list of strings.
        
        Args:
            value: Value to convert to list
            
        Returns:
            List of strings
        """
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str):
            if value.strip():
                # Try to split by common delimiters
                items = re.split(r'[,;]', value)
                return [item.strip().strip('"\'') for item in items if item.strip()]
            return []
        else:
            return []
    
    def _extract_bullet_points(self, text: str) -> List[str]:
        """
        Extract bullet points from text.
        
        Args:
            text: Text to extract bullet points from
            
        Returns:
            List of bullet points
        """
        bullet_patterns = [
            r'[•·▪▫‣⁃]\s+(.+)',
            r'[-*]\s+(.+)',
            r'\d+\.\s+(.+)',
            r'[→➤➔]\s+(.+)'
        ]
        
        points = []
        for pattern in bullet_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            points.extend([match.strip() for match in matches])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_points = []
        for point in points:
            if point not in seen and len(point) > 10:  # Filter out very short points
                seen.add(point)
                unique_points.append(point)
        
        return unique_points[:5]  # Limit to 5 points


# Global instance
robust_parser = RobustJsonParser()
