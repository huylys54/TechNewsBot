"""
Enhanced Discord message tracking system for accurate URL capture.
"""

import logging
import time
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import yaml
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

load_dotenv()

@dataclass
class MessageInfo:
    """Information about a sent Discord message."""
    content_preview: str
    category: str
    message_id: str
    channel_id: str
    guild_id: str
    timestamp: float
    
    @property
    def discord_url(self) -> str:
        """Get the full Discord URL for this message."""
        return f"https://discord.com/channels/{self.guild_id}/{self.channel_id}/{self.message_id}"


class DiscordMessageTracker:
    """
    Enhanced Discord webhook system with comprehensive message tracking.
    """
    
    def __init__(self, config_path: str = "config/notifications.yml"):
        """Initialize the message tracker."""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.webhook_url = self._prepare_webhook_url()
        self.guild_id = os.getenv('DISCORD_GUILD_ID', "")
        self.channel_id = os.getenv('DISCORD_CHANNEL_ID', "")
        self.sent_messages: List[MessageInfo] = []
        self.monthly_papers_config_path = "config/monthly_papers.yml"
        self.monthly_papers_config = self._load_monthly_papers_config()

        # Validate configuration
        if not self.webhook_url:
            self.logger.error("Discord webhook URL not configured")
        if not self.guild_id or not self.channel_id:
            self.logger.warning("Guild ID or Channel ID not configured - URLs may not work")

    def _load_monthly_papers_config(self) -> Dict[str, Any]:
        """Load monthly papers configuration from file."""
        try:
            if os.path.exists(self.monthly_papers_config_path):
                with open(self.monthly_papers_config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            return {}
        except Exception as e:
            self.logger.warning(f"Failed to load monthly papers config from {self.monthly_papers_config_path}: {e}")
            return {}

    def update_last_monthly_papers_sent(self):
        """Update the last_monthly_papers_sent timestamp in the config file."""
        try:
            timestamp = datetime.now().isoformat()
            self.monthly_papers_config['last_monthly_papers_sent'] = timestamp
            
            p = Path(self.monthly_papers_config_path)
            if not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.monthly_papers_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.monthly_papers_config, f)
            self.logger.info(f"Updated last_monthly_papers_sent to {timestamp}")
        except Exception as e:
            self.logger.error(f"Failed to update last_monthly_papers_sent: {e}")
            
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            return {}
        except Exception as e:
            self.logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def _prepare_webhook_url(self) -> str:
        """Prepare webhook URL with wait parameter for message ID capture."""
        webhook_url = (
            self.config.get("discord", {}).get("webhook_url", "") or 
            os.getenv("DISCORD_WEBHOOK_URL", "")
        )
        
        if not webhook_url:
            return ""
        
        # Ensure ?wait=true is present for message ID capture
        if "?wait=true" not in webhook_url:
            separator = "&" if "?" in webhook_url else "?"
            webhook_url += f"{separator}wait=true"
        
        return webhook_url
    
    def send_message(self, content: str, category: str = "General", 
                    username: Optional[str] = None, 
                    retry_count: int = 3) -> Tuple[bool, Optional[MessageInfo]]:
        """
        Send a message to Discord and track the response.
        
        Args:
            content: Message content to send
            category: Category name for tracking
            username: Optional username override
            retry_count: Number of retries on failure
            
        Returns:
            Tuple of (success: bool, message_info: Optional[MessageInfo])
        """
        if not self.webhook_url:
            self.logger.error("Cannot send message - webhook URL not configured")
            return False, None
        
        if not content.strip():
            self.logger.error("Cannot send empty message")
            return False, None
        
        payload = {
            "content": content,
            "username": username or self.config.get("discord", {}).get("username", "TechNews Bot")
        }
        
        for attempt in range(retry_count + 1):
            try:
                self.logger.info(f"Sending Discord message for category '{category}' (attempt {attempt + 1}/{retry_count + 1})")
                
                response = requests.post(
                    self.webhook_url, 
                    json=payload, 
                    timeout=30,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    message_data = response.json()
                    message_info = self._process_message_response(message_data, content, category)
                    
                    if message_info:
                        self.sent_messages.append(message_info)
                        self.logger.info(f"✅ Message sent successfully: {message_info.discord_url}")
                        return True, message_info
                    else:
                        self.logger.warning("Message sent but could not extract message info")
                        return True, None
                
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = response.headers.get("Retry-After", "1")
                    wait_time = int(retry_after) + 1
                    self.logger.warning(f"Rate limited, waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    self.logger.error(f"Discord API error: {response.status_code} - {response.text}")
                    if attempt < retry_count:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return False, None
                    
            except requests.exceptions.Timeout:
                self.logger.error(f"Request timeout on attempt {attempt + 1}")
                if attempt < retry_count:
                    time.sleep(2 ** attempt)
                    continue
                return False, None
                
            except Exception as e:
                self.logger.error(f"Error sending message on attempt {attempt + 1}: {e}")
                if attempt < retry_count:
                    time.sleep(2 ** attempt)
                    continue
                return False, None
        
        return False, None
    
    def _process_message_response(self, message_data: Dict[str, Any], 
                                 content: str, category: str) -> Optional[MessageInfo]:
        """
        Process the Discord API response to extract message information.
        
        Args:
            message_data: Response JSON from Discord API
            content: Original message content
            category: Message category
            
        Returns:
            MessageInfo object or None if processing fails
        """
        try:
            message_id = message_data.get("id")
            channel_id = message_data.get("channel_id")
            
            if not message_id:
                self.logger.error("No message ID in Discord response")
                return None
            
            # Use channel_id from response if available, otherwise use config
            actual_channel_id = channel_id or self.channel_id
            
            if not actual_channel_id:
                self.logger.error("No channel ID available")
                return None
            
            # Create content preview (first 100 chars)
            content_preview = content[:100] + ("..." if len(content) > 100 else "")
            
            return MessageInfo(
                content_preview=content_preview,
                category=category,
                message_id=message_id,
                channel_id=actual_channel_id,
                guild_id=self.guild_id,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process message response: {e}")
            return None
    
    def send_messages_batch(self, messages: List[Tuple[str, str]], 
                           delay: float = 1.5) -> Dict[str, str]:
        """
        Send multiple messages with automatic tracking.
        
        Args:
            messages: List of (content, category) tuples
            delay: Delay between messages in seconds
            
        Returns:
            Dictionary mapping category names to Discord URLs
        """
        category_urls = {}
        
        for i, (content, category) in enumerate(messages):
            success, message_info = self.send_message(content, category)
            
            if success and message_info:
                category_urls[category] = message_info.discord_url
                self.logger.info(f"Tracked message {i+1}/{len(messages)}: {category}")
            else:
                self.logger.error(f"Failed to send/track message {i+1}/{len(messages)}: {category}")
            
            # Add delay between messages (except for the last one)
            if i < len(messages) - 1:
                time.sleep(delay)
        
        return category_urls
