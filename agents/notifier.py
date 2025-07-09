"""
Notification system for delivering tech news digests via Discord and email.
Supports multiple delivery channels with customizable formatting.
"""

import logging
import os
import smtplib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import glob
import json
import yaml
import discord
from discord import SyncWebhook
from dataclasses import dataclass
from dotenv import load_dotenv
from utils.message_tracker import DiscordMessageTracker

# Load environment variables from .env file
load_dotenv()


class DiscordWebhookWithMessageIds:
    """    Discord webhook sender that captures real message IDs using ?wait=true parameter.
    """
    
    def __init__(self, config_path: str = "config/notifications.yml"):
        """Initialize the webhook sender."""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.webhook_url = self._get_webhook_url()
        self.guild_id = os.getenv("DISCORD_GUILD_ID", "")
        self.channel_id = os.getenv("DISCORD_CHANNEL_ID", "")
        self.webhook: Optional[SyncWebhook] = None
        if self.webhook_url:
            try:
                self.webhook = SyncWebhook.from_url(self.webhook_url)
            except Exception as e:
                self.logger.error(f"Failed to initialize Discord webhook: {e}")
                self.webhook = None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            return {}
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}")
            return {}
    
    def _get_webhook_url(self) -> str:
        """Get webhook URL from environment variables (preferred) or config."""
        # Always prioritize environment variable
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        
        # Fallback to config file if env var is not set
        if not webhook_url:
            webhook_url = self.config.get("discord", {}).get("webhook_url", "")
        
        # discord.py handles ?wait=true automatically
        return webhook_url
    
    def send_message_with_id(self, content: str, category_name: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Send a message and return the Discord URL and message ID.
        
        Args:
            content: Message content to send.
            category_name: Optional category name for logging.
            
        Returns:
            Tuple of (success: bool, discord_url: Optional[str], message_id: Optional[int])
        """
        if not self.webhook:
            self.logger.error("Webhook not initialized or URL not configured")
            return False, None, None
        
        username = self.config.get("discord", {}).get("username", "TechNews Bot")
        
        try:
            message = self.webhook.send(content=content, username=username, wait=True)
            
            if message and message.id:
                # The message object from discord.py contains channel and guild info
                # However, for webhooks, guild_id might not be directly available on the message object
                # We rely on the pre-configured guild_id and the message's channel_id
                
                # If guild_id is not set in env, we can't form the full URL
                if self.guild_id and message.channel.id:
                    discord_url = f"https://discord.com/channels/{self.guild_id}/{message.channel.id}/{message.id}"
                    self.logger.info(f"Message sent successfully for {category_name or 'unknown'}: {discord_url}")
                    return True, discord_url, message.id
                else:
                    self.logger.warning(f"Message sent but could not generate full URL (missing guild_id or channel_id)")
                    return True, None, message.id # Still return success and message_id
            else:
                self.logger.error(f"Failed to send message: No message object or ID returned.")
                return False, None, None
                
        except discord.errors.HTTPException as e:
            self.logger.error(f"Discord HTTP error sending message: {e.status} - {e.text}")
            return False, None, None
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False, None, None
    
    def edit_message(self, message_id: int, content: str) -> bool:
        """
        Edit an existing Discord message.
        
        Args:
            message_id: The ID of the message to edit.
            content: The new content for the message.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.webhook:
            self.logger.error("Webhook not initialized or URL not configured")
            return False
        
        try:
            # Webhooks can edit their own messages by ID
            self.webhook.edit_message(message_id, content=content)
            self.logger.info(f"Message {message_id} edited successfully.")
            return True
        except discord.errors.HTTPException as e:
            self.logger.error(f"Discord HTTP error editing message {message_id}: {e.status} - {e.text}")
            return False
        except Exception as e:
            self.logger.error(f"Error editing message {message_id}: {e}")
            return False

    def send_messages_batch(self, messages: List[Tuple[str, str]], delay: float = 1.0) -> Dict[str, Tuple[str, int]]:
        """
        Send multiple messages and return their Discord URLs and message IDs.
        
        Args:
            messages: List of (content, category_name) tuples.
            delay: Delay between messages in seconds.
            
        Returns:
            Dictionary mapping category names to (Discord URL, message ID) tuples.
        """
        results = {}
        
        for i, (content, category_name) in enumerate(messages):
            success, discord_url, message_id = self.send_message_with_id(content, category_name)
            
            if success and (discord_url or message_id): # message_id is crucial for editing, even if URL is not formed
                results[category_name] = (discord_url, message_id)
            
            # Add delay between messages (except for the last one)
            if i < len(messages) - 1:
                import time
                time.sleep(delay)
        
        return results


@dataclass
class DiscordDigestItem:
    title: str
    summary: str
    category: str
    url: str = ""
    tags: Optional[List[str]] = None
    key_points: Optional[List[str]] = None
    impact: str = ""
    source_type: str = ""
    source_info: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.key_points is None:
            self.key_points = []


@dataclass
class DiscordDigest:
    """Data class for complete Discord digest."""
    title: str
    date: str
    categories_news: Dict[str, List[DiscordDigestItem]]  # News organized by category
    categories_papers: Dict[str, List[DiscordDigestItem]]  # Papers organized by category
    statistics: Dict[str, Any]
    all_categories: List[str]


class NotificationConfig:
    """Configuration data class for notifications."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.discord = config_dict.get("discord", {})
        self.email = config_dict.get("email", {})
        self.formatting = config_dict.get("formatting", {})
        self.delivery = config_dict.get("delivery", {})


class TechNewsNotifier:
    """
    Multi-channel notification system for tech news digests.
    Supports Discord webhooks and email delivery.
    """
    def __init__(self, config_path: str = "config/notifications.yml"):
        """
        Initialize the notifier.
        
        Args:
            config_path: Path to notifications configuration file.
        """
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.config = self._load_config()
        # Initialize enhanced webhook for capturing Discord message IDs
        self.message_tracker = DiscordMessageTracker(config_path)
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the notifier."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self) -> NotificationConfig:
        """Load notification configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
                return NotificationConfig(config_dict)
            else:
                return NotificationConfig(self._get_default_config())
        except Exception as e:
            self.logger.warning(f"Failed to load config from {self.config_path}: {e}")
            return NotificationConfig(self._get_default_config())
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default notification configuration."""
        return {
            "discord": {
                "enabled": False,
                "webhook_url": "",
                "username": "Tech News Bot",
                "avatar_url": "",
                "max_message_length": 2000,
                "use_embeds": True,
                "embed_color": 0x00ff00,
                "split_long_messages": True
            },
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_address": "",
                "to_addresses": [],
                "subject_format": "Tech News Digest - {date}",
                "use_html": True,
                "attach_markdown": False
            },
            "formatting": {
                "discord_max_items": 5,
                "email_max_items": 20,
                "include_statistics": True,
                "include_summary": True,
                "truncate_descriptions": True,
                "max_description_length": 200
            },
            "delivery": {
                "retry_attempts": 3,
                "retry_delay": 5,
                "timeout": 30,
                "verify_ssl": True
            }
        }
    
    
    
    
    
    def send_email_notification(self, 
                               digest_content: str, 
                               title: str = "Tech News Digest",
                               report_path: Optional[str] = None) -> bool:
        """
        Send digest via email.
        
        Args:
            digest_content: Content of the digest.
            title: Subject line for the email.
            report_path: Optional path to report file to attach.
              Returns:
            True if successful, False otherwise.
        """
        if not self.config.email.get("enabled", False):
            self.logger.info("Email notifications are disabled")
            return False
        
        try:
            # Get email configuration - prioritize environment variables
            smtp_server = self.config.email.get("smtp_server", "smtp.gmail.com")
            smtp_port = self.config.email.get("smtp_port", 587)
            username = os.getenv("EMAIL_USERNAME", "") or self.config.email.get("username", "")
            password = os.getenv("EMAIL_PASSWORD", "") or self.config.email.get("password", "")
            from_address = os.getenv("EMAIL_FROM_ADDRESS", "") or self.config.email.get("from_address", "") or username
            
            # Handle comma-separated email addresses from environment variable
            env_to_addresses = os.getenv("EMAIL_TO_ADDRESSES", "")
            if env_to_addresses:
                to_addresses = [addr.strip() for addr in env_to_addresses.split(",") if addr.strip()]
            else:
                to_addresses = self.config.email.get("to_addresses", [])
            
            if not username or not password:
                self.logger.error("Email credentials not configured")
                return False
            
            if not to_addresses:
                self.logger.error("No recipient email addresses configured")
                return False
            
            # Create message
            msg = MIMEMultipart('alternative')
            subject_format = self.config.email.get("subject_format", "Tech News Digest - {date}")
            msg['Subject'] = subject_format.format(date=datetime.now().strftime("%Y-%m-%d"))
            msg['From'] = from_address
            msg['To'] = ', '.join(to_addresses)
            
            # Format content for email
            if self.config.email.get("use_html", True):
                html_content = self._markdown_to_html(digest_content)
                html_part = MIMEText(html_content, 'html')
                msg.attach(html_part)
            else:
                text_part = MIMEText(digest_content, 'plain')
                msg.attach(text_part)
            
            # Attach report file if requested
            if report_path and os.path.exists(report_path) and self.config.email.get("attach_markdown", False):
                with open(report_path, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {os.path.basename(report_path)}'
                    )
                    msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent to {len(to_addresses)} recipients")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _markdown_to_html(self, markdown_content: str) -> str:
        """Convert markdown content to HTML for email."""
        # Simple markdown to HTML conversion
        html = markdown_content
        
        # Convert headers
        html = html.replace('# ', '<h1>').replace('\n# ', '</h1>\n<h1>')
        html = html.replace('## ', '<h2>').replace('\n## ', '</h1>\n<h2>')
        html = html.replace('### ', '<h3>').replace('\n### ', '</h1>\n<h3>')
        
        # Convert bold text
        import re
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        
        # Convert links
        html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
        
        # Convert line breaks
        html = html.replace('\n\n', '</p>\n<p>')
        html = html.replace('\n', '<br>\n')
        
        # Wrap in HTML structure
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
                h1 {{ color: #333; border-bottom: 2px solid #4CAF50; }}
                h2 {{ color: #555; margin-top: 30px; }}
                h3 {{ color: #777; }}
                .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .category {{ margin-bottom: 30px; }}
                hr {{ border: 1px solid #ddd; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <p>{html}</p>
        </body>
        </html>
        """
        
        return html
    
    def send_notifications(self, 
                          digest_content: str, 
                          title: str = "Tech News Digest",
                          report_path: Optional[str] = None) -> Dict[str, bool]:
        """
        Send notifications via all enabled channels.
        
        Args:
            digest_content: Content of the digest.
            title: Title for notifications.
            report_path: Optional path to report file.
            
        Returns:
            Dictionary with success status for each channel.
        """
        results = {}
        
        # Send Discord notification
        if self.config.discord.get("enabled", False):
            # This is now handled by EnhancedDiscordDigestBot
            pass
        
        # Send email notification
        if self.config.email.get("enabled", False):
            results["email"] = self.send_email_notification(digest_content, title, report_path)
        
        # Log results
        successful_channels = [channel for channel, success in results.items() if success]
        failed_channels = [channel for channel, success in results.items() if not success]
        
        if successful_channels:
            self.logger.info(f"Notifications sent successfully via: {', '.join(successful_channels)}")
        
        if failed_channels:
            self.logger.warning(f"Failed to send notifications via: {', '.join(failed_channels)}")
        
        return results
    
    def send_digest_from_file(self, report_path: str) -> Dict[str, bool]:
        """
        Send notifications using a digest report file.
        
        Args:
            report_path: Path to the markdown report file.
            
        Returns:
            Dictionary with success status for each channel.
        """
        try:
            if not os.path.exists(report_path):
                self.logger.error(f"Report file not found: {report_path}")
                return {}
            
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title from content
            lines = content.split('\n')
            title = "Tech News Digest"
            for line in lines:
                if line.startswith('# '):
                    title = line[2:].strip()
                    break
            
            return self.send_notifications(content, title, report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to send digest from file: {e}")
            return {}
    


class EnhancedDiscordDigestBot:
    """
    Enhanced Discord digest system that creates beautiful, user-friendly
    notifications with top news and papers.
    """
    
    def __init__(self, config_path: str = "config/notifications.yml"):
        """
        Initialize the enhanced Discord digest bot.
        
        Args:
            config_path: Path to notifications configuration file.
        """
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.config = self._load_config()
        self.webhook_url = self._get_webhook_url()
        # Initialize enhanced webhook for capturing Discord message IDs
        self.enhanced_webhook = DiscordWebhookWithMessageIds(config_path)
        # Initialize message tracker for better URL management
        self.message_tracker = DiscordMessageTracker(config_path)
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the digest bot."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            return self._get_default_config()
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "discord": {
                "enabled": True,
                "use_embeds": False,
                "username": "TechNews Digest Bot",
                "max_message_length": 2000,
                "max_items_per_category": 5
            }
        }
    
    def _get_webhook_url(self) -> str:
        """Get Discord webhook URL from config or environment."""
        webhook_url = self.config.get("discord", {}).get("webhook_url", "")
        if not webhook_url:
            webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        return webhook_url
    
    def create_digest_from_latest_files(self) -> Optional[DiscordDigest]:
        """
        Create a digest from the latest generated files.
        
        Returns:
            DiscordDigest object or None if no files found.
        """
        try:
            # Find latest files
            latest_files = self._find_latest_files()
            if not latest_files:
                self.logger.warning("No latest files found")
                return None
            
            # Load and process data
            digest_data = self._load_digest_data(latest_files)
            return self._create_digest_from_data(digest_data)
            
        except Exception as e:
            self.logger.error(f"Failed to create digest from files: {e}")
            return None
    
    def _find_latest_files(self) -> Dict[str, str]:
        """Find the latest generated files."""
       
        files = {}
        
        # Find latest summary file
        summary_files = glob.glob("data/summaries/summaries_*.json")
        if summary_files:
            files["summaries"] = max(summary_files, key=os.path.getctime)
            self.logger.info(f"Found latest summaries file: {files['summaries']}")
        
        # Find latest classification file
        class_files = glob.glob("data/classifications/classifications_*.json")
        if class_files:
            files["classifications"] = max(class_files, key=os.path.getctime)
            self.logger.info(f"Found latest classifications file: {files['classifications']}")
        
        # Find latest content file
        content_files = glob.glob("data/content/processed_*_combined.json")
        if content_files:
            files["content"] = max(content_files, key=os.path.getctime)
            self.logger.info(f"Found latest content file: {files['content']}")
        
        self.logger.info(f"Total files found: {len(files)}")
        return files
    
    def _load_digest_data(self, file_paths: Dict[str, str]) -> Dict[str, Any]:
        """Load data from files."""
        data = {}
        
        for data_type, file_path in file_paths.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data[data_type] = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load {data_type} from {file_path}: {e}")
                data[data_type] = []
        
        return data
    
    def _create_digest_from_data(self, data: Dict[str, Any]) -> DiscordDigest:
        """Create a Discord digest from loaded data."""
        summaries = data.get("summaries", [])
        classifications = data.get("classifications", [])
        
        items = []
        
        # Process summaries data
        for i, summary_item in enumerate(summaries):
            classification = classifications[i] if i < len(classifications) else {}
            classification_data = classification.get("classification", {})
            
            source_type = self._determine_source_type(summary_item, summary_item.get("summary_data", {}))
            # Try both 'link' and 'url' keys for robustness
            url = summary_item.get("link", "") or summary_item.get("url", "")
            
            # Handle both old and new summary formats
            summary_data = summary_item.get("summary_data", {})
            if not summary_data and isinstance(summary_item.get("summary"), dict):
                # Old format: summary was the structured data
                summary_data = summary_item.get("summary", {})
            
            item = DiscordDigestItem(
                title=summary_data.get("title", summary_item.get("title", "Untitled")),
                summary=summary_data.get("summary", summary_item.get("summary", "")),
                category=classification_data.get("category", "Other"),
                url=url,
                tags=classification_data.get("tags", []),
                key_points=summary_data.get("key_points", []),
                impact=summary_data.get("impact", "")
            )
            
            item.source_type = source_type
            # Store original source information for paper categorization
            item.source_info = summary_item.get("source", "")
            items.append(item)
        
        # Separate news and papers
        news_items = [item for item in items if item.source_type == "news"]
        paper_items = [item for item in items if item.source_type == "paper"]
        
        # Filter to tech-only content
        tech_news_items = [item for item in news_items if self._is_tech_item(item)]
        tech_paper_items = [item for item in paper_items if self._is_tech_item(item)]
        
        self.logger.info(f"Tech filtering: {len(tech_news_items)}/{len(news_items)} news, {len(tech_paper_items)}/{len(paper_items)} papers")

        # Organize tech news by categories (top 3 per category)
        categories_news = {}
        for item in tech_news_items:
            category = item.category
            if category not in categories_news:
                categories_news[category] = []
            categories_news[category].append(item)
        
        # Sort each category and keep top 3 per category
        for category in categories_news:
            categories_news[category] = categories_news[category][:3]        
        # Log final category distribution
        if categories_news:
            self.logger.info(f"Final categories: {list(categories_news.keys())}")
            for category, items in categories_news.items():
                self.logger.info(f"  {category}: {len(items)} articles")
        else:
            self.logger.warning("No categories with news items after filtering")
        
        # Organize tech papers by categories (top 5 per category)
        categories_papers = {}
        for item in tech_paper_items:
            category = item.category
            if category not in categories_papers:
                categories_papers[category] = []
            categories_papers[category].append(item)
        
        # Sort each category and keep top 5 (removed confidence sorting)
        for category in categories_papers:
            categories_papers[category] = categories_papers[category][:5]
          # Ensure minimum paper counts: at least 5 HuggingFace papers and 2 arXiv papers
        self._ensure_minimum_papers(categories_papers, tech_paper_items)
        
        # Log final paper distribution
        if categories_papers:
            self.logger.info(f"Final paper categories: {list(categories_papers.keys())}")
            for category, items in categories_papers.items():
                self.logger.info(f"  {category}: {len(items)} papers")
        
        # Generate statistics(removed confidence-related stats)
        all_tech_items = tech_news_items + tech_paper_items
        all_categories = list(set([item.category for item in all_tech_items]))

        # Calculate category counts for news and papers
        categories_news_count = {category: len(items) for category, items in categories_news.items()}
        categories_papers_count = {category: len(items) for category, items in categories_papers.items()}

        statistics = {
            "total_news": len(tech_news_items),
            "total_papers": len(tech_paper_items),
            "total_items": len(all_tech_items),
            "total_filtered": len(items) - len(all_tech_items),
            "categories": all_categories,
            "categories_news_count": categories_news_count,
            "categories_papers_count": categories_papers_count
        }
        
        return DiscordDigest(
            title=f"🚀 Tech News & Research Digest - {datetime.now().strftime('%B %d, %Y')}",
            date=datetime.now().strftime('%Y-%m-%d'),
            categories_news=categories_news,
            categories_papers=categories_papers,
            statistics=statistics,
            all_categories=all_categories
        )
    
    def generate_and_send_digest(self) -> tuple[bool, Dict[str, str], Dict[str, str]]:
        """
        Generate and send a complete digest from latest files.
        
        Returns:
            Tuple of (success: bool, category_urls: Dict[str, str], paper_section_urls: Dict[str, str])
            category_urls maps category names to Discord message URLs
            paper_section_urls maps paper section names to Discord message URLs
        """
        self.logger.info("Generating and sending enhanced Discord digest...")
        
        digest = self.create_digest_from_latest_files()
        if not digest:
            self.logger.error("Failed to create digest")
            return False, {}, {}
        
        success, category_urls, paper_section_urls = self.send_discord_digest(digest)
        
        # The category_urls and paper_section_urls are now real Discord URLs from the enhanced webhook
        if success:
            total_news = sum(len(items) for items in digest.categories_news.values())
            total_papers = sum(len(items) for items in digest.categories_papers.values())
            self.logger.info(f"Successfully sent digest with {total_news} news and {total_papers} papers")
        
        return success, category_urls, paper_section_urls

    def send_empty_digest_notice(self, empty_message: Dict) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Sends a notification when no new content is found.

        Args:
            empty_message: A dictionary containing the formatted empty day message.

        Returns:
            Tuple of (success: bool, discord_url: Optional[str], message_id: Optional[int])
        """
        if not self.config.get("discord", {}).get("enabled", False):
            self.logger.info("Discord notifications are disabled, skipping empty digest notice.")
            return False, None, None

        self.logger.info("Sending empty digest notice to Discord...")
        content = empty_message.get("message", "No new tech content found today.")
        title = empty_message.get("title", "No New Tech Content Found")
        
        # Use the enhanced webhook to send the message
        success, discord_url, message_id = self.enhanced_webhook.send_message_with_id(
            content=f"**{title}**\n\n{content}", # Format for direct message
            category_name="Empty Digest Notice"
        )
        
        if success:
            self.logger.info(f"Empty digest notice sent successfully. URL: {discord_url}")
        else:
            self.logger.error("Failed to send empty digest notice.")
            
        return success, discord_url, message_id
    
    def _determine_source_type(self, content_item: Dict, summary: Dict) -> str:
        """Determine if an item is news or paper."""
        content_type = content_item.get("type", "")
        if content_type == "arxiv_paper":
            return "paper"
        elif content_type == "news_article":
            return "news"
        
        url = content_item.get("link", content_item.get("url", "")).lower()
        title = summary.get("title", content_item.get("title", "")).lower()
        
        if "arxiv" in url or "arxiv" in title:
            return "paper"
        
        news_domains = ["techcrunch", "theverge", "wired", "engadget", "arstechnica"]
        if any(domain in url for domain in news_domains):
            return "news"
        
        return "news"
    
    def _is_tech_category(self, category: str) -> bool:
        """Check if a category is tech-related."""
        return category in self._get_tech_categories()
    
    def _is_tech_item(self, item: DiscordDigestItem) -> bool:
        """
        Check if an item is tech-related based on its category or tags.
        
        Args:
            item: The DiscordDigestItem to evaluate.
        
        Returns:
            bool: True if the item is tech-related, False otherwise.
        """
        # Check if the category is tech-related
        if self._is_tech_category(item.category):
            return True
        
        # Get the list of tech categories as lowercase tags for matching
        tech_tags = {cat.lower() for cat in self._get_tech_categories()}
          # Check if any tag matches a tech category
        if item.tags:
            for tag in item.tags:
                if tag.lower() in tech_tags:
                    return True
        
        return False

    def _get_tech_categories(self) -> List[str]:
        """
        Return the list of tech-related categories.
        
        Returns:
            List[str]: List of tech category names.
        """
        return [
            # Core AI/ML categories
            "Artificial Intelligence", "Machine Learning", "Deep Learning",
            "Computer Vision", "Natural Language Processing", "Robotics",
            
            # Software Development categories
            "Software Development", "Programming", "Web Development",
            "Mobile Development", "DevOps", "Programming Languages",
            "Frameworks",
              # Infrastructure categories
            "Cloud Computing", "Hardware", "Hardware & Systems", "Cybersecurity",
            "Data Science", "Big Data", "Network Security",
            "Cloud Computing & Infrastructure",
            
            # Emerging Tech categories
            "Blockchain", "Cryptocurrency", "Internet of Things",
            "AR/VR", "Quantum Computing",
            
            # Business/Industry categories
            "Startups & Business", "Social Media & Platforms",
            "Gaming", "Technology",
            
            # General categories
            "Computing", "Software", "Computer Science",
            "Information Technology", "Research & Academia", "Other"
        ]
    
    def _ensure_minimum_papers(self, categories_papers: Dict[str, List[DiscordDigestItem]], 
                              all_tech_papers: List[DiscordDigestItem]) -> None:
        """
        Ensure minimum paper counts: at least 5 HuggingFace papers and 2 arXiv papers.
        
        Args:
            categories_papers: Papers organized by category
            all_tech_papers: All tech paper items available
        """
        # Count current papers by source
        current_hf_count = 0
        current_arxiv_count = 0
        
        for category, papers in categories_papers.items():
            for paper in papers:
                if "HuggingFace" in paper.source_info:
                    current_hf_count += 1
                else:
                    current_arxiv_count += 1
        
        self.logger.info(f"Current paper counts: {current_hf_count} HuggingFace, {current_arxiv_count} arXiv")
        
        # If we need more papers, add them from available items
        if current_hf_count < 5 or current_arxiv_count < 2:
            # Get all available papers by source type
            available_hf = [p for p in all_tech_papers if "HuggingFace" in p.source_info]
            available_arxiv = [p for p in all_tech_papers if "HuggingFace" not in p.source_info]
            
            # Get papers already included
            included_papers = []
            for papers in categories_papers.values():
                included_papers.extend(papers)
            included_titles = {p.title for p in included_papers}
            
            # Add more HuggingFace papers if needed (up to 5)
            if current_hf_count < 5:
                needed_hf = 5 - current_hf_count
                additional_hf = [p for p in available_hf if p.title not in included_titles][:needed_hf]
                
                for paper in additional_hf:
                    category = paper.category
                    if category not in categories_papers:
                        categories_papers[category] = []
                    # Add to category if not already at limit
                    if len(categories_papers[category]) < 7:  # Allow up to 7 per category for minimums
                        categories_papers[category].append(paper)
                        current_hf_count += 1
                        self.logger.info(f"Added HuggingFace paper: {paper.title}")
            
            # Add more arXiv papers if needed (up to 2 minimum)
            if current_arxiv_count < 2:
                needed_arxiv = 2 - current_arxiv_count
                additional_arxiv = [p for p in available_arxiv if p.title not in included_titles][:needed_arxiv]
                
                for paper in additional_arxiv:
                    category = paper.category
                    if category not in categories_papers:
                        categories_papers[category] = []                    # Add to category if not already at limit
                    if len(categories_papers[category]) < 7:  # Allow up to 7 per category for minimums
                        categories_papers[category].append(paper)
                        current_arxiv_count += 1
                        self.logger.info(f"Added arXiv paper: {paper.title}")
        
        self.logger.info(f"Final paper counts: {current_hf_count} HuggingFace, {current_arxiv_count} arXiv")
    
    def send_discord_digest(self, digest: DiscordDigest) -> Tuple[bool, Dict[str, str], Dict[str, str]]:
        """
        Send Discord digest using the simplified send_messages_batch method.
        
        Args:
            digest: DiscordDigest object with categorized content
            
        Returns:
            Tuple of (success: bool, category_urls: Dict[str, str], paper_section_urls: Dict[str, str])
        """
        if not self.config.get("discord", {}).get("enabled", False):
            self.logger.info("Discord notifications are disabled")
            return False, {}, {}
        
        try:
            # Prepare all messages to send
            messages_to_send = []
            
            # Emoji mapping for categories
            emoji_map = {
                "Artificial Intelligence": "🤖",
                "Software Development": "💻", 
                "Hardware": "🔧",
                "Cybersecurity": "🔒",
                "Cloud Computing": "☁️",
                "Startups & Business": "🚀",
                "Social Media & Platforms": "📱",
                "Gaming": "🎮",
                "Research & Academia": "🎓",
                "Other": "📰"
            }

            # 1. Send statistics message
            stats_content = self._format_statistics(digest)
            stats_content = self._ensure_message_length(stats_content)
            success, _, stats_message_id = self.enhanced_webhook.send_message_with_id(
                stats_content, "Digest Statistics"
            )
            if not success or not stats_message_id:
                self.logger.error("Failed to send initial Statistics message.")
                return False, {}, {}

            # 2. Send initial TOC message (placeholder)
            toc_placeholder_content = self._format_toc_placeholder(digest)
            toc_placeholder_content = self._ensure_message_length(toc_placeholder_content)
            messages_to_send.append((toc_placeholder_content, "Table of Contents"))
            
            # 3. Process news categories
            for category, news_items in digest.categories_news.items():
                if news_items:
                    emoji = emoji_map.get(category, "📰")
                    
                    # Create category header message
                    category_header = f"## {emoji} {category} ({len(news_items)} items)"
                    category_header = self._ensure_message_length(category_header)
                    messages_to_send.append((category_header, category))
                    
                    # Create individual item messages
                    for i, item in enumerate(news_items, 1):
                        item_message = self._format_single_item(item)
                        # Split long item messages if needed
                        split_messages = self._split_long_message(item_message, f"{category}_item_{i}")
                        messages_to_send.extend(split_messages)
            
            # 4. Process paper categories (paper-only categories)
            paper_only_categories = set()
            for category, paper_items in digest.categories_papers.items():
                if paper_items and category not in digest.categories_news:
                    paper_only_categories.add(category)
                    emoji = emoji_map.get(category, "📚") # Use paper emoji for paper-only categories
                    
                    # Create category header for paper-only category
                    category_header = f"## {emoji} {category} ({len(paper_items)} papers)"
                    category_header = self._ensure_message_length(category_header)
                    messages_to_send.append((category_header, category))
                    
                    # Create individual paper messages
                    for i, item in enumerate(paper_items, 1):
                        item_message = self._format_single_item(item)
                        # Split long item messages if needed
                        split_messages = self._split_long_message(item_message, f"{category}_paper_{i}")
                        messages_to_send.extend(split_messages)
            
            # 5. Process grouped paper sections (HuggingFace and arXiv)
            
            # Separate papers by source
            hf_papers = []
            arxiv_papers = []
            
            for category, paper_items in digest.categories_papers.items():
                for paper in paper_items:
                    if "HuggingFace" in paper.source_info:
                        print("HuggingFace paper found:", paper.title)
                        hf_papers.append(paper)
                    else:
                        arxiv_papers.append(paper)
            
            # Add HuggingFace papers section
            if hf_papers:
                hf_header = f"## 🏆 Papers of the Month (HuggingFace) ({len(hf_papers)} papers)"
                hf_header = self._ensure_message_length(hf_header)
                messages_to_send.append((hf_header, "HuggingFace Papers"))
                
                for i, paper in enumerate(hf_papers, 1):
                    item_message = self._format_single_item(paper)
                    # Split long item messages if needed
                    split_messages = self._split_long_message(item_message, f"HuggingFace_paper_{i}")
                    messages_to_send.extend(split_messages)
            
            # Add arXiv papers section
            if arxiv_papers:
                arxiv_header = f"## 📚 Recent Papers (arXiv) ({len(arxiv_papers)} papers)"
                arxiv_header = self._ensure_message_length(arxiv_header)
                messages_to_send.append((arxiv_header, "arXiv Papers"))
                
                for i, paper in enumerate(arxiv_papers, 1):
                    item_message = self._format_single_item(paper)
                    # Split long item messages if needed
                    split_messages = self._split_long_message(item_message, f"arXiv_paper_{i}")
                    messages_to_send.extend(split_messages)
            
            # Send all messages (excluding the initial Statistics and TOC)
            self.logger.info(f"Sending {len(messages_to_send)} messages to Discord...")
            all_urls_and_ids = self.enhanced_webhook.send_messages_batch(messages_to_send, delay=1.0)
            
            # Extract category and paper section URLs from the results
            category_urls = {}
            paper_section_urls = {}
            toc_message_id = None # Initialize toc_message_id

            for message_name, (url, msg_id) in all_urls_and_ids.items():
                if message_name == "Table of Contents":
                    toc_message_id = msg_id
                elif not any(suffix in message_name for suffix in ["_item_", "_paper_", "_part_"]):
                    if message_name in ["HuggingFace Papers", "arXiv Papers"]:
                        paper_section_urls[message_name] = url
                    else:
                        category_urls[message_name] = url
            
            # 6. Update the initial TOC message with actual URLs
            if toc_message_id:
                final_toc_content = self._format_final_toc(digest, category_urls, paper_section_urls)
                final_toc_content = self._ensure_message_length(final_toc_content)
                edit_success = self.enhanced_webhook.edit_message(toc_message_id, final_toc_content)

                if not edit_success:
                    self.logger.error(f"Failed to edit TOC message {toc_message_id}.")
            else:
                self.logger.error("TOC message ID not found, cannot edit TOC.")
            
            # Log results
            total_categories = len(category_urls)
            total_paper_sections = len(paper_section_urls)
            
            if total_categories > 0 or total_paper_sections > 0:
                self.logger.info(f"✅ Successfully sent {total_categories} categories and {total_paper_sections} paper sections")
                for category, url in category_urls.items():
                    self.logger.info(f"   Category - {category}: {url}")
                for section, url in paper_section_urls.items():
                    self.logger.info(f"   Papers - {section}: {url}")
                return True, category_urls, paper_section_urls
            else:
                self.logger.error("❌ Failed to send any category or paper section messages")
                return False, {}, {}
            
        except Exception as e:
            self.logger.error(f"Discord digest failed: {e}")
            return False, {}, {}

    def _ensure_message_length(self, content: str, max_length: int = 2000) -> str:
        """
        Ensure message content doesn't exceed Discord's character limit.
        
        Args:
            content: Message content to check
            max_length: Maximum allowed length (default 2000 for Discord)
            
        Returns:
            Truncated content if necessary
        """
        if len(content) <= max_length:
            return content
        
        # Smart truncation - try to truncate at word boundaries
        truncated = content[:max_length - 50]  # Leave space for truncation message
        
        # Find the last complete sentence or line break
        last_break = max(
            truncated.rfind('.'),
            truncated.rfind('\n'),
            truncated.rfind('!'),
            truncated.rfind('?'),
            truncated.rfind('---')
        )
        
        if last_break > len(truncated) * 0.7:  # If we found a good break point
            result = truncated[:last_break + 1] + "\n\n... *[Content truncated]*"
        else:
            # Fallback to word boundary
            last_space = truncated.rfind(' ')
            if last_space > len(truncated) * 0.8:
                result = truncated[:last_space] + "\n\n... *[Content truncated]*"
            else:
                result = truncated + "\n\n... *[Content truncated]*"
        
        self.logger.warning(f"Message truncated from {len(content)} to {len(result)} characters")
        return result

    def _split_long_message(self, content: str, base_name: str, max_length: int = 2000) -> List[Tuple[str, str]]:
        """
        Split a long message into multiple parts if it exceeds the character limit.
        
        Args:
            content: Message content to split
            base_name: Base name for the message parts
            max_length: Maximum allowed length per message
            
        Returns:
            List of (content, name) tuples for each message part
        """
        if len(content) <= max_length:
            return [(content, base_name)]
        
        parts = []
        remaining_content = content
        part_number = 1
        
        while remaining_content:
            if len(remaining_content) <= max_length:
                # Last part
                parts.append((remaining_content, f"{base_name}_part_{part_number}"))
                break
            
            # Find a good split point
            split_point = max_length - 50  # Leave space for continuation marker
            
            # Try to split at paragraph boundaries first
            paragraph_break = remaining_content[:split_point].rfind('\n\n')
            if paragraph_break > split_point * 0.5:
                split_point = paragraph_break
            else:
                # Try to split at sentence boundaries
                sentence_break = max(
                    remaining_content[:split_point].rfind('. '),
                    remaining_content[:split_point].rfind('.\n'),
                    remaining_content[:split_point].rfind('! '),
                    remaining_content[:split_point].rfind('? ')
                )
                if sentence_break > split_point * 0.7:
                    split_point = sentence_break + 1
                else:
                    # Fallback to word boundary
                    word_break = remaining_content[:split_point].rfind(' ')
                    if word_break > split_point * 0.8:
                        split_point = word_break
            
            # Create the part
            part_content = remaining_content[:split_point]
            parts.append((part_content, f"{base_name}_part_{part_number}"))
            
            # Update remaining content
            remaining_content = remaining_content[split_point:].lstrip()
            part_number += 1
        
        if len(parts) > 1:
            self.logger.info(f"Split long message '{base_name}' into {len(parts)} parts")
        
        return parts
    
    def _format_toc_placeholder(self, digest: DiscordDigest) -> str:
        """
        Format a placeholder Table of Contents message.
        """
        lines = [
            f"# 📰 Tech News & Research Digest - {digest.date}",
            "",
            "## Table of Contents (Updating...)",
            "",
            "Please wait while the digest content is being sent and links are being updated."
        ]
        return "\n".join(lines)

    def _format_final_toc(self, digest: DiscordDigest, category_urls: Dict[str, str], paper_section_urls: Dict[str, str]) -> str:
        """
        Format the final Table of Contents message with actual Discord URLs.
        """
        lines = [
            "## 📑 **Table of Contents**",
            "",
            "-" * 50,
            ""
        ]

        # Add news categories to TOC
        if digest.categories_news:
            lines.append("### 🗞️ **News Categories**:")
            for category in sorted(digest.categories_news.keys()):
                url = category_urls.get(category)
                if url:
                    lines.append(f"- [{category}]({url})")
                else:
                    lines.append(f"- {category} (Link N/A)")
            lines.append("")

        # Add paper-only categories to TOC
        paper_only_categories = [cat for cat in digest.categories_papers.keys() if cat not in digest.categories_news]
        if paper_only_categories:
            lines.append("### 🔬 **Paper Categories**:")
            for category in sorted(paper_only_categories):
                url = category_urls.get(category)
                if url:
                    lines.append(f"- [{category}]({url})")
                else:
                    lines.append(f"- {category} (Link N/A)")
            lines.append("")

        # Add grouped paper sections to TOC
        if paper_section_urls:
            lines.append("### 📚 **Research Papers**:")
            if "HuggingFace Papers" in paper_section_urls:
                lines.append(f"- [🏆 Papers of the Month (HuggingFace)]({paper_section_urls['HuggingFace Papers']})")
            if "arXiv Papers" in paper_section_urls:
                lines.append(f"- [📚 Recent Papers (arXiv)]({paper_section_urls['arXiv Papers']})")
            lines.append("")

        lines.append("---")
        lines.append("Click on the links above to jump to the relevant section in Discord.")
        lines.append("")
        lines.append("---")
        
        return "\n".join(lines)

    def _format_statistics(self, digest: DiscordDigest) -> str:
        """
        Format the statistics section for Discord.
        """
        lines = [
            f"# 📰 Tech News & Research Digest - {digest.date}",
            "## 📊 **Digest Statistics**",
            "",
            "-" * 50,
            ""
        ]
        
        total_news_categories = len(digest.statistics.get("categories_news_count", {}))
        total_paper_categories = len(digest.statistics.get("categories_papers_count", {}))

        lines.append(f"✨ **Total News Articles Sent:** `{digest.statistics.get('total_news', 0)}`")
        lines.append(f"📚 **Total Research Papers Sent:** `{digest.statistics.get('total_papers', 0)}`")
        lines.append(f"🏷️ **Unique News Categories:** `{total_news_categories}`")
        lines.append(f"🏷️ **Unique Paper Categories:** `{total_paper_categories}`")
        lines.append(f"📦 **Total Items Sent:** `{digest.statistics.get('total_items', 0)}`")
        lines.append("")
        lines.append("-" * 50)
        lines.append("`Report generated on:` **`" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "`**")
        lines.append("")
        lines.append("-" * 50)

        return "\n".join(lines)
    
    def _format_single_item(self, item: DiscordDigestItem) -> str:
        """
        Format a single news item or paper for Discord.
        
        Args:
            item: DiscordDigestItem to format
            index: Item number
            is_paper: Whether this is a research paper
            
        Returns:
            Formatted item string optimized for Discord
        """
        lines = []
        
        # Title with link - more concise format
        if item.url:
            lines.append(f"## **[{item.title}]({item.url})**")
        else:
            lines.append(f"## {item.title}")
        # Summary - full content since we're sending individual messages
        if item.summary:
            lines.append(f"### 📝 Summary:")
            lines.append(item.summary)
        
        # Key points - include all key points since we have space
        if item.key_points:
            lines.append("### 🔑 Key Points:")
            for point in item.key_points:  # Include all points
                lines.append(f"  • {point}")
        
        # Impact - include since we have space for individual messages
        if item.impact:
            lines.append(f"### 💥 Impact:")
            lines.append(item.impact)
        
        # Tags - include all tags
        if item.tags:
            tag_text = ", ".join(item.tags)  # Include all tags
            lines.append(f"*🏷️ Tags: {tag_text}*")
            
        lines.append("-" * 50)
        
        return "\n".join(lines) + "\n\n"

    def send_simple_notification(self, content: str, title: str) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Sends a simple notification to Discord.

        Args:
            content: The content of the notification.
            title: The title of the notification.

        Returns:
            A tuple containing a boolean indicating success, the Discord URL of the message, and the message ID.
        """
        if not self.config.get("discord", {}).get("enabled", False):
            self.logger.info("Discord notifications are disabled, skipping simple notification.")
            return False, None, None

        self.logger.info(f"Sending simple notification to Discord: {title}")
        
        # Use the enhanced webhook to send the message
        success, discord_url, message_id = self.enhanced_webhook.send_message_with_id(
            content=f"**{title}**\n\n{content}",
            category_name="Simple Notification"
        )
        
        if success:
            self.logger.info(f"Simple notification sent successfully. URL: {discord_url}")
        else:
            self.logger.error("Failed to send simple notification.")
            
        return success, discord_url, message_id
