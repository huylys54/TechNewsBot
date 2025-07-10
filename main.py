"""
Main orchestration script for the Tech News Digest Bot.
Coordinates fetching, classification, summarization, report generation, and notifications.
"""

import logging
import os
import sys
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import yaml
from pathlib import Path


from dotenv import load_dotenv
load_dotenv()


from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED

# Import our agents
from agents.fetcher import ContentFetcher, FetchResult
from agents.classifier import TechNewsClassifier
from agents.notifier import TechNewsNotifier, EnhancedDiscordDigestBot
from agents.notifier import TechNewsNotifier
from agents.summarizer import TechContentSummarizer
from agents.content_parser import ContentParser
from agents.report_generator import TechNewsReportGenerator
from agents.notifier import TechNewsNotifier, EnhancedDiscordDigestBot


class TechNewsDigestBot:
    """
    Main orchestrator for the Tech News Digest Bot.
    Coordinates all components to create and deliver tech news digests.
    """
    
    def __init__(self, config_path: str = "config/main.yml"):
        """
        Initialize the digest bot.
        
        Args:
            config_path: Path to main configuration file.
        """
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.config = self._load_config()
        
        # Initialize all components
        self.logger.info("Initializing Tech News Digest Bot components...")
        self.fetcher = ContentFetcher()
        self.classifier = TechNewsClassifier()
        self.summarizer = TechContentSummarizer()
        self.content_processor = ContentParser()
        self.report_generator = TechNewsReportGenerator()
        self.notifier = TechNewsNotifier()
        self.discord_digest_bot = EnhancedDiscordDigestBot()  # Initialize Discord digest bot
        
        self.logger.info("All components initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the bot."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # General logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # General file handler
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = log_dir / f"technews_bot_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        # Scheduler logger
        scheduler_logger = logging.getLogger('apscheduler')
        scheduler_logger.setLevel(logging.INFO)
        if not scheduler_logger.handlers:
            scheduler_log_file = log_dir / "scheduler.log"
            scheduler_file_handler = logging.FileHandler(scheduler_log_file)
            scheduler_file_handler.setLevel(logging.INFO)
            scheduler_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            scheduler_file_handler.setFormatter(scheduler_formatter)
            scheduler_logger.addHandler(scheduler_file_handler)
        
        return logger
    
    def _deep_merge(self, source, destination):
        """Deep merge two dictionaries."""
        for key, value in source.items():
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                self._deep_merge(value, node)
            else:
                destination[key] = value
        return destination

    def _load_config(self) -> Dict[str, Any]:
        """Load main configuration from file."""
        try:
            with open("config/defaults.yml", 'r', encoding='utf-8') as f:
                default_config = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load default config: {e}")
            return {}

        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                return self._deep_merge(user_config, default_config)
            else:
                return default_config
        except Exception as e:
            self.logger.warning(f"Failed to load user config from {self.config_path}: {e}")
            return default_config

    def generate_digest(self,
                       max_articles: Optional[int] = None,
                       max_papers: Optional[int] = None,
                       custom_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a complete tech news digest.
        
        Args:
            max_articles: Maximum number of articles to fetch.
            max_papers: Maximum number of papers to fetch.
            custom_keywords: Optional custom keywords for filtering.
            
        Returns:
            Dictionary with generation results and file paths.
        """
        self.logger.info("Starting digest generation process...")
        
        results = {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "files": {},
            "statistics": {},
            "errors": []
        }
        
        try:
            # Step 1: Fetch content
            self.logger.info("Step 1: Fetching content...")
            max_articles = max_articles or self.config["digest_settings"]["max_articles"]
            max_papers = max_papers or self.config["digest_settings"]["max_papers"]
            
            # Load arXiv categories from config
            arxiv_categories = None
            try:
                with open("config/feeds.yml", 'r', encoding='utf-8') as f:
                    feeds_config = yaml.safe_load(f)
                    arxiv_categories = feeds_config.get('arxiv_categories', [])
            except Exception as e:
                self.logger.warning(f"Could not load arXiv categories from config: {e}")
            
            # Determine if monthly papers should be excluded
            today = datetime.now()
            # Run on the day specified in config, default to 1
            monthly_send_day = self.config.get("monthly_papers", {}).get("send_day", 1)
            is_monthly_run_day = today.day == monthly_send_day

            fetch_result: FetchResult = self.fetcher.fetch_all_content(
                max_articles_per_feed=max_articles // 2 or 2,
                max_arxiv_papers=max_papers // 2 or 1,
                arxiv_categories=arxiv_categories,
                arxiv_keywords=custom_keywords,
                # exclude_monthly_papers=not is_monthly_run_day
                exclude_monthly_papers=False
            )
            
            article_count = len(fetch_result.news)
            paper_count = len(fetch_result.papers)
            
            self.logger.info(f"   Fetched {article_count} articles and {paper_count} papers")
            
            if fetch_result.is_empty():
                self.logger.warning("No content fetched - generating empty day notice.")
                empty_message = self.fetcher.generate_empty_day_message(fetch_result)
                
                # Send empty day notification via notifier
                if self.config["digest_settings"]["enable_notifications"]:
                    self.logger.info("Step 6: Sending empty day notification...")
                    # Use the Discord digest bot to send the empty message
                    discord_success, _, _ = self.discord_digest_bot.send_empty_digest_notice(empty_message)
                    if discord_success:
                        self.logger.info("   Empty day notice sent successfully to Discord.")
                    else:
                        self.logger.error("   Failed to send empty day notice to Discord.")
                
                results["success"] = True # Consider an empty day notice a successful run
                results["status"] = "empty_day"
                results["message"] = empty_message["message"]
                results["statistics"] = {
                    "articles_fetched": 0,
                    "papers_fetched": 0,
                    "items_classified": 0,
                    "items_summarized": 0,
                    "items_in_digest": 0,
                    "categories_found": 0,
                    "notifications_sent": 1 if self.config["digest_settings"]["enable_notifications"] else 0
                }
                return results
            
            # Save fetched content
            if self.config["output"]["save_individual_files"]:
                content_files = self.fetcher.save_content(fetch_result)
                results["files"].update(content_files)
            
            # Step 2: Process content
            processed_content = {}
            papers_as_dicts = []  # Initialize here to avoid unbound variable
            if self.config["digest_settings"]["enable_content_processing"]:
                self.logger.info("Step 2: Processing content...")
                
                # Convert arXiv papers to dict format before processing
                papers_as_dicts = [{
                    'title': paper.title,
                    'id': paper.get_short_id(),
                    'abstract': paper.summary,
                    'link': paper.entry_id,
                    'pdf_url': paper.pdf_url,
                    'source': "arXiv (HuggingFace Top)" if is_top else "arXiv",
                    'authors': ', '.join([author.name for author in paper.authors]),
                    'published': paper.published.isoformat(),
                    'category': paper.primary_category,
                    'fetch_time': datetime.now().isoformat(),
                    'local_markdown': f"data/papers/{paper.get_short_id()}.md",
                    'type': 'academic paper'
                } for paper, is_top in fetch_result.papers]
                
                # Process news articles
                processed_news = self.content_processor.extract_content(fetch_result.news)
                news_count = len(processed_news)
                self.logger.info(f"   Processed {news_count} articles")
                
                # Prepare processed content structure
                processed_content = {
                    'news': processed_news,  # Only news articles get content parsing
                    'arxiv': papers_as_dicts  # Papers are processed directly by summarizer using markdown files
                }
                
                # Save processed content if enabled
                if self.config["output"]["save_individual_files"]:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    prefix = f"processed_{timestamp}"
                    
                    # Save news content
                    news_filename = f"{prefix}_news.json"
                    news_path = self.content_processor.save_content_to_file(processed_news, news_filename)
                    processed_files = {'news_content_path': news_path}
                    
                    # Save papers content
                    arxiv_filename = f"{prefix}_arxiv.json"
                    os.makedirs("data/content", exist_ok=True)
                    arxiv_filepath = os.path.join("data/content", arxiv_filename)
                    
                    p = Path(arxiv_filepath)
                    if not p.parent.exists():
                        p.parent.mkdir(parents=True, exist_ok=True)
                    with open(arxiv_filepath, 'w', encoding='utf-8') as file:
                        json.dump(papers_as_dicts, file, indent=2, ensure_ascii=False)
                    processed_files['arxiv_content_path'] = arxiv_filepath
                    
                    # Save combined content
                    combined_filename = f"{prefix}_combined.json"
                    combined_filepath = os.path.join("data/content", combined_filename)
                    
                    p = Path(combined_filepath)
                    if not p.parent.exists():
                        p.parent.mkdir(parents=True, exist_ok=True)
                    with open(combined_filepath, 'w', encoding='utf-8') as file:
                        json.dump(processed_content, file, indent=2, ensure_ascii=False)
                    processed_files['combined_content_path'] = combined_filepath
                    results["files"].update(processed_files)
            
            # Step 3: Classify content
            classifications = []
            if self.config["digest_settings"]["enable_classification"]:
                self.logger.info("Step 3: Classifying content...")
                
                # Combine news and converted papers for classification
                all_content = fetch_result.news + papers_as_dicts
                classifications = self.classifier.classify_batch(all_content)
                
                if self.config["output"]["save_individual_files"]:
                    class_file = self.classifier.save_classifications(classifications, all_content)
                    results["files"]["classifications"] = class_file
                
                self.logger.info(f"   Classified {len(classifications)} items")
            
            # Step 4: Summarize content (news only, papers use abstracts)
            summaries = []
            if self.config["digest_settings"]["enable_summarization"]:
                self.logger.info("Step 4: Summarizing content...")
                
                # Summarize articles only
                article_summaries = []
                if fetch_result.news:
                    # Get processed content if available
                    content_texts = None
                    if processed_content.get('news'):
                        content_texts = [item.get('content', '') for item in processed_content['news']]
                    
                    article_summaries = self.summarizer.summarize_batch(
                        fetch_result.news,
                        content_texts,
                        content_type="news"
                    )
                
                # Summarize papers directly using their markdown files
                paper_summaries = []
                if fetch_result.papers:
                    self.logger.info(f"Summarizing {len(fetch_result.papers)} papers from markdown files")
                    paper_summaries = self.summarizer.summarize_batch(
                        [paper for paper, _ in fetch_result.papers],
                        content_type="papers",
                        data_dir="./data/papers/"
                    )
                
                summaries = article_summaries + paper_summaries
                
                if self.config["output"]["save_individual_files"]:
                    all_content = fetch_result.news + papers_as_dicts
                    summary_file = self.summarizer.save_summaries(summaries, all_content)
                    results["files"]["summaries"] = summary_file
                
                self.logger.info(f"   Summarized {len(article_summaries)} articles and {len(paper_summaries)} papers with enhanced analysis")
            
            # Step 5: Generate report
            self.logger.info("Step 5: Generating report...")
            
            # Prepare data for report generation
            content_data = {
                "articles": fetch_result.news,
                "papers": papers_as_dicts,
                "classifications": [],
                "summaries": []
            }
            
            # Add classifications if available
            if classifications:
                all_content = fetch_result.news + papers_as_dicts
                for item, classification in zip(all_content, classifications):
                    content_data["classifications"].append({
                        "content": item,
                        "classification": {
                            "category": classification.category,
                            "subcategory": classification.subcategory,
                            "confidence": classification.confidence,
                            "tags": classification.tags,
                            "reasoning": classification.reasoning
                        }
                    })
            
            # Add summaries if available
            if summaries:
                all_content = fetch_result.news + papers_as_dicts
                for item, summary in zip(all_content, summaries):
                    # Create summary dict with common fields
                    summary_dict = {
                        "title": summary.title,
                        "summary": summary.summary,
                        "key_points": summary.key_points,
                        "impact": summary.impact,
                        "technical_level": summary.technical_level,
                        "confidence": summary.confidence
                    }
                    
                    content_data["summaries"].append({
                        "original_content": item,
                        "summary": summary_dict
                    })
            
            # Generate the digest report
            report = self.report_generator.generate_digest_report(content_data)
            report_path = self.report_generator.save_report(report, format="markdown")
            results["files"]["report"] = report_path
            
            self.logger.info(f"   Generated report: {report_path}")
            
            # Step 6: Send non-Discord notifications (email, etc.)
            notification_results = {}
            if self.config["digest_settings"]["enable_notifications"]:
                self.logger.info("Step 6: Sending non-Discord notifications...")
                
                # Read the report content
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                
                # Send notifications but exclude Discord (handled separately in Step 7)
                # Temporarily disable Discord for old notification system
                original_discord_enabled = self.notifier.config.discord.get("enabled", False)
                self.notifier.config.discord["enabled"] = False
                
                notification_results = self.notifier.send_notifications(
                    report_content,
                    report.title,
                    report_path
                )
                
                # Restore Discord setting
                self.notifier.config.discord["enabled"] = original_discord_enabled
                
                self.logger.info(f"   Non-Discord notification results: {notification_results}")
            
            # Step 7: Send Discord digest and generate TOC-only report
            if self.config["digest_settings"]["enable_notifications"]:
                self.logger.info("Step 7: Sending Discord digest...")
                
                # Use the Discord digest bot to read from latest files and send
                discord_success, category_urls, paper_section_urls = self.discord_digest_bot.generate_and_send_digest()
                
                self.logger.info(f"   Discord digest success: {discord_success}")
                if category_urls:
                    self.logger.info(f"   News categories sent: {list(category_urls.keys())}")
                if paper_section_urls:
                    self.logger.info(f"   Paper sections sent: {list(paper_section_urls.keys())}")
                
                # Step 8: Generate TOC-only markdown report with Discord URLs
                if discord_success:
                    self.logger.info("Step 8: Generating TOC-only markdown report...")
                    
                    try:
                        # Generate a TOC-only report with both category and paper section URLs
                        toc_report = self.report_generator.generate_toc_only_report(report, category_urls, paper_section_urls, is_monthly_papers=False)

                        # Save TOC-only report
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        toc_report_path = f"data/reports/toc_digest_{timestamp}.md"
                        os.makedirs("data/reports", exist_ok=True)
                        
                        p = Path(toc_report_path)
                        if not p.parent.exists():
                            p.parent.mkdir(parents=True, exist_ok=True)
                        with open(toc_report_path, 'w', encoding='utf-8') as f:
                            f.write(toc_report)
                        
                        results["files"]["toc_report"] = toc_report_path
                        self.logger.info(f"   Generated TOC report: {toc_report_path}")
                        
                        if category_urls:
                            self.logger.info(f"   TOC includes Discord URLs for: {list(category_urls.keys())}")
                        else:
                            self.logger.info("   TOC generated without Discord URLs (no news categories found)")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to generate TOC report: {e}")
            
            # Compile statistics
            results["statistics"] = {
                "articles_fetched": article_count,
                "papers_fetched": paper_count,
                "items_classified": len(classifications),
                "items_summarized": len(summaries),
                "items_in_digest": report.statistics.get("total_in_digest", 0),
                "categories_found": report.statistics.get("categories_count", 0),
                "notifications_sent": len([k for k, v in notification_results.items() if v])
            }
            
            results["success"] = True
            self.logger.info("Digest generation completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Digest generation failed: {e}", exc_info=True)
            results["errors"].append(str(e))
        
        return results
    
    def run_scheduled_digest(self, max_articles: int, max_papers: int):
        """Run a scheduled digest generation."""
        self.logger.info("Starting scheduled digest generation job...")

        results = self.generate_digest(
            max_articles=max_articles,
            max_papers=max_papers
        )
        
        if results["success"]:
            self.logger.info(f"Scheduled digest job completed successfully. Stats: {results['statistics']}")
        else:
            self.logger.error(f"Scheduled digest job failed. Errors: {results['errors']}")
        
        return results

    def run_monthly_papers_digest(self, max_papers: int):
        """
        Run a scheduled monthly papers digest generation.
        This job now runs the full daily digest but ensures monthly papers are included.
        """
        self.logger.info("Starting first-of-month combined digest generation job...")

        # On the first day of the month, we run the full digest,
        # but ensure that the monthly papers are included.
        # The daily digest will automatically handle whether to include them or not.
        results = self.generate_digest(
            max_papers=max_papers,
            # We can add other parameters if needed, e.g., to force-run even if not the first day
        )

        if results["success"]:
            self.logger.info(f"Monthly combined digest job completed successfully. Stats: {results['statistics']}")
        else:
            self.logger.error(f"Monthly combined digest job failed. Errors: {results['errors']}")

        return results

    def scheduler_listener(self, event):
        """Listener for scheduler events."""
        if event.exception:
            self.logger.error(f"Job {event.job_id} crashed: {event.exception}")
        else:
            self.logger.info(f"Job {event.job_id} executed successfully")

    def _add_daily_digest_job(self, scheduler: BlockingScheduler, max_articles: int, max_papers: int):
        """Adds the daily digest job to the scheduler."""
        if self.config["scheduling"]["enabled"]:
            cron_expression = self.config["scheduling"]["cron_expression"]
            timezone = self.config["scheduling"]["timezone"]

            parts = cron_expression.split()
            if len(parts) != 5:
                self.logger.error(f"Invalid cron expression: {cron_expression}")
                return

            minute, hour, _, _, _ = parts
            day_of_week = "*" if self.config["scheduling"].get("include_weekends", False) else "MON-FRI"

            # The daily digest now handles the logic for monthly papers
            scheduler.add_job(
                lambda: self.run_scheduled_digest(max_articles, max_papers),
                CronTrigger(
                    minute=minute,
                    hour=hour,
                    day_of_week=day_of_week,
                    timezone=timezone
                ),
                id='digest_generation',
                name='Tech News Digest Generation'
            )
            self.logger.info(f"Daily digest scheduler started with cron expression: {cron_expression}")

    def _add_monthly_papers_job(self, scheduler: BlockingScheduler):
        """Adds the monthly papers job to the scheduler."""
        if self.config["monthly_papers"]["enabled"]:
            send_day = self.config["monthly_papers"]["send_day"]
            send_time = self.config["monthly_papers"]["send_time"]
            max_monthly_papers = self.config["monthly_papers"]["max_papers"]
            timezone = self.config["scheduling"]["timezone"]

            time_parts = send_time.split(":")
            if len(time_parts) != 2:
                self.logger.error(f"Invalid send time: {send_time}")
                return

            monthly_hour, monthly_minute = time_parts
            monthly_cron_expression = f"{monthly_minute} {monthly_hour} {send_day} * *"

            scheduler.add_job(
                lambda: self.run_monthly_papers_digest(max_monthly_papers),
                CronTrigger(
                    minute=monthly_minute,
                    hour=monthly_hour,
                    day=send_day,
                    month="*",
                    day_of_week="*",
                    timezone=timezone
                ),
                id='monthly_papers_generation',
                name='Tech News Monthly Papers Generation'
            )
            self.logger.info(f"Monthly papers scheduler started with cron expression: {monthly_cron_expression}")

    def start_scheduler(self, max_articles: int, max_papers: int):
        """Start the scheduled digest generation."""
        scheduler = BlockingScheduler()
        scheduler.add_listener(self.scheduler_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED)

        self._add_daily_digest_job(scheduler, max_articles, max_papers)
        self._add_monthly_papers_job(scheduler)

        if not scheduler.get_jobs():
            self.logger.warning("No jobs scheduled. Exiting.")
            return

        self.logger.info("Press Ctrl+C to stop the scheduler")
        try:
            scheduler.start()
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Scheduler error: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tech News Digest Bot')
    parser.add_argument('--generate', action='store_true', help='Generate a digest now')
    parser.add_argument('--schedule', action='store_true', help='Start scheduled digest generation')
    parser.add_argument('--max-articles', type=int, help='Maximum number of articles per rss to fetch', default=4)
    parser.add_argument('--max-papers', type=int, help='Maximum number of papers to fetch', default=5)
    parser.add_argument('--keywords', nargs='+', help='Custom keywords for filtering')
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize bot
    config_path = args.config or "config/main.yml"
    bot = TechNewsDigestBot(config_path)
    
    if args.generate:
        # Generate a digest now
        print("Generating tech news digest...")
        results = bot.generate_digest(
            max_articles=args.max_articles,
            max_papers=args.max_papers,
            custom_keywords=args.keywords
        )
        
        if results["success"]:
            print(f"✅ Digest generated successfully!")
            print(f"📊 Statistics: {results['statistics']}")
            print(f"📁 Files: {results['files']}")
        else:
            print(f"❌ Digest generation failed: {results['errors']}")
            sys.exit(1)
    
    elif args.schedule:
        # Start scheduler
        print("Starting scheduled digest generation...")
        bot.start_scheduler(
            max_articles=args.max_articles,
            max_papers=args.max_papers
        )
    
    else:
        # Show help
        parser.print_help()


if __name__ == "__main__":
    main()
