"""
Fetches top tech news headlines from RSS feeds and arXiv papers.
"""

import feedparser
import arxiv
import logging
import json
import yaml
import os
import requests
from requests.exceptions import ConnectionError
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time
from collections import deque
from datetime import timedelta, datetime, timezone
import random
import pathlib
import pymupdf
import pymupdf4llm


class FetchResult:
    """
    Represents the result of a content fetching operation.
    """
    def __init__(self, news: List[Dict], papers: List[Tuple[arxiv.Result, bool]], status: str = "success",
                 sources_checked: Optional[List[str]] = None, date_range: str = "",
                 next_check_time: Optional[datetime] = None):
        self.news = news
        self.papers = papers
        self.status = status  # "success", "empty", "partial", "error"
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.sources_checked = sources_checked if sources_checked is not None else []
        self.date_range = date_range
        # Ensure next_check_time is always a datetime object if not explicitly provided
        self.next_check_time = next_check_time if next_check_time is not None else datetime.now(timezone.utc) + timedelta(days=1)

    def is_empty(self) -> bool:
        return not self.news and not self.papers



class ArxivRateLimiter:
    """
    Rate limiter for arXiv API calls to respect their terms of use.
    
    arXiv rate limits:
    - No more than 1 request every 3 seconds on average
    - No more than 1 burst of 5 requests in a 5-second period
    """
    
    def __init__(self):
        self.request_times = deque()
        self.min_interval = 3.0  # Minimum 3 seconds between requests
        self.burst_limit = 5     # Maximum 5 requests in 5 seconds
        self.burst_window = 5.0  # 5-second window for burst detection
        
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits before making a request."""
        current_time = time.time()
        
        # Remove old requests outside the burst window
        while self.request_times and current_time - self.request_times[0] > self.burst_window:
            self.request_times.popleft()
        
        # Check burst limit
        if len(self.request_times) >= self.burst_limit:
            # We've hit the burst limit, wait until the oldest request is outside the window
            sleep_time = self.burst_window - (current_time - self.request_times[0]) + 0.1
            if sleep_time > 0:
                print(f"Rate limiting: waiting {sleep_time:.1f}s due to burst limit")
                time.sleep(sleep_time)
                current_time = time.time()
                # Clean up old requests again
                while self.request_times and current_time - self.request_times[0] > self.burst_window:
                    self.request_times.popleft()
        
        # Check minimum interval
        if self.request_times:
            time_since_last = current_time - self.request_times[-1]
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                print(f"Rate limiting: waiting {sleep_time:.1f}s for minimum interval")
                time.sleep(sleep_time)
                current_time = time.time()
        
        # Record this request
        self.request_times.append(current_time)


class NewsFetcher:
    """
    Fetches tech news from RSS feeds and news APIs.
    
    This class handles fetching news headlines from various tech news sources
    defined in the feeds configuration file.
    """
    
    def __init__(self, feeds_config_path: str = "config/feeds.yml"):
        """
        Initialize the NewsFetcher with a configuration file.
        
        Args:
            feeds_config_path: Path to the YAML configuration file containing RSS feed URLs.
        """
        self.feeds_config_path = feeds_config_path
        self.logger = self._setup_logging()
        self.feeds = self._load_feeds()
        self.keywords = self._load_keywords()
    
    def _load_feeds(self) -> List[Dict[str, str]]:
        """
        Load RSS feed URLs from configuration file.
        
        Returns:
            List of feed configurations with name and URL.
        """
        try:
            with open(self.feeds_config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config.get('feeds', self._get_default_feeds())
        except FileNotFoundError:
            self.logger.warning(f"Feeds config file not found: {self.feeds_config_path}")
            return self._get_default_feeds()
        except Exception as e:
            self.logger.error(f"Error loading feeds config: {e}")
            return self._get_default_feeds()
        
    def _get_default_feeds(self) -> List[Dict[str, str]]:
        """
        Provide default tech news RSS feeds if config file is not available.
        
        Returns:
            List of default feed configurations.
        """
        return [
            {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
            {"name": "Ars Technica", "url": "http://feeds.arstechnica.com/arstechnica/index"},
            {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
            {"name": "Wired", "url": "https://www.wired.com/feed/rss"}
        ]
        
    def _load_keywords(self) -> List[str]:
        """
        Load keywords from configuration file for filtering news articles.
        
        Returns:
            List of keywords for filtering articles.
        """
        try:
            with open(self.feeds_config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config.get('keywords', [])
        except (FileNotFoundError, Exception) as e:
            self.logger.warning(f"Could not load keywords from config: {e}")
            return []
    
    

    def _deduplicate_across_feeds(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Deduplicate articles across all feeds and apply keyword filtering.
        
        Args:
            articles: List of all articles from all feeds.
            
        Returns:
            Deduplicated and filtered list of articles.
        """
        if not self.keywords:
            # If no keywords, just remove duplicates without filtering
            seen_content = set()
            deduplicated = []
            
            for article in articles:
                title = article.get('title', '').lower().strip()
                description = article.get('description', '').lower().strip()
                clean_desc = BeautifulSoup(description, 'html.parser').get_text()[:100]
                content_key = (title, clean_desc)
                
                if content_key not in seen_content:
                    deduplicated.append(article)
                    seen_content.add(content_key)
            
            return deduplicated
        
        # Apply keyword filtering with deduplication
        filtered_articles = []
        seen_content = set()
        
        for article in articles:
            # Create content hash for duplicate detection
            title = article.get('title', '').lower().strip()
            description = article.get('description', '').lower().strip()
            
            
            clean_desc = BeautifulSoup(description, 'html.parser').get_text()[:500]
            content_key = (title, clean_desc)
            
            if content_key in seen_content:
                self.logger.debug(f"Skipping duplicate article across feeds: {title}")
                continue
            
            # Combine title and description for keyword matching
            combined_text = f"{article.get('title', '')} {article.get('description', '')}"
            clean_text = BeautifulSoup(combined_text, 'html.parser').get_text().lower()
            
            # Keyword matching
            matched_keywords = []
            keyword_score = 0
            
            for keyword in self.keywords:
                keyword_lower = keyword.lower()
                
                if len(keyword_lower.split()) > 1:
                    # Multi-word keywords: exact phrase matching
                    if keyword_lower in clean_text:
                        matched_keywords.append(keyword)
                        keyword_score += 2
                else:
                    # Single-word keywords: word boundary matching
                    import re
                    pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                    if re.search(pattern, clean_text):
                        matched_keywords.append(keyword)
                        keyword_score += 1
            
            # Require at least 1 keyword match
            if keyword_score >= 1:
                article['matched_keywords'] = ', '.join(matched_keywords[:5])
                article['keyword_score'] = str(keyword_score)
                filtered_articles.append(article)
                seen_content.add(content_key)
                self.logger.debug(f"Article matched keywords: {title} (score: {keyword_score})")
            else:
                self.logger.debug(f"Article filtered out (no keyword matches): {title}")
        
        # Sort by keyword score
        filtered_articles.sort(key=lambda x: int(x.get('keyword_score', '0')), reverse=True)
        
        print(f"Articles: {[article['title'] for article in articles]}")
        self.logger.info(f"Cross-feed deduplication: {len(articles)} -> {len(filtered_articles)} articles")
        return filtered_articles

    def fetch_headlines(self, max_articles_per_feed: int = 5) -> List[Dict[str, str]]:
        """
        Fetch headlines from all configured RSS feeds with cross-feed deduplication.
        
        Args:
            max_articles_per_feed: Maximum number of articles to fetch per feed.
            
        Returns:
            List of news articles with title, description, link, and source.
        """
        all_articles = []
        
        for feed in self.feeds:
            try:
                articles = self._fetch_from_feed(feed, max_articles_per_feed)
                all_articles.extend(articles)
                self.logger.info(f"Fetched {len(articles)} articles from {feed['name']}")
            except Exception as e:
                self.logger.error(f"Error fetching from {feed['name']}: {e}")
          # Apply cross-feed deduplication and keyword filtering
        all_articles = self._deduplicate_across_feeds(all_articles)
        
        self.logger.info(f"Total articles fetched after deduplication: {len(all_articles)}")
        return all_articles

    def _fetch_from_feed(self, feed: Dict[str, str], max_articles: int) -> List[Dict[str, str]]:
        """
        Fetch articles from a single RSS feed without filtering (done globally).
        
        Args:
            feed: Feed configuration with name and URL.
            max_articles: Maximum number of articles to fetch.
            
        Returns:
            List of articles from this feed.
        """
        parsed_feed = feedparser.parse(feed['url'])
        articles = []
        
        for entry in parsed_feed.entries[:max_articles]:
            article = {
                'title': entry.get('title', 'No Title'),
                'description': entry.get('summary', entry.get('description', 'No Description')),
                'link': entry.get('link', ''),
                'source': feed['name'],
                'published': entry.get('published', ''),
                'fetch_time': datetime.now().isoformat(),
                'type': 'news_article'
            }
            articles.append(article)
        
        return articles
    
    def save_articles_to_file(self, articles: List[Dict[str, str]], filename: Optional[str] = None) -> str:
        """
        Save fetched articles to a JSON file.
        
        Args:
            articles: Articles to save.
            filename: Custom filename. If None, uses timestamp.
            
        Returns:
            Path to the saved file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"articles_{timestamp}.json"
        
        # Ensure data directory exists
        os.makedirs("data/articles", exist_ok=True)
        filepath = os.path.join("data/articles", filename)
        
        p = pathlib.Path(filepath)
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(articles, file, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(articles)} articles to {filepath}")
        return filepath
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the news fetcher."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger


class ArxivFetcher:
    """
    Fetches research papers from arXiv using various search criteria.
    
    This class handles fetching recent papers from arXiv based on categories,
    keywords, and date ranges.
    """
    def __init__(self, feeds_config_path: str = "config/feeds.yml", dir_path: str = "data/papers"):
        """
        Initialize the ArxivFetcher with default settings.
        
        Args:
            feeds_config_path: Path to the YAML configuration file.
        """
        
        self.dir_path = dir_path
        self.feeds_config_path = feeds_config_path
        self.client = arxiv.Client(
            page_size=50,
            delay_seconds=0,  # We'll handle rate limiting manually
            num_retries=3,
        )
        self.logger = self._setup_logging()
        self.categories = self._load_categories()
        self.rate_limiter = ArxivRateLimiter()
    
    def _load_categories(self) -> List[str]:
        """
        Load arXiv categories from configuration file.
        
        Returns:
            List of arXiv categories.
        """
        try:
            with open(self.feeds_config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config.get('arxiv_categories', self._get_default_categories())
        except (FileNotFoundError, Exception) as e:
            self.logger.warning(f"Could not load categories from config: {e}")
            return self._get_default_categories()
    
    def fetch_papers(self,
                    categories: Optional[List[str]] = None,
                    max_papers: int = 10,
                    days_back: int = 3) -> List[arxiv.Result]:
        """
        Fetch recent arXiv papers from specified categories.
        
        Args:
            categories: List of arXiv categories (e.g., ['cs.AI', 'cs.LG', 'cs.CL']).
            max_papers: Maximum number of papers to fetch.
            days_back: Number of days to look back for papers.
            
        Returns:
            List of arXiv papers with metadata.
        """
        if categories is None:
            categories = self.categories
        
        all_papers = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for category in categories:
            try:
                self.logger.info(f"Fetching papers from {category} (respecting arXiv rate limits)")
                self.rate_limiter.wait_if_needed()  # Respect arXiv rate limits
                papers = self._fetch_by_category(category, max_papers, cutoff_date)
                all_papers.extend(papers)
                self.logger.info(f"Fetched {len(papers)} papers from {category}")
            except Exception as e:
                self.logger.error(f"Error fetching arXiv papers from {category}: {e}")
        
        # Sort by submission date (most recent first)
        all_papers.sort(key=lambda x: x.published, reverse=True)
        # Randomly select papers from different categories and limit to max_papers
        
        # Group papers by category
        papers_by_category = {}
        for paper in all_papers:
            main_category = paper.primary_category
            if main_category not in papers_by_category:
                papers_by_category[main_category] = []
                papers_by_category[main_category].append(paper)
        
        # Randomly select papers from each category
        selected_papers = []
        categories_list = list(papers_by_category.keys())
        random.shuffle(categories_list)  # Randomize category order
        
        papers_per_category = max(1, max_papers // len(categories_list)) if categories_list else 0
        
        for category in categories_list:
            category_papers = papers_by_category[category]
            random.shuffle(category_papers)  # Randomize papers within category
            
            # Take up to papers_per_category from this category
            take_count = min(papers_per_category, len(category_papers), max_papers - len(selected_papers))
            selected_papers.extend(category_papers[:take_count])
            
            if len(selected_papers) >= max_papers:
                break
        
        # If we still have room, fill with remaining papers randomly
        if len(selected_papers) < max_papers:
            remaining_papers = [p for p in all_papers if p not in selected_papers]
            random.shuffle(remaining_papers)
            needed = max_papers - len(selected_papers)
            selected_papers.extend(remaining_papers[:needed])
        
        all_papers = selected_papers[:max_papers]
        
        for paper in all_papers:
            self.save_paper_to_markdown(paper)
            
        self.logger.info(f"Total arXiv papers fetched: {len(all_papers)}")
        return all_papers
    
    def fetch_huggingface_top_papers(self, max_papers: int = 5) -> List[arxiv.Result]:
        """
        Fetch top papers of the month from Hugging Face.
        
        Args:
            max_papers: Maximum number of top papers to fetch.
            
        Returns:
            List of top papers with metadata from arXiv.
        """
        try:
            # Build current month URL
            now = datetime.now()
            default_url_template = "https://huggingface.co/papers/month/{year}-{month:02d}"
            hf_url = default_url_template.format(year=now.year, month=now.month)
            
            # Load HuggingFace URL template from config
            try:
                with open(self.feeds_config_path, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                    hf_config = config.get('huggingface_papers', {})
                    if hf_config.get('enabled', True):
                        url_template = hf_config.get('url_template', default_url_template)
                        hf_url = url_template.format(year=now.year, month=now.month)
                        max_papers = hf_config.get('max_papers', max_papers)
                    else:
                        self.logger.info("HuggingFace top papers disabled in config")
                        return []
            except Exception as e:
                self.logger.warning(f"Could not load HuggingFace config, using defaults: {e}")
            
            self.logger.info(f"Fetching top papers from HuggingFace for {now.year}-{now.month:02d}: {hf_url}")
            
            # Scrape the page to get paper IDs
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                "Accept-Encoding": "*",
                "Connection": "keep-alive"
            }
            response = requests.get(hf_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')            
            # Find paper links with the pattern /papers/{paper_id}
            paper_links = soup.find_all('a', href=re.compile(r'^/papers/[\d.]+v?\d*$'))
            
            # Extract paper IDs from the links, ensuring uniqueness
            paper_ids_set = set()
            paper_ids = []
            
            for link in paper_links:
                href = link.get('href')
                # Extract paper ID from /papers/{paper_id}
                paper_id = href.split('/papers/')[-1]
                
                # Only add if we haven't seen this ID before
                if paper_id not in paper_ids_set and len(paper_ids) < max_papers:
                    paper_ids_set.add(paper_id)
                    paper_ids.append(paper_id)
                
            self.logger.info(f"Found {len(paper_ids)} unique top paper IDs from HuggingFace: {paper_ids}")
            
            if not paper_ids:
                self.logger.warning("No paper IDs found on HuggingFace page")
                return []
              # Fetch papers from arXiv using the IDs
            return self._fetch_papers_by_ids(paper_ids)
            
        except Exception as e:
            self.logger.error(f"Error fetching HuggingFace top papers: {e}")
            return []

    def _fetch_papers_by_ids(self, paper_ids: List[str]) -> List[arxiv.Result]:
        """
        Fetch arXiv papers by their IDs with enhanced error handling.
        
        Args:
            paper_ids: List of arXiv paper IDs.
            
        Returns:
            List of papers with metadata.
        """
        if not paper_ids:
            return []
            
        
        
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                # Use rate limiter for this request too
                self.rate_limiter.wait_if_needed()
                
                self.logger.info(f"Attempting to fetch {len(paper_ids)} papers from arXiv (attempt {attempt + 1}/{max_retries})")
                
                # Use our existing client instead of creating a new one
                results = self.client.results(
                    arxiv.Search(
                        query="",
                        id_list=paper_ids,
                        max_results=len(paper_ids),
                        sort_by=arxiv.SortCriterion.Relevance
                    )
                )
               
                papers = []
                for result in results:
                    self.save_paper_to_markdown(result)
                    papers.append(result)
                self.logger.info(f"Successfully fetched {len(papers)} papers from arXiv using HuggingFace IDs")
                return papers
                
            except (ConnectionError, ConnectionResetError, OSError) as e:
                self.logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.error(f"Failed to fetch papers after {max_retries} attempts due to connection issues")
                    return []
                    
            except Exception as e:
                self.logger.error(f"Unexpected error fetching papers by IDs from arXiv: {e}")
                # For other errors, don't retry
                return []
        
        # If we get here, all retries failed
        return []

    def fetch_by_keywords(self,
                         keywords: List[str],
                         max_papers: int = 10,
                         days_back: int = 30) -> List[arxiv.Result]:
        """
        Fetch arXiv papers by keyword search.
        
        Args:
            keywords: List of keywords to search for.
            max_papers: Maximum number of papers to fetch.
            days_back: Number of days to look back for papers.
            
        Returns:
            List of arXiv papers matching the keywords.
        """
        query = " OR ".join([f'all:"{keyword}"' for keyword in keywords])
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        search = arxiv.Search(
            query=query,
            max_results=max_papers * 2,  # Fetch more to filter by date
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        try:
            for result in self.client.results(search):
                # Check if paper is recent enough
                if result.published.replace(tzinfo=None) < cutoff_date:
                    continue
                
                papers.append(result)
                
                if len(papers) >= max_papers:
                    break
            
            self.logger.info(f"Fetched {len(papers)} papers for keywords: {keywords}")
        except Exception as e:
            self.logger.error(f"Error fetching papers by keywords {keywords}: {e}")
        
        return papers
    
    def _fetch_by_category(self, 
                          category: str, 
                          max_papers: int, 
                          cutoff_date: datetime) -> List[arxiv.Result]:
        """
        Fetch arXiv papers from a specific category.
        
        Args:
            category: arXiv category (e.g., 'cs.AI').
            max_papers: Maximum number of papers to fetch.
            cutoff_date: Only fetch papers newer than this date.
            
        Returns:
            List of papers from this category.
        """
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_papers * 2,  # Fetch more to filter by date
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in self.client.results(search):
            # Check if paper is recent enough
            if result.published.replace(tzinfo=None) < cutoff_date:
                continue
                
            papers.append(result)
            
            if len(papers) >= max_papers:
                break
        
        return papers

    
    def _get_default_categories(self) -> List[str]:
        """
        Get default arXiv categories for tech and AI content.
        
        Returns:
            List of default categories.
        """
        return ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.CR', 'cs.DC']

    def save_papers_to_file(self, papers: List[Tuple[arxiv.Result, bool]], filename: Optional[str] = None) -> str:
        """
        Save fetched papers to a JSON file.
        
        Args:
            papers: Papers to save.
            filename: Custom filename. If None, uses timestamp.
            
        Returns:
            Path to the saved file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arxiv_papers_{timestamp}.json"
        
        # Ensure data directory exists
        os.makedirs("data/papers", exist_ok=True)
        filepath = os.path.join("data/papers", filename)
        results = []

        for paper, is_top in papers:
            result = {
                "title": paper.title,
                "id": paper.get_short_id(),
                "abstract": paper.summary,
                "link": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "authors": ', '.join([author.name for author in paper.authors]),
                "published": paper.published.isoformat(),
                "category": paper.primary_category,
                "fetch_time": datetime.now().isoformat(),
                "local_markdown": f"{self.dir_path}/{paper.get_short_id()}.md",
                "type": "academic paper",
            }
            result["source"] = "arXiv (HuggingFace Top)" if is_top else "arXiv"

            results.append(result)
        p = pathlib.Path(filepath)
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(results)} papers to {filepath}")
        return filepath
    
    def save_paper_to_markdown(self, paper: arxiv.Result) -> str:
        """
        Save a single arXiv paper to a Markdown file.
        
        Args:
            paper: arXiv paper result object.
            filename: Custom filename. If None, uses paper ID.
            
        Returns:
            Path to the saved Markdown file.
        """
        filename = f"{paper.get_short_id()}.pdf"
        paper.download_pdf(filename=filename)  # Download PDFs to local storage
        doc = pymupdf.open(filename)
        num_pages = int(doc.page_count * 0.8)  # First 80% of pages
        md_text = pymupdf4llm.to_markdown(filename, pages=range(num_pages))
        doc.close()  # Close the document to release the file
        md_path = pathlib.Path(self.dir_path, filename.replace('.pdf', '.md'))
        if not md_path.parent.exists():
            md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_bytes(md_text.encode())
        os.remove(filename)  # Remove PDF after conversion
        return str(md_path)

    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the arXiv fetcher."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger


class ContentFetcher:
    """
    Unified content fetcher that combines news articles and research papers.
    
    This class orchestrates fetching from multiple sources including RSS feeds
    and arXiv to provide comprehensive tech content.
    """
    
    def __init__(self, feeds_config_path: str = "config/feeds.yml"):
        """
        Initialize the ContentFetcher with both news and arXiv fetchers.
        
        Args:
            feeds_config_path: Path to the YAML configuration file.
        """
        self.feeds_config_path = feeds_config_path
        self.news_fetcher = NewsFetcher(feeds_config_path)
        self.arxiv_fetcher = ArxivFetcher(feeds_config_path)
        self.logger = self._setup_logging()
        self.feeds_config = self._load_feeds_config() # Load full config for sources and schedule

    def _load_feeds_config(self) -> Dict:
        """Load the entire feeds configuration file."""
        try:
            with open(self.feeds_config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Feeds config file not found: {self.feeds_config_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading feeds config for ContentFetcher: {e}")
            return {}

    def _get_checked_sources(self) -> List[str]:
        """Get a list of sources that were checked during fetching."""
        sources = []
        if self.feeds_config.get('feeds'):
            sources.extend([feed['name'] for feed in self.feeds_config['feeds']])
        if self.feeds_config.get('arxiv_categories') or self.feeds_config.get('huggingface_papers', {}).get('enabled', True):
            sources.append("arXiv")
            if self.feeds_config.get('huggingface_papers', {}).get('enabled', True):
                sources.append("HuggingFace Papers")
        return sorted(list(set(sources))) # Return unique and sorted sources

    def _get_date_range(self, days_back: int = 7) -> str:
        """Get the date range for which content was fetched."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"


    def fetch_all_content(self,
                         max_articles_per_feed: int = 5,
                         max_arxiv_papers: int = 10,
                         arxiv_categories: Optional[List[str]] = None,
                         arxiv_keywords: Optional[List[str]] = None,
                         arxiv_days_back: int = 7,
                         exclude_monthly_papers: bool = False) -> FetchResult:
        """
        Fetch both news articles and arXiv papers (recent + top papers).

        Args:
            max_articles_per_feed: Maximum articles per RSS feed.
            max_arxiv_papers: Maximum arXiv papers to fetch (for recent papers).
            arxiv_categories: arXiv categories to search.
            arxiv_keywords: Keywords to search for in arXiv.
            arxiv_days_back: Days to look back for papers.
            exclude_monthly_papers: Exclude HuggingFace top papers from the fetch.

        Returns:
            FetchResult object containing news, arxiv papers, and status.
        """
        self.logger.info("Starting comprehensive content fetch...")

        news_articles = self.news_fetcher.fetch_headlines(max_articles_per_feed)

        # Load paper settings from config
        max_recent_per_category = 3
        max_top_papers = 5
        try:
            paper_settings = self.feeds_config.get('paper_settings', {})
            max_recent_per_category = paper_settings.get('max_recent_papers_per_category', 3)
            max_top_papers = paper_settings.get('max_top_papers', 5)
        except Exception as e:
            self.logger.warning(f"Could not load paper settings from config: {e}")

        all_papers: List[Tuple[arxiv.Result, bool]] = []
        # 1. Fetch top papers from HuggingFace, unless excluded
        if not exclude_monthly_papers:
            self.logger.info("Fetching top papers from HuggingFace...")
            top_papers = self.arxiv_fetcher.fetch_huggingface_top_papers(max_papers=max_top_papers)
            if top_papers:
                all_papers.extend((paper, True) for paper in top_papers)
                self.logger.info(f"Added {len(top_papers)} top papers from HuggingFace")

        # 2. Fetch recent papers by categories
        if arxiv_categories:
            self.logger.info("Fetching recent papers by categories...")
            category_papers = self.arxiv_fetcher.fetch_papers(
                categories=arxiv_categories,
                max_papers=max_recent_per_category,  # Per category limit
                days_back=arxiv_days_back
            )
            all_papers.extend((paper, False) for paper in category_papers)
            self.logger.info(f"Added {len(category_papers)} recent papers from categories")

        # 3. Fetch recent papers by keywords if specified
        if arxiv_keywords:
            keyword_papers = self.arxiv_fetcher.fetch_by_keywords(
                keywords=arxiv_keywords,
                max_papers=max_arxiv_papers // 2,
                days_back=arxiv_days_back
            )
            all_papers.extend((paper, False) for paper in keyword_papers)
            self.logger.info(f"Added {len(keyword_papers)} recent papers from keywords")

        # 4. If no categories or keywords specified, use default categories
        if not arxiv_categories and not arxiv_keywords:
            default_papers = self.arxiv_fetcher.fetch_papers(
                max_papers=max_recent_per_category,
                days_back=arxiv_days_back
            )
            all_papers.extend((paper, False) for paper in default_papers)
            self.logger.info(f"Added {len(default_papers)} papers from default categories")

        # Remove duplicates from arXiv papers (by URL)
        seen_urls = set()
        unique_arxiv_papers = []
        for paper in all_papers:
            if paper[0].entry_id not in seen_urls:
                seen_urls.add(paper[0].entry_id)
                unique_arxiv_papers.append(paper)

        # Sort by published date (most recent first for recent papers)
        unique_arxiv_papers.sort(key=lambda x: x[0].published, reverse=True)

        fetch_result = FetchResult(
            news=news_articles,
            papers=unique_arxiv_papers,
            sources_checked=self._get_checked_sources(),
            date_range=self._get_date_range(arxiv_days_back),
            next_check_time=datetime.now(timezone.utc) + timedelta(days=1) # Assuming daily run
        )

        if fetch_result.is_empty():
            fetch_result.status = "empty"
            self.logger.info("No new content found. Setting fetch result status to 'empty'.")
        else:
            fetch_result.status = "success"
            self.logger.info(f"Fetch complete: {len(news_articles)} news articles, {len(unique_arxiv_papers)} arXiv papers")

        return fetch_result

    def generate_empty_day_message(self, fetch_result: FetchResult) -> Dict:
        """
        Generates a standardized message for days with no new content.
        """
        if not fetch_result.is_empty():
            self.logger.warning("generate_empty_day_message called but content is not empty.")
            return {}

        return {
            "type": "empty_day_notice",
            "title": "No New Tech Content Found Today",
            "message": (
                f"It looks like there's no new tech news or research papers for the period "
                f"**{fetch_result.date_range}**.\n\n"
                f"Sources checked: {', '.join(fetch_result.sources_checked)}\n"
                f"The next content check is scheduled for **{fetch_result.next_check_time.strftime('%Y-%m-%d %H:%M UTC')}**."
            ),
            "metadata": {
                "status": fetch_result.status,
                "timestamp": fetch_result.timestamp,
                "sources": fetch_result.sources_checked,
                "date_range": fetch_result.date_range,
                "next_check_time": fetch_result.next_check_time.isoformat()
            }
        }




    def save_content(self, content: FetchResult, filename: Optional[str] = None) -> Dict[str, str]:
        """
        Save content using each class's specific save method.

        Args:
            content: FetchResult object with news and arxiv papers.
            filename: Custom filename prefix. If None, uses timestamp.

        Returns:
            Dictionary with 'news_path' and 'arxiv_path' keys containing file paths.
        """
        if content.is_empty():
            self.logger.info("No content to save for an empty fetch result.")
            return {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result_paths = {}

        # Save news articles
        if content.news:
            news_filename = f"articles_{timestamp}.json" if filename is None else f"{filename}_articles.json"
            news_path = self.news_fetcher.save_articles_to_file(content.news, news_filename)
            result_paths['news_path'] = news_path
        # Save arXiv papers
        if content.papers:
            arxiv_filename = f"arxiv_papers_{timestamp}.json" if filename is None else f"{filename}_papers.json"
            arxiv_path = self.arxiv_fetcher.save_papers_to_file(content.papers, arxiv_filename)
            result_paths['arxiv_path'] = arxiv_path

        total_items = len(content.news) + len(content.papers)
        self.logger.info(f"Saved {total_items} total items to separate files: {result_paths}")

        return result_paths

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the content fetcher."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger
