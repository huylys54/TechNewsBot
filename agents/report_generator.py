"""
Report generation system for creating formatted tech news digests.
Generates markdown and HTML reports from classified and summarized content.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import json
import yaml
from pathlib import Path


class DigestReport:
    """Data class for digest report structure."""
    
    def __init__(self):
        self.title: str = ""
        self.date: str = ""
        self.categories: Dict[str, List[Dict]] = {}
        self.statistics: Dict[str, Any] = {}
        self.summary: str = ""


class TechNewsReportGenerator:
    """
    Generate formatted reports from classified and summarized tech news content.
    Supports markdown and HTML output formats.
    """
    
    def __init__(self, config_path: str = "config/reports.yml"):
        """
        Initialize the report generator.
        
        Args:
            config_path: Path to report configuration file.
        """
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.config = self._load_config()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the report generator."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self) -> Dict[str, Any]:
        """Load report configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            self.logger.warning(f"Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default report configuration."""
        return {            "report_settings": {
                "title_format": "Tech News Digest - {date}",
                "include_statistics": False,  # Disabled for TOC-only reports
                "include_summary": False,     # Disabled for TOC-only reports
                "include_content": False,     # Disabled for TOC-only reports
                "max_items_per_category": 10,
                "sort_by_confidence": True,
                "minimum_confidence": 0.1
            },
            "format_settings": {
                "markdown": {
                    "include_toc": True,
                    "include_metadata": True,
                    "code_highlighting": True
                },
                "html": {
                    "include_css": True,
                    "responsive_design": True,
                    "dark_mode_support": True
                }
            },
            "category_order": [
                "Artificial Intelligence",
                "Software Development", 
                "Hardware",
                "Cybersecurity",
                "Cloud Computing",
                "Startups & Business",
                "Social Media & Platforms",
                "Gaming",
                "Blockchain & Crypto",
                "Electric Vehicles & Transportation",
                "Other"
            ]
        }
    
    def load_content_data(self, 
                         articles_file: Optional[str] = None,
                         papers_file: Optional[str] = None,
                         classifications_file: Optional[str] = None,
                         summaries_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load content data from various sources.
        
        Args:
            articles_file: Path to articles JSON file.
            papers_file: Path to papers JSON file.
            classifications_file: Path to classifications JSON file.
            summaries_file: Path to summaries JSON file.
            
        Returns:
            Dictionary containing loaded data.
        """
        data = {
            "articles": [],
            "papers": [],
            "classifications": [],
            "summaries": []
        }
        
        # Load articles
        if articles_file and os.path.exists(articles_file):
            try:
                with open(articles_file, 'r', encoding='utf-8') as f:
                    data["articles"] = json.load(f)
                self.logger.info(f"Loaded {len(data['articles'])} articles from {articles_file}")
            except Exception as e:
                self.logger.error(f"Failed to load articles: {e}")
        
        # Load papers
        if papers_file and os.path.exists(papers_file):
            try:
                with open(papers_file, 'r', encoding='utf-8') as f:
                    data["papers"] = json.load(f)
                self.logger.info(f"Loaded {len(data['papers'])} papers from {papers_file}")
            except Exception as e:
                self.logger.error(f"Failed to load papers: {e}")
        
        # Load classifications
        if classifications_file and os.path.exists(classifications_file):
            try:
                with open(classifications_file, 'r', encoding='utf-8') as f:
                    data["classifications"] = json.load(f)
                self.logger.info(f"Loaded {len(data['classifications'])} classifications from {classifications_file}")
            except Exception as e:
                self.logger.error(f"Failed to load classifications: {e}")
        
        # Load summaries
        if summaries_file and os.path.exists(summaries_file):
            try:
                with open(summaries_file, 'r', encoding='utf-8') as f:
                    data["summaries"] = json.load(f)
                self.logger.info(f"Loaded {len(data['summaries'])} summaries from {summaries_file}")
            except Exception as e:
                self.logger.error(f"Failed to load summaries: {e}")
        
        return data
    
    def find_latest_files(self) -> Dict[str, str]:
        """
        Find the latest data files in the data directory.
        
        Returns:
            Dictionary with paths to latest files.
        """
        data_dir = Path("data")
        latest_files = {}
          # Find latest articles file
        articles_dir = data_dir / "articles"
        if articles_dir.exists():
            articles_files = list(articles_dir.glob("articles_*.json"))
            if articles_files:
                latest_files["articles_file"] = str(max(articles_files, key=os.path.getctime))
        
        # Find latest papers file
        papers_dir = data_dir / "papers"
        if papers_dir.exists():
            papers_files = list(papers_dir.glob("arxiv_papers_*.json"))
            if papers_files:
                latest_files["papers_file"] = str(max(papers_files, key=os.path.getctime))
        
        # Find latest classifications file
        classifications_dir = data_dir / "classifications"
        if classifications_dir.exists():
            class_files = list(classifications_dir.glob("classifications_*.json"))
            if class_files:
                latest_files["classifications_file"] = str(max(class_files, key=os.path.getctime))
        
        # Find latest summaries file
        summaries_dir = data_dir / "summaries"
        if summaries_dir.exists():
            summary_files = list(summaries_dir.glob("summaries_*.json"))
            if summary_files:
                latest_files["summaries_file"] = str(max(summary_files, key=os.path.getctime))
        
        return latest_files
    
    def generate_digest_report(self, 
                              content_data: Optional[Dict[str, Any]] = None,
                              report_date: Optional[str] = None) -> DigestReport:
        """
        Generate a digest report from content data.
        
        Args:
            content_data: Content data dictionary. If None, loads latest files.
            report_date: Date string for the report. If None, uses today.
            
        Returns:
            DigestReport object.
        """
        if content_data is None:
            latest_files = self.find_latest_files()
            content_data = self.load_content_data(**latest_files)
        
        if report_date is None:
            report_date = date.today().strftime("%Y-%m-%d")
        
        report = DigestReport()
        report.title = self.config["report_settings"]["title_format"].format(date=report_date)
        report.date = report_date
        
        # Organize content by categories
        report.categories = self._organize_by_categories(content_data)
        
        # Generate statistics
        report.statistics = self._generate_statistics(content_data, report.categories)
        
        # Generate summary
        report.summary = self._generate_report_summary(report.statistics)
        
        return report
    
    def _organize_by_categories(self, content_data: Dict[str, Any], content_type: str = "all") -> Dict[str, List[Dict]]:
        """Organize content by categories from classifications."""
        categories = {}
        
        # Create mapping from content to classification and summary
        content_map = {}
          # Map classifications
        for item in content_data.get("classifications", []):
            content = item.get("content", {})
            
            # Skip if content_type is specified and doesn't match
            if content_type != "all" and content.get("type") != content_type:
                continue
            
            classification = item.get("classification", {})
            url = content.get("url", "") or content.get("link", "")
            if url:
                content_map[url] = {
                    "content": content,
                    "classification": classification,
                    "summary": None
                }
        
        # Map summaries
        for item in content_data.get("summaries", []):
            content = item.get("original_content", {})
            summary = item.get("summary", {})
            url = content.get("url", "") or content.get("link", "")
            if url and url in content_map:
                content_map[url]["summary"] = summary
        
        # Organize by categories
        min_confidence = self.config["report_settings"]["minimum_confidence"]
        max_items = self.config["report_settings"]["max_items_per_category"]
        
        for url, data in content_map.items():
            classification = data["classification"]
            confidence = classification.get("confidence", 0)
            
            # Filter by confidence
            if confidence < min_confidence:
                continue
            
            category = classification.get("category", "Other")
            
            if category not in categories:
                categories[category] = []
            
            # Combine all data
            item_data = {
                "content": data["content"],
                "classification": classification,
                "summary": data["summary"],
                "confidence": confidence
            }
            
            categories[category].append(item_data)
        
        # Sort and limit items per category
        sort_by_confidence = self.config["report_settings"]["sort_by_confidence"]
        
        for category in categories:
            if sort_by_confidence:
                categories[category].sort(key=lambda x: x["confidence"], reverse=True)
            
            categories[category] = categories[category][:max_items]
        
        # Sort categories by predefined order
        category_order = self.config["category_order"]
        ordered_categories = {}
        
        for category in category_order:
            if category in categories:
                ordered_categories[category] = categories[category]
        
        # Add any remaining categories
        for category in categories:
            if category not in ordered_categories:
                ordered_categories[category] = categories[category]
        
        return ordered_categories

    def generate_monthly_papers_report(self, content_data: Optional[Dict[str, Any]] = None,
                                     report_date: Optional[str] = None) -> DigestReport:
        """
        Generate a digest report for monthly papers.

        Args:
            content_data: Content data dictionary. If None, loads latest files.
            report_date: Date string for the report. If None, uses today.

        Returns:
            DigestReport object.
        """
        if content_data is None:
            latest_files = self.find_latest_files()
            content_data = self.load_content_data(**latest_files)

        if report_date is None:
            report_date = date.today().strftime("%Y-%m-%d")

        report = DigestReport()
        report.title = "🏆 Papers of the Month - " + report_date
        report.date = report_date

        # Organize content by categories (only papers)
        report.categories = self._organize_by_categories(content_data, content_type="papers")

        # Generate statistics
        report.statistics = self._generate_statistics(content_data, report.categories)

        # Generate summary
        report.summary = self._generate_report_summary(report.statistics)

        return report
    
    def _generate_statistics(self, content_data: Dict[str, Any], categories: Dict[str, List]) -> Dict[str, Any]:
        """Generate statistics for the report."""
        total_articles = len(content_data.get("articles", []))
        total_papers = len(content_data.get("papers", []))
        total_classified = len(content_data.get("classifications", []))
        total_summarized = len(content_data.get("summaries", []))
        
        category_counts = {cat: len(items) for cat, items in categories.items()}
        
        # Calculate confidence statistics
        confidences = []
        for items in categories.values():
            for item in items:
                confidences.append(item["confidence"])
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "total_articles": total_articles,
            "total_papers": total_papers,
            "total_classified": total_classified,
            "total_summarized": total_summarized,
            "total_in_digest": sum(category_counts.values()),
            "categories_count": len(categories),
            "category_breakdown": category_counts,
            "average_confidence": avg_confidence,
            "confidence_range": [min(confidences), max(confidences)] if confidences else [0, 0]
        }
    
    def _generate_report_summary(self, statistics: Dict[str, Any]) -> str:
        """Generate a summary for the report."""
        total_items = statistics["total_in_digest"]
        categories_count = statistics["categories_count"]
        avg_confidence = statistics["average_confidence"]
        
        summary = f"This digest contains {total_items} tech news items and research papers "
        summary += f"across {categories_count} categories. "
        summary += f"Average classification confidence: {avg_confidence:.2f}. "
        
        # Highlight top categories
        top_categories = sorted(
            statistics["category_breakdown"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        if top_categories:
            top_cat_names = [cat for cat, _ in top_categories]
            summary += f"Top categories: {', '.join(top_cat_names)}."
        
        return summary
    
    def generate_markdown_report(self, report: DigestReport) -> str:
        """
        Generate a markdown report.
        
        Args:
            report: DigestReport object.
            
        Returns:
            Markdown string.
        """
        md = []
        
        # Title and date
        md.append(f"# {report.title}")
        md.append(f"*Generated on {report.date}*")
        md.append("")
          # Table of contents
        if self.config["format_settings"]["markdown"]["include_toc"]:
            md.append("## Table of Contents")
            md.append("")

            for category in report.categories:
                # Use placeholder URLs that can be updated later with actual Discord message URLs
                md.append(f"- [{category}](#{category.lower().replace(' ', '-').replace('&', '').replace('/', '')})")
            md.append("")
        
        # Summary
        if self.config["report_settings"]["include_summary"]:
            md.append("## Summary")
            md.append("")
            md.append(report.summary)
            md.append("")
        
        # Statistics
        if self.config["report_settings"]["include_statistics"]:
            md.append("## Statistics")
            md.append("")
            md.append(f"- **Total Items**: {report.statistics['total_in_digest']}")
            md.append(f"- **Categories**: {report.statistics['categories_count']}")
            md.append(f"- **Average Confidence**: {report.statistics['average_confidence']:.2f}")
            md.append("")
            
            # Category breakdown
            md.append("### Category Breakdown")
            md.append("")
            for category, count in report.statistics["category_breakdown"].items():
                md.append(f"- **{category}**: {count} items")
            md.append("")
        
        # Content by categories (only if include_content is enabled)
        if self.config["report_settings"].get("include_content", True):
            for category, items in report.categories.items():
                if not items:
                    continue
                    
                md.append(f"## {category}")
                md.append("")
                
                for i, item in enumerate(items, 1):
                    content = item["content"]
                    classification = item["classification"]
                    summary = item["summary"]
                    
                    # Title and metadata
                    title = content.get("title", "Unknown Title")
                    md.append(f"### {i}. {title}")
                    md.append("")
                    
                    # Classification info
                    subcategory = classification.get("subcategory", "")
                    confidence = classification.get("confidence", 0)
                    tags = classification.get("tags", [])
                    
                    if subcategory:
                        md.append(f"**Subcategory**: {subcategory}")
                    md.append(f"**Confidence**: {confidence:.2f}")
                    if tags:
                        md.append(f"**Tags**: {', '.join(tags)}")
                    md.append("")
                    
                    # Summary
                    if summary:
                        summary_text = summary.get("summary", "")
                        key_points = summary.get("key_points", [])
                        impact = summary.get("impact", "")
                        technical_level = summary.get("technical_level", "")
                        
                        if summary_text:
                            md.append(f"**Summary**: {summary_text}")
                            md.append("")
                        
                        if key_points:
                            md.append("**Key Points**:")
                            for point in key_points:
                                md.append(f"- {point}")
                            md.append("<br>")
                        
                        if impact:
                            md.append(f"**Impact**: {impact}")
                            md.append("<br>")

                        if technical_level:
                            md.append(f"**Technical Level**: {technical_level}")
                            md.append("<br>")

                    # Original description
                    description = content.get("description", "")
                    if description:
                        md.append(f"**Description**: {description}")
                        md.append("<br>")

                    # URL
                    url = content.get("url", "") or content.get("link", "")
                    if url:
                        md.append(f"**URL**: [{url}]({url})")
                        md.append("<br>")

                    # Authors (for papers)
                    authors = content.get("authors", [])
                    if authors:
                        if isinstance(authors, list):
                            authors_str = ", ".join(authors)
                        else:
                            authors_str = str(authors)
                        md.append(f"**Authors**: {authors_str}")
                        md.append("<br>")

                    md.append("---")
                    md.append("")
        
        # Footer
        md.append("---")
        md.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(md)
    
    def save_report(self,
                   report: DigestReport, 
                   format: str = "markdown",
                   output_path: Optional[str] = None) -> str:
        """
        Save a report to file.
        
        Args:
            report: DigestReport object.
            format: Output format ("markdown" or "html").
            output_path: Custom output path.
            
        Returns:
            Path to the saved file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path is None:
            os.makedirs("data/reports", exist_ok=True)
            if format == "markdown":
                output_path = f"data/reports/digest_{timestamp}.md"
            else:
                output_path = f"data/reports/digest_{timestamp}.html"
        
        if format == "markdown":
            content = self.generate_markdown_report(report)
        else:
            # For now, just use markdown (HTML generation can be added later)
            content = self.generate_markdown_report(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"Saved {format} report to {output_path}")
        return output_path
    
    

    def generate_toc_only_report(self, report: DigestReport, discord_urls: Optional[Dict[str, str]] = None,
                                paper_urls: Optional[Dict[str, str]] = None) -> str:
        """
        Generate a minimal markdown report with only table of contents.
        
        Args:
            report: DigestReport object
            discord_urls: Optional dictionary mapping categories to Discord message URLs
            paper_urls: Optional dictionary mapping paper section names to Discord URLs
            include_papers_sections: Whether to include papers sections in TOC
            
        Returns:
            Markdown string with only TOC
        """
        md = []
        
        # Title and date
        md.append(f"# {report.title}")
        md.append(f"*Generated on {report.date}*")
        md.append("")
        
        # Table of contents only
        md.append("## Table of Contents")
        md.append("")
        md.append("News Categories:")
        
        # Add category links (news sections)
        category_count = 0
        for category in report.categories:
            if discord_urls and category in discord_urls:
                # Use actual Discord URL
                md.append(f"- [{category}]({discord_urls[category]})")
            else:
                # Use placeholder that can be updated later
                md.append(f"- [{category}](#{category.lower().replace(' ', '-').replace('&', '').replace('/', '')})")
            category_count += 1
        
        # Add papers sections (always sent to Discord when there are papers)
        papers_count = sum(len(items) for items in report.categories.values() if any(item.get("content", {}).get("authors") for item in items))
        if papers_count > 0:
            if category_count > 0:
                md.append("")  # Add space if there were categories above
            md.append("### Research Papers")
            
            # Use real paper URLs if provided, otherwise use placeholders
            if paper_urls:
                # Use real Discord URLs for paper sections
                hf_url = paper_urls.get("HuggingFace Papers", paper_urls.get("huggingface", ""))
                arxiv_url = paper_urls.get("arXiv Papers", paper_urls.get("arxiv", ""))
                
                if hf_url:
                    md.append(f"- [🏆 Papers of the Month (HuggingFace)]({hf_url})")
                if arxiv_url:
                    md.append(f"- [📚 Recent Papers (arXiv)]({arxiv_url})")
                    
                if not hf_url and not arxiv_url:
                    # Fallback if paper_urls dict is empty or has unexpected keys
                    for section_name, url in paper_urls.items():
                        md.append(f"- [📄 {section_name}]({url})")
            else:
                # Use placeholder URLs that indicate real URLs should be captured
                md.append(f"- [🏆 Papers of the Month (HuggingFace)](#huggingface-papers-placeholder)")
                md.append(f"- [📚 Recent Papers (arXiv)](#arxiv-papers-placeholder)")
        
        md.append("")
        
        # Footer
        md.append("---")
        md.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        md.append("")
        md.append("*Click on category links above to view the content in Discord*")
        
        return "\n".join(md)
