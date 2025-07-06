import logging
import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime 
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.schema import HumanMessage, SystemMessage
from langchain_together import ChatTogether
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from utils.rate_limiter import together_rate_limiter
from utils.token_manager import token_manager
from utils.robust_parser import robust_parser
from utils.html_utils import clean_html
from utils.paper_chunker import PaperChunker
import time
from arxiv import Result as PaperResult

class ContentSummary(BaseModel):
    """Pydantic model for content summarization results."""
    title: str = Field(description="Original title of the content")
    summary: str = Field(description="Concise summary of the content")
    key_points: List[str] = Field(description="List of key points from the content")
    impact: str = Field(description="Potential impact or significance")
    technical_level: str = Field(description="Technical complexity level: beginner, intermediate, advanced")
    confidence: float = Field(description="Confidence score for the summary quality")


class TechContentSummarizer:
    """
    AI-powered summarizer for tech news articles and research papers.
    Generates concise, informative summaries with key insights.
    """
    
    def __init__(self, config_path: str = "config/summarization.yml"):
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.config = self._load_config()
        self.llm_summarizer, self.llm_chunker = self._setup_llms()
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _load_config(self) -> Dict[str, Any]:
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
        return {
            "summary_settings": {
                "max_summary_length": 200,
                "max_key_points": 5,
                "include_technical_analysis": True,
                "include_impact_assessment": True
            },
            "model_settings": {
                "temperature": 0.3,
                "max_tokens": None,
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
            }
        }

    def _setup_llms(self) -> List[ChatTogether | None]:
        api_keys = [os.getenv("TOGETHER_API_KEY_summarizer"), os.getenv("GROQ_API_KEY")]
        if not all(api_keys):
            self.logger.warning("TOGETHER_API_KEY not found. Summarization will use fallback method.")
            return [None, None]
        model_settings = self.config.get("model_settings", {})
        llm_summarizer = ChatTogether(
            api_key=api_keys[0],
            model=model_settings.get("summary_model", "meta-llama/llama-3.3-70b-instruct-turbo-free"),
            temperature=model_settings.get("temperature", 0.3),
            max_tokens=model_settings.get("max_tokens", 1200)
        )

        llm_chunker = ChatGroq(
            api_key=api_keys[1],
            model=model_settings.get("chunk_model", "gemma2-9b-it"),
            temperature=model_settings.get("temperature", 0.3),
            max_tokens=model_settings.get("max_tokens", 600)
        )
        
        return [llm_summarizer, llm_chunker]

    def summarize_news_article(self, article: Dict[str, str], content: Optional[str] = None) -> ContentSummary:
        try:
            if self.llm_summarizer:
                together_rate_limiter.wait_if_needed()
                return self._summarize_news_with_ai(article, content)
            else:
                return self._create_fallback_summary(article, "news")
        except Exception as e:
            self.logger.error(f"Summarization failed for article '{article.get('title', 'Unknown')}': {e}")
            return self._create_fallback_summary(article, "news")

    def _get_paper_md(self, id: str, dir_path: str="/data/papers") -> str:
        """
        Read Markdown content from a file.
        """
        
        path = os.path.join(dir_path, f"{id}.md")
        try:
            with open(path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            self.logger.error(f"Failed to read Markdown file {path}: {e}")
            return ""

    def summarize_research_paper(self, paper: PaperResult, data_dir: str) -> ContentSummary:
        """
        Enhanced research paper summarization with Markdown support and relevance-based chunk selection.
        """
        try:
            if not self.llm_chunker:
                paper_content = {
                    "title": paper.title,
                    "summary": paper.summary,
                }
                return self._create_fallback_summary(paper_content, "papers")
                
            chunker = PaperChunker()
            abstract = paper.summary
            markdown_content = self._get_paper_md(paper.get_short_id(), data_dir)

            # Split markdown into chunks
            all_chunks = chunker.chunk_markdown(markdown_content)
            if not all_chunks:
                raise ValueError("No valid chunks extracted from Markdown content")

            # Skip first 10%
            total_chunks = len(all_chunks)
            start_idx = int(total_chunks * 0.1)
            selected_chunks = all_chunks[start_idx:]

            # Apply relevance scoring if abstract exists
            if abstract:
                selected_chunks = chunker.compute_relevance_scores(abstract, selected_chunks)
                selected_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

            print(f"Selected chunks: {selected_chunks}")
            selected_chunks = selected_chunks[:10]  # Limit to top 10 relevant chunks

            if not selected_chunks:
                # Fallback to abstract-only summarization
                return self._summarize_paper_with_ai(paper, abstract)

            # Summarize chunks in parallel
            chunk_summaries = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for i, chunk in enumerate(selected_chunks):
                    # Add small delay to avoid rate limit bursts
                    if i > 0:
                        time.sleep(0.5)
                    futures.append(executor.submit(
                        self._summarize_chunk,
                        chunk,
                        paper.title,
                        abstract
                    ))
                future_to_chunk = {future: chunk for future, chunk in zip(futures, selected_chunks)}
                
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        result = future.result()
                        chunk_summaries.append(result)
                    except Exception as e:
                        self.logger.warning(f"Chunk summarization failed for {chunk['header']}: {e}")
                        chunk_summaries.append({
                            'header': chunk['header'],
                            'type': chunk['type'],
                            'summary': f"Summary unavailable: {e}"
                        })

            # Combine chunk summaries
            together_rate_limiter.wait_if_needed()
            context = "\n\n".join(
                f"### {s['header']} ({s['type']})\n{s['summary']}"
                for s in chunk_summaries
            )
            context = token_manager.prepare_content_for_summarization(
                context, max_tokens=3000, model_provider='together'
            )
            
            # Generate final paper summary
            return self._summarize_paper_with_ai(paper, context)

        except Exception as e:
            self.logger.error(f"Paper summarization failed for '{paper.title}': {e}")
            paper_content = {
                "title": paper.title,
                "summary": paper.summary,
            }
            return self._create_fallback_summary(paper_content, "papers")

    def _summarize_chunk(self, chunk: Dict[str, Any], title: str, abstract: str) -> Dict[str, Any]:
        """Summarize a single chunk with rate limiting and error handling."""
        if not self.llm_chunker:
            return {
                'header': chunk['header'],
                'type': chunk['type'],
                'summary': f"Fallback summary: {chunk['chunk_content'][:200]}..."
            }
            
        together_rate_limiter.wait_if_needed()
        chunk_content = token_manager.prepare_content_for_summarization(
            chunk['chunk_content'], max_tokens=1000, model_provider='groq'
        )
        prompt = self._get_chunk_prompt(title, abstract, chunk_content)
        messages = [
            SystemMessage(content="You are an advanced language model tasked with summarizing academic papers. Respond with concise, informative summaries only."),
            HumanMessage(content=prompt)
        ]
        try:
            response = self.llm_chunker.invoke(messages)
            summary_text = response.content if hasattr(response, 'content') else str(response)
            return {
                'header': chunk['header'],
                'type': chunk['type'],
                'summary': summary_text.strip()
            }
        except Exception as e:
            raise Exception(f"Failed to summarize chunk {chunk['header']}: {e}")

    def _get_chunk_prompt(self, title: str, abstract: str, chunk_content: str) -> str:
        prompt = f"""
        Your task is to read and summarize the following section refer to the abstraction of a research paper.
        Focus on extracting the key ideas, main arguments, and essential findings while maintaining clarity and coherence.
        The summary should be concise, approximately 3-5 sentences long, and should capture the essence of the text without losing important details.
        Here is the context to summarize:
        Paper Title: {title} \n
        Abstract: {abstract} \n
        Section: {chunk_content} \n
        Provide the summary below:
        """

        return prompt

    def _summarize_news_with_ai(self, article: Dict[str, str], content: Optional[str]) -> ContentSummary:
        full_content = content or article.get('content', '') or article.get('description', '')
        processed_content = token_manager.prepare_content_for_summarization(
            full_content, max_tokens=3000, model_provider='together'
        )
        prompt_text = f"""
        Article Details:
        Title: {article.get('title', '')}
        Description: {article.get('description', '')}
        Full Content: {processed_content}
        URL: {article.get('url', '')}

        Instructions:
        1. Create a concise summary (max 500 words)
        2. Extract at least 2 key points
        3. Assess the potential impact
        4. Determine the technical complexity level
        5. Provide a confidence score

        Respond with valid JSON:
        {{
            "summary": "concise summary",
            "key_points": ["point 1", "point 2", ...],
            "impact": "impact analysis",
            "technical_level": "beginner/intermediate/advanced",
            "confidence": 0.85
        }}
        """
        messages = [
            SystemMessage(content="You are an expert tech journalist and analyst. Always respond with valid JSON and nothing else."),
            HumanMessage(content=prompt_text)
        ]
        together_rate_limiter.wait_if_needed()
        if self.llm_summarizer is None:
            return self._create_fallback_summary(article, "news")
        try:
            response = self.llm_summarizer.invoke(messages)
            content_text = response.content if hasattr(response, 'content') else str(response)
            parsed_result = robust_parser.parse_summarization_response(
                content_text, original_title=article.get('title', '')
            )
            return ContentSummary(
                title=parsed_result['title'],
                summary=parsed_result['summary'],
                key_points=parsed_result['key_points'],
                impact=parsed_result['impact'],
                technical_level=parsed_result['technical_level'],
                confidence=parsed_result['confidence']
            )
        except Exception as e:
            self.logger.warning(f"LLM summarization failed: {e}")
            return self._create_fallback_summary(article, "news")

    def _summarize_paper_with_ai(self, paper: PaperResult, content: Optional[str]) -> ContentSummary:

        abstract = paper.summary
        
        if content:
            # If context is provided, combine abstract and context
            combined_content = f"{abstract}\n\n{content}" if abstract else content
        else:
            # If no context, use abstract only
            combined_content = abstract
            
        processed_context = token_manager.prepare_content_for_summarization(
            combined_content, max_tokens=3000, model_provider='together'
        )
        prompt_text = f"""
        Paper Metadata:
        Title: {paper.title}
        Authors: {', '.join([author.name for author in paper.authors]) if paper.authors else 'Unknown'}
        Categories: {', '.join(paper.categories) if isinstance(paper.categories, list) else paper.categories}

        Content: {processed_context}

        Instructions:
        1. Identify the core research problem and methodology
        2. Highlight significant findings
        3. Focus on novel contributions
        4. Keep summary concise (250-300 words)
        5. Extract at least 2 key technical points
        6. Assess impact and applications
        7. Rate technical complexity

        Respond with valid JSON:
        {{
            "title": "paper title",
            "applications": "real-world applications",
            "summary": "concise overall summary",
            "key_points": ["point 1", "point 2", ...],
            "impact": "impact analysis",
            "technical_level": "beginner/intermediate/advanced",
            "confidence": 0.85
        }}
        """
        messages = [
            SystemMessage(content="You are an expert researcher and technical analyst. Always respond with valid JSON."),
            HumanMessage(content=prompt_text)
        ]
        together_rate_limiter.wait_if_needed()
        if self.llm_summarizer is None:
            paper_content = {
                    "title": paper.title,
                    "summary": paper.summary,
                }
            return self._create_fallback_summary(paper_content, "papers")
        try:
            response = self.llm_summarizer.invoke(messages)
            content_text = response.content if hasattr(response, 'content') else str(response)
            parsed_result = robust_parser.parse_summarization_response(
                content_text, original_title=paper.title
            )
            return ContentSummary(
                title=parsed_result['title'],
                summary=parsed_result['summary'],
                key_points=parsed_result['key_points'],
                impact=parsed_result['impact'],
                technical_level=parsed_result['technical_level'],
                confidence=parsed_result['confidence']
            )
        except Exception as e:
            self.logger.warning(f"LLM paper summarization failed: {e}")
            paper_content = {
                    "title": paper.title,
                    "summary": paper.summary,
                }
            return self._create_fallback_summary(paper_content, "papers")

    def _create_fallback_summary(self, content: Dict[str, str], content_type: str = "news") -> Any:
        title = content.get('title', 'Unknown ' + ('Article' if content_type == "news" else 'Paper'))
        text = clean_html(content.get('description', '') or content.get('summary', '') or content.get('abstract', ''))
        summary = text[:300] if len(text) > 300 else text
        if not summary:
            summary = f"{content_type} about {title}"
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 20]
        key_points = sentences[:3] if sentences else [f"Content discusses {title.lower()}"]
        if content_type == "news":
            return ContentSummary(
                title=title,
                summary=summary,
                key_points=key_points,
                impact="Technology industry impact",
                technical_level="intermediate",
                confidence=0.5
            )

    def summarize_batch(self, content_list: List[Any], 
                       content_texts: Optional[List[str]] = None,
                       content_type: str = "news",
                       data_dir: str = "data/markdown") -> List[ContentSummary]:
        results = []
        for i, content in enumerate(content_list):
            text = content_texts[i] if content_texts and i < len(content_texts) else None
            try:
                if content_type == "papers":
                    # content is arxiv.Result
                    summary = self.summarize_research_paper(content, data_dir)
                else:
                    # content is Dict[str, str]
                    summary = self.summarize_news_article(content, text)
                results.append(summary)
                
                # Get title for logging
                title = content.title if content_type == "papers" else content.get('title', 'Unknown')
                self.logger.info(f"Summarized {content_type} {i+1}/{len(content_list)}: {title[:50]}...")
            except Exception as e:
                self.logger.error(f"Failed to summarize {content_type} {i+1}: {e}")
                
                # Create fallback content dict for papers
                if content_type == "papers":
                    fallback_content = {"title": content.title, "summary": content.summary}
                else:
                    fallback_content = content
                    
                fallback = self._create_fallback_summary(fallback_content, content_type)
                results.append(fallback)
        return results

    def save_summaries(self, summaries: List[ContentSummary], 
                      content_list: List[Dict[str, str]], 
                      filename: Optional[str] = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summaries_{timestamp}.json"
        os.makedirs("data/summaries", exist_ok=True)
        filepath = os.path.join("data/summaries", filename)
        results = []
        for content, summary in zip(content_list, summaries):
            result = {**content, 'summary_data': summary.model_dump(), 'summary': summary.summary, 'timestamp': datetime.now().isoformat()}
            results.append(result)
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved {len(summaries)} summaries to {filepath}")
        return filepath
