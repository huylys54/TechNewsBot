import logging
import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain.schema import HumanMessage, SystemMessage
from langchain_together import ChatTogether
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, SecretStr
from utils.rate_limiter import together_rate_limiter
from utils.token_manager import token_manager
from utils.robust_parser import robust_parser
from utils.html_utils import clean_html
from utils.paper_chunker import PaperChunker
from arxiv import Result as PaperResult

class ContentSummary(BaseModel):
    """Pydantic model for content summarization results."""
    title: str = Field(description="Original title of the content")
    applications: Optional[str] = Field(description="Real-world applications of the research", default=None)
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
            api_key=SecretStr(api_keys[1]) if api_keys[1] else None,
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
        Summarizes a research paper using relevance-scored chunks with a section-aware,
        dynamic selection strategy to ensure comprehensive coverage.
        """
        try:
            if not self.llm_chunker or not self.llm_summarizer:
                return self._create_fallback_summary({"title": paper.title, "summary": paper.summary}, "papers")

            chunker = PaperChunker()
            abstract = paper.summary
            markdown_content = self._get_paper_md(paper.get_short_id(), data_dir)

            if not markdown_content.strip():
                self.logger.warning(f"Markdown for '{paper.title}' is empty. Summarizing abstract only.")
                return self._summarize_paper_with_ai(paper, None)

            all_chunks = chunker.chunk_markdown(markdown_content)
            if not all_chunks:
                self.logger.warning(f"No chunks from '{paper.title}'. Summarizing abstract only.")
                return self._summarize_paper_with_ai(paper, None)

            # Score all chunks based on relevance to the abstract
            scored_chunks = chunker.compute_relevance_scores(abstract, all_chunks)
            
            # Group chunks by their classified section type
            sections = {}
            for chunk in scored_chunks:
                sec_type = chunk.get('type', 'other')
                if sec_type not in sections:
                    sections[sec_type] = []
                sections[sec_type].append(chunk)

            # Select the top N chunks from each section to ensure diversity
            diverse_chunks = []
            for sec_type, chunks_in_sec in sections.items():
                # Sort chunks within the section by relevance
                chunks_in_sec.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                # Take top 2 from important sections, 1 from others
                limit = 2 if sec_type in ['introduction', 'methodology', 'results', 'conclusion'] else 1
                diverse_chunks.extend(chunks_in_sec[:limit])
            
            if not diverse_chunks:
                self.logger.warning(f"No diverse chunks selected for '{paper.title}'. Summarizing abstract only.")
                return self._summarize_paper_with_ai(paper, None)

            # Summarize the selected diverse chunks in parallel
            chunk_summaries = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_chunk = {
                    executor.submit(self._summarize_chunk_by_type, chunk, paper.title): chunk
                    for chunk in diverse_chunks
                }
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        summary_text = future.result()
                        if summary_text:
                            chunk_summaries.append({
                                'type': chunk.get('type', 'other'),
                                'summary': summary_text
                            })
                    except Exception as e:
                        self.logger.error(f"Chunk summarization failed for {chunk.get('header', 'Unknown')}: {e}")

            if not chunk_summaries:
                self.logger.warning(f"All chunk summaries failed for '{paper.title}'. Summarizing abstract only.")
                return self._summarize_paper_with_ai(paper, None)

            # Combine chunk summaries for the final synthesis step
            synthesis_context = "\n\n".join(
                f"## Section: {s['type'].replace('_', ' ').title()}\n{s['summary']}"
                for s in chunk_summaries
            )
            
            return self._summarize_paper_with_ai(paper, synthesis_context)

        except Exception as e:
            self.logger.error(f"Paper summarization failed for '{paper.title}': {e}")
            return self._create_fallback_summary({"title": paper.title, "summary": paper.summary}, "papers")

    def _summarize_chunk_by_type(self, chunk: Dict[str, Any], title: str) -> str:
        """Summarizes a single chunk using a prompt targeted to its section type."""
        if not self.llm_chunker:
            return ""

        together_rate_limiter.wait_if_needed()
        
        chunk_content = token_manager.prepare_content_for_summarization(
            chunk['chunk_content'], max_tokens=1500, model_provider='groq'
        )
        section_type = chunk.get('type', 'other')
        
        prompt = self._get_targeted_chunk_prompt(title, section_type, chunk_content)
        messages = [
            SystemMessage(content="You are an academic assistant. Your task is to summarize a specific section of a research paper based on a targeted prompt. Focus only on the information requested."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm_chunker.invoke(messages)
            summary_text = response.content if hasattr(response, 'content') else str(response)
            return summary_text.strip()
        except Exception as e:
            self.logger.warning(f"Could not summarize chunk for paper '{title}': {e}")
            return ""

    def _get_targeted_chunk_prompt(self, title: str, section_type: str, chunk_content: str) -> str:
        """Generates a targeted prompt for summarizing a chunk based on its section type."""
        
        prompts = {
            'introduction': "Based on the introduction text below, what is the core research problem, the key question the authors are trying to answer, and their main objective?",
            'methodology': "Based on the methodology section below, describe the primary methods, techniques, and models used in this research. Focus on the 'how', not the results.",
            'results': "Based on the results section below, what are the main experimental findings and key quantitative or qualitative results? Be specific and data-driven.",
            'conclusion': "Based on the conclusion and discussion below, what are the main takeaways, the authors' interpretation of the results, and the potential implications or future directions mentioned?",
            'abstract': "Summarize the abstract in 2-3 sentences, capturing the essence of the paper.",
            'related_work': "Briefly describe the main related works mentioned and how this paper differs or builds upon them.",
            'other': "Provide a concise summary of the following text from the research paper."
        }
        
        instruction = prompts.get(section_type, prompts['other'])

        return f"""
        Paper Title: {title}
        Section Type: {section_type.replace('_', ' ').title()}

        Instruction: {instruction}

        Content of the chunk to summarize:
        ---
        {chunk_content}
        ---

        Provide a concise summary based *only* on the content of the chunk provided.
        """

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
        2. Extract at least 2 key points, using easy-to-understand language
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
                title=parsed_result.get('title', article.get('title', '')),
                summary=parsed_result.get('summary', ''),
                key_points=parsed_result.get('key_points', []),
                impact=parsed_result.get('impact', ''),
                technical_level=parsed_result.get('technical_level', 'intermediate'),
                confidence=parsed_result.get('confidence', 0.5),
                applications=parsed_result.get('applications', None)
            )
        except Exception as e:
            self.logger.warning(f"LLM summarization failed: {e}")
            return self._create_fallback_summary(article, "news")

    def _summarize_paper_with_ai(self, paper: PaperResult, content: Optional[str]) -> ContentSummary:
        """
        Generates the final, structured summary of a research paper by synthesizing
        section summaries or using the abstract as a fallback.
        """
        abstract = paper.summary
        
        if not content:
            self.logger.warning(f"No synthesized content for '{paper.title}'. Using abstract for final summary.")
            processed_context = token_manager.prepare_content_for_summarization(
                abstract, max_tokens=3000, model_provider='together'
            )
            context_source_description = "This summary is based on the paper's abstract."
        else:
            processed_context = token_manager.prepare_content_for_summarization(
                content, max_tokens=4000, model_provider='together'
            )
            context_source_description = "This summary was synthesized from key sections of the paper:"

        prompt_text = f"""
        You are an expert researcher and technical analyst. Your task is to create a high-quality, structured summary of a research paper.

        Paper Metadata:
        - Title: {paper.title}
        - Authors: {', '.join([author.name for author in paper.authors]) if paper.authors else 'N/A'}
        - Original Abstract (for reference only): {abstract}

        {context_source_description}
        ---
        {processed_context}
        ---

        Instructions:
        1.  **Narrative Summary:** Write a narrative summary (250-300 words) that tells the story of the research. Start with the core problem, explain the proposed method, and conclude with the primary outcomes. This should be for a technical audience who has not read the paper. Do NOT simply rephrase the original abstract.
        2.  **Key Points:** Extract exactly 3-5 distinct and specific takeaways. Each point must be a complete sentence and a concrete finding or contribution. Do not use generic statements. These points should be different from the narrative summary.
        3.  **Impact & Applications:** Briefly describe the potential impact of this research and its real-world applications.
        4.  **Technical Level:** Rate the technical complexity as "Beginner", "Intermediate", or "Advanced".
        5.  **Confidence Score:** Provide a confidence score (0.0 to 1.0) for the summary quality.

        Respond with a valid JSON object matching this structure exactly:
        {{
            "title": "{paper.title}",
            "applications": "concise description of real-world applications",
            "summary": "The narrative summary as described in instruction #1.",
            "key_points": [
                "A specific, concrete key point.",
                "Another distinct, data-driven finding."
            ],
            "impact": "A brief analysis of the research's potential impact.",
            "technical_level": "Beginner/Intermediate/Advanced",
            "confidence": 0.9
        }}
        """
        messages = [
            SystemMessage(content="You are an expert researcher and technical analyst. Always respond with valid JSON and nothing else."),
            HumanMessage(content=prompt_text)
        ]
        
        together_rate_limiter.wait_if_needed()
        if self.llm_summarizer is None:
            return self._create_fallback_summary({"title": paper.title, "summary": paper.summary}, "papers")

        try:
            response = self.llm_summarizer.invoke(messages)
            content_text = response.content if hasattr(response, 'content') else str(response)
            parsed_result = robust_parser.parse_summarization_response(
                content_text, original_title=paper.title
            )
            return ContentSummary(
                title=parsed_result.get('title', paper.title),
                summary=parsed_result.get('summary', 'Summary not available.'),
                key_points=parsed_result.get('key_points', []),
                impact=parsed_result.get('impact', 'Impact analysis not available.'),
                technical_level=parsed_result.get('technical_level', 'intermediate'),
                confidence=parsed_result.get('confidence', 0.5),
                applications=parsed_result.get('applications', 'Not specified.')
            )
        except Exception as e:
            self.logger.error(f"LLM paper summarization failed for '{paper.title}': {e}")
            return self._create_fallback_summary({"title": paper.title, "summary": paper.summary}, "papers")

    def _create_fallback_summary(self, content: Dict[str, str], content_type: str = "news") -> ContentSummary:
        title = content.get('title', 'Unknown ' + ('Article' if content_type == "news" else 'Paper'))
        text = clean_html(content.get('description', '') or content.get('summary', '') or content.get('abstract', ''))
        summary = text[:300] if len(text) > 300 else text
        if not summary:
            summary = f"A {content_type} about {title}"
        
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 20]
        key_points = sentences[:3] if sentences else [f"Content discusses {title.lower()}."]
        
        impact = "Technology industry impact" if content_type == "news" else "Impact analysis not available."
        
        return ContentSummary(
            title=title,
            summary=summary,
            key_points=key_points,
            impact=impact,
            technical_level="intermediate",
            confidence=0.5,
            applications="Applications not specified."
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
        p = Path(filepath)
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved {len(summaries)} summaries to {filepath}")
        return filepath
