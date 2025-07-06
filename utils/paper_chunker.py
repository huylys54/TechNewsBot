"""Enhanced paper chunking using Markdown headers for summarization pipeline"""
import re
import logging
from typing import List, Dict
import numpy as np
import torch
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2", 
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
class PaperChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.logger = logging.getLogger(__name__)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_markdown(self, markdown: str) -> List[Dict]:
        """Hierarchically splits markdown by headers, then ensures chunks < max_len"""
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header1"),
                ("##", "Header2"), 
                ("###", "Header3")
            ],
            strip_headers=False
        )
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = []
        header_docs = header_splitter.split_text(markdown)
        has_headers = bool(header_docs and any(doc.metadata for doc in header_docs))

        if has_headers:
            for doc in header_docs:
                if len(doc.page_content) > self.chunk_size:
                    split_texts = char_splitter.split_text(doc.page_content)
                    for text in split_texts:
                        if text.strip():
                            meta = doc.metadata.copy()
                            chunks.append(self._create_chunk_dict(text, meta))
                elif doc.page_content.strip():
                    chunks.append(self._create_chunk_dict(doc.page_content, doc.metadata))
        else:
            split_texts = char_splitter.split_text(markdown)
            for text in split_texts:
                if text.strip():
                    chunks.append(self._create_chunk_dict(text, {}))

        if not chunks and markdown.strip():
            chunks.append(self._create_chunk_dict(markdown.strip()[:self.chunk_size], {}))

        return [chunk for chunk in chunks if chunk["chunk_content"].strip()]

    def _create_chunk_dict(self, text: str, metadata: Dict) -> Dict:
        """Helper to create standardized chunk dictionary"""
        header = self._extract_header(text)
        return {
            "chunk_id": self._generate_chunk_id(text, metadata),
            "chunk_content": text.strip(),
            "keep": None,
            "header": header,
            "type": self._classify_section_type(header or text),
            "source": "markdown",
            **metadata
        }

    def _generate_chunk_id(self, text: str, metadata: Dict) -> str:
        """Generate a unique ID for the chunk"""
        import hashlib
        unique_str = f"{text[:50]}{str(metadata)}"
        return f"chunk_{hashlib.md5(unique_str.encode()).hexdigest()[:8]}"

    def compute_relevance_scores(self, abstract: str, chunks: List[Dict]) -> List[Dict]:
        """Compute relevance scores for chunks based on similarity to abstract"""
        if not abstract:
            return chunks
        abstract_embedding = embedding.embed_query(abstract)
        for chunk in chunks:
            chunk_text = chunk['chunk_content']
            chunk_embedding = embedding.embed_query(chunk_text)
            similarity = self._cosine_similarity(abstract_embedding, chunk_embedding)
            chunk['relevance_score'] = similarity
        return chunks

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

    def _extract_header(self, content: str) -> str:
        lines = content.split('\n', 1)
        if len(lines) > 1:
            header = lines[0].strip()
            if len(header.split()) <= 10 and not header.endswith('.'):
                return header
        header_match = re.match(r'^(\d+\.|\#{2,})\s*([^\n]+)', content)
        return header_match.group(2).strip() if header_match else ""

    def _classify_section_type(self, text: str) -> str:
        text_lower = text.lower().strip()
        clean_text = re.sub(r'^(\d+\.)+\s*|[A-Z]\.\s*', '', text_lower)
        section_map = {
            'abstract': ['abstract', 'summary'],
            'introduction': ['intro', 'background', 'motivation'],
            'methodology': ['method', 'approach', 'algorithm', 'model', 'framework'],
            'results': ['result', 'finding', 'experiment', 'evaluation', 'analysis'],
            'conclusion': ['conclu', 'discuss', 'implication', 'future work'],
            'related_work': ['related', 'literature', 'prior work'],
            'appendix': ['appendix', 'supplement'],
            'references': ['reference', 'bibliography']
        }
        for section_type, keywords in section_map.items():
            if any(kw in clean_text for kw in keywords):
                return section_type
        return 'other'
