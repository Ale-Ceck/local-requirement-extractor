from collections import defaultdict
from typing import List, Optional, Tuple
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer
import os

from utils.logging_config import setup_logger

logger = setup_logger(__name__)

class MarkdownSplitter:
    """Split Markdown file in chunks ready to be processed by LLM."""

    def __init__(self):
        """Initialize the Markdown Splitter"""
        self.logger = logger

    def split_markdown(self, 
                       markdown_path: str, 
                       headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
                       strip_headers = False # Mantains headers into chunk's content
                       ) -> List[Document]:
        """
        Splits a markdown file into chunks based on headers using MarkdownHeaderTextSplitter.

        Args:
            markdown_path (str): Path to the markdown (.md) file.
            headers_to_split_on (Optional[List[Tuple[str, str]]]): 
                A list of tuples specifying which headers to split on.
                Example: [("#", "Title"), ("##", "Section")]

        Returns:
            List[Document]: A list of optimized Document objects.
        
        Raises:
            ValueError: If the file is empty or headers are incorrectly specified.
            FileNotFoundError: If the markdown file is not found.
            Exception: For any other unexpected errors.
        """
        if not os.path.isfile(markdown_path):
            self.logger.error(f"File not found: {markdown_path}")
            return []
        
        try:
            with open(markdown_path, "r", encoding="utf-8") as file:
                markdown_text = file.read()
        except Exception as e:
            self.logger.error(f"Error reading the file: {e}")
            return []
        
        if not markdown_text.strip():
            self.logger.error("The markdown file is empty.")
            return []

        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#",  "Header 1"),
                ("##", "Header 2")
            ]

        if not isinstance(headers_to_split_on, list) or not all(
            isinstance(h, tuple) and len(h) == 2 for h in headers_to_split_on
        ):
            self.logger.error("headers_to_split_on must be a list of 2-element tuples (level, name).")
            return []

        try:
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers= False)
            documents = splitter.split_text(markdown_text)

            optimized_chunks = optimize_chunks(documents=documents)

            return optimized_chunks
        
        except Exception as e:
            self.logger.error(f"Error during markdown splitting: {e}")
            return []
        
def optimize_chunks(documents: List[Document], tokenizer=None, max_tokens: int = 3500) -> List[Document]:
    """
    Optimize chunk length in order to reduce the number of calls to the LLM.
    First, it sub-splits any chunk that is too large.
    Then, it merges smaller chunks that share the same Header 1 without exceeding LLM's context window.

    Args:
        documents: List of Document objects from the initial split.
        tokenizer: Optional tokenizer for accurate token counting.
        max_tokens: Maximum tokens per optimized chunk.
    """
    if not documents:
        return []
    
    processable_docs = []
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(max_tokens * 3.8),  # Using heuristic: ~3.8 chars/token
        chunk_overlap=200
    )

    for doc in documents:
        doc_tokens = count_tokens(doc.page_content, tokenizer)

        if doc_tokens > max_tokens:
            logger.warning(f"A single chunk under header '{doc.metadata.get('Header 2', 'N/A')}' is too large ({doc_tokens} tokens) and will be sub-split.")
            sub_chunks = recursive_splitter.split_text(doc.page_content)

            for i, sub_chunk_content in enumerate(sub_chunks):
                new_doc = Document(
                    page_content=sub_chunk_content,
                    metadata=doc.metadata.copy()
                )
                new_doc.metadata['sub_chunk_index'] = i + 1
                processable_docs.append(new_doc)
        else:
            # Add chunks that are already a good size to the list for processing.
            processable_docs.append(doc)

    grouped = defaultdict(list)
    for doc in processable_docs:
        header1 = doc.metadata.get('Header 1', 'unknown')
        grouped[header1].append(doc)

    optimized_chunks = []
    for header1, group_docs in grouped.items():
        logger.info(f"Processing header group: '{header1}' with {len(group_docs)} chunks")

        current_tokens = 0
        current_content = []
        current_metadata = []

        for i, doc in enumerate(group_docs):
            doc_tokens = count_tokens(doc.page_content, tokenizer)
            logger.debug(f"  Chunk {i+1}: {doc_tokens} tokens")

            if current_tokens > 0 and (current_tokens + doc_tokens > max_tokens):
                optimized_chunks.append(Document(
                    page_content="\n\n".join(current_content),
                    metadata={
                        "Header 1": header1,
                        "merged_from": current_metadata.copy(),
                        "token_count": current_tokens,
                        "chunk_count": len(current_content)
                    }
                ))
                logger.debug(f"  Created chunk with {current_tokens} tokens from {len(current_content)} sections")

                current_content = [doc.page_content]
                current_metadata = [doc.metadata.get("Header 2", "UNKNOWN")]
                current_tokens = doc_tokens
            else:
                current_content.append(doc.page_content)
                current_metadata.append(doc.metadata.get("Header 2", "UNKNOWN"))
                current_tokens += doc_tokens

        if current_content:
            optimized_chunks.append(Document(
                page_content="\n\n".join(current_content),
                metadata={
                    "Header 1": header1,
                    "merged_from": current_metadata.copy(),
                    "token_count": current_tokens,
                    "chunk_count": len(current_content)
                }
            ))
            logger.debug(f"  Created final chunk with {current_tokens} tokens from {len(current_content)} sections")

    return optimized_chunks

def count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens in text using tokenizer or heuristic fallback."""
    if tokenizer:
        return len(tokenizer.encode(text, truncation=False))
    else:
        return len(text) // 4

def split_markdown(markdown_path: str, headers_to_split_on: Optional[List[Tuple[str, str]]] = None) -> List[Document]:
    """Convenience function to split markdown file."""
    splitter = MarkdownSplitter()
    return splitter.split_markdown(markdown_path, headers_to_split_on)