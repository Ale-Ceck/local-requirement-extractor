import json
import concurrent.futures
from typing import Optional
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter

from pdf_processing.pdf_to_markdown import convert_pdf_to_markdown
from llm_integration.ollama_client import get_client
from llm_integration.prompt_templates import get_prompt
from data_models.requirement import Requirement, RequirementList
from utils.logging_config import setup_logger
from utils.markdown_splitter import split_markdown

logger = setup_logger(__name__)

class RequirementExtractor:
    """Extract requirements from PDF documents using LLM processing."""

    def __init__(self, model_name: str = "llama3:latest", max_workers: int = 5):
        """Initialize the requirement extractor.
        
        Args:
            model_name: Name of the Ollama model to use for extraction
            max_workers: Maximum number of concurrent workers for parallel processing
        """
        self.model_name = model_name
        self.max_workers = max_workers
        self.ollama_client = get_client()

    def extract_requirements_from_pdf(self, pdf_path: str, markdown_dir: Optional[str] = None) ->  RequirementList:
        """
        Extract requirements from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            markdown_dir: Optional directory for markdown output

        Returns:
            List of extracted Requirement objects

        """
        try:
            logger.info(f"Starting requirement extraction from PDF: {pdf_path}")

            requirement_schema = RequirementList.model_json_schema()

            #Convert PDF to markdown
            markdown_path = convert_pdf_to_markdown(pdf_path, markdown_dir)
            logger.info(f"PDF converted to markdown:{markdown_path}")

#            with open(markdown_path,'r') as f:
#                markdown_doc = f.read()
            #Chunk the markdown file
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2")
                ]
#            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers= False)
#            docs  = markdown_splitter.split_text(markdown_doc)
            docs  = split_markdown(markdown_path=markdown_path, headers_to_split_on=headers_to_split_on)
            requirements = RequirementList([])

            print(docs)
####
            # Process chunks in parallel
            #all_requirements = self.process_chunks_parallel(docs, requirement_schema)
            #requirements = RequirementList(all_requirements)
                
            logger.info(f"Successfully extracted {len(requirements.root)} requirements")
            return requirements
        except Exception as e:
            logger.error(f"Error extracting requirements from PDF: {e}")
            return []

    def extract_requirements_from_markdown(self, md_path: str) ->  RequirementList:
        """
        Extract requirements from a Markdown file.

        Args:
        md_path: path to the Markdown file that need to be processed.

        Returns:
        List of extracted Requirement objects


        """
        try:
            logger.info(f"Starting requirement extraction from Markdown: {md_path}")

            requirement_schema = RequirementList.model_json_schema()

            #Chunk the markdown file
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2")
                ]
            docs  = split_markdown(markdown_path=md_path, headers_to_split_on=headers_to_split_on)
            requirements = RequirementList([])

            # Process chunks in parallel
            all_requirements = self.process_chunks_parallel(docs, requirement_schema)
            requirements = RequirementList(all_requirements)
                
            logger.info(f"Successfully extracted {len(requirements.root)} requirements")
            return requirements
        except Exception as e:
            logger.error(f"Error extracting requirements from markdown: {e}")
            raise

    def process_single_chunk(self, doc, requirement_schema) -> RequirementList:
        """
        Process a single document chunk to extract requirements.
        
        Args:
            doc: Document chunk from langchain splitter
            requirement_schema: JSON schema for requirements
            
        Returns:
            RequirementList with extracted requirements from this chunk
        """
        try:
            # Prepare prompt
            prompt = get_prompt("requirement_extraction", doc, requirement_schema, include_few_shot=True) 
            
            # Get structured response from LLM
            response = self.ollama_client.get_structured_response(prompt, model_name=self.model_name)
            print(response)
            
            if response is None:
                logger.error("Failed to get response from LLM for chunk")
                return RequirementList([])
            
            # Parse the LLM response into a RequirementList
            return self.parse_llm_response(response)
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return RequirementList([])

    def process_chunks_parallel(self, docs, requirement_schema) -> list:
        """
        Process document chunks in parallel to extract requirements.
        
        Args:
            docs: List of document chunks from langchain splitter
            requirement_schema: JSON schema for requirements
            
        Returns:
            List of Requirement objects from all chunks
        """
        all_requirements = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(self.process_single_chunk, doc, requirement_schema): i 
                for i, doc in enumerate(docs)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    chunk_requirements = future.result()
                    logger.info(f"Processed chunk {chunk_index+1}/{len(docs)}: found {len(chunk_requirements)} requirements")
                    all_requirements.extend(chunk_requirements.root)
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_index+1}: {e}")
        
        return all_requirements

    def parse_llm_response(self, json_response: str) -> RequirementList:
        """
        Parse LLM JSON response into validated requirements
        
        Args:
            json_response: JSON string from LLM
            
        Returns:
            RequirementsList object
            
        Raises:
            ValidationError: If requirements are invalid
            JSONDecodeError: If JSON is malformed
        """
        try:
            # Parse JSON
            data = json.loads(json_response)
            
            # Handle different response formats
            if isinstance(data, dict):
                # If LLM returns {"requirements": [...]}
                if "requirements" in data:
                    data = data["requirements"]
                # If LLM returns {"requirement": {...}} (single)
                elif "requirement" in data:
                    data = [data["requirement"]]
                else:
                    # If dict has code/description directly
                    if "code" in data and "description" in data:
                        data = [data]
                    else:
                        data = []
            
            # Ensure it's a list
            if not isinstance(data, list):
                data = []
                
            # Validate with Pydantic
            return RequirementList(data)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from LLM: {e}")

def extract_requirements_from_pdf(pdf_path: str, model_name: str = "llama3:latest", markdown_dir: Optional[str] = "data/markdown_output", max_workers: int = 5) -> RequirementList:
    """Convenience function to extract requirements from a PDF file.
      
    Args:
        pdf_path: Path to the PDF file
        model_name: Name of the Ollama model to use
        markdown_dir: Optional directory for markdown output
        max_workers: Maximum number of concurrent workers for parallel processing
       
    Returns:
        List of extracted Requirement objects
    """
    extractor = RequirementExtractor(model_name=model_name, max_workers=max_workers)
    return extractor.extract_requirements_from_pdf(pdf_path, markdown_dir)

def extract_requirements_from_markdown(md_path: str, model_name: str = "gpt-oss:latest", max_workers: int = 5) -> RequirementList:
    """Convenience function to extract requirements from a Markdown file.
      
    Args:
        md_path: Path to the Markdown file
        model_name: Name of the Ollama model to use
        max_workers: Maximum number of concurrent workers for parallel processing
       
    Returns:
        List of extracted Requirement objects
    """
    extractor = RequirementExtractor(model_name=model_name, max_workers=max_workers)
    return extractor.extract_requirements_from_markdown(md_path)

#  DeepSeek-R1-Distill-Llama-8B-Q4_K_M:latest