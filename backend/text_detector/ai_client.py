"""
AI Client for Text Analysis
Handles communication with Groq API using Llama 4 Scout for document extraction.
"""
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class AIDocumentExtractor:
    """
    AI-powered document text extraction using Llama 4 Scout vision model.
    """
    
    def __init__(self):
        """Initialize Groq client with text analysis API key."""
        api_key = os.getenv('GROQ_TEXT_API_KEY')
        if not api_key:
            raise ValueError("GROQ_TEXT_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.model = os.getenv('GROQ_TEXT_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')
        print(f"[AIDocumentExtractor] Initialized with model: {self.model}")
    
    def extract_text(self, text_prompt: str, context: str = "") -> str:
        """
        Extract and process text using AI.
        
        Args:
            text_prompt: The main prompt for extraction
            context: Additional context (e.g., filename, format)
            
        Returns:
            Extracted text as string
        """
        try:
            full_prompt = f"{context}\n\n{text_prompt}" if context else text_prompt
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert document text extractor. Extract all text from the provided content accurately, preserving structure and formatting where possible."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_completion_tokens=4096,
                top_p=1,
                stream=False
            )
            
            extracted_text = completion.choices[0].message.content
            return extracted_text.strip()
            
        except Exception as e:
            print(f"[AIDocumentExtractor] Error extracting text: {e}")
            raise
    
    def extract_from_pages(self, page_texts: list) -> str:
        """
        Extract and combine text from multiple pages.
        
        Args:
            page_texts: List of text content from each page
            
        Returns:
            Combined extracted text
        """
        if not page_texts:
            return ""
        
        # For simple text, just combine directly
        # AI can be used for cleanup/enhancement if needed
        combined = "\n\n".join(page_texts)
        
        # Optional: Use AI to clean up and structure the combined text
        try:
            prompt = f"""Clean up and structure the following document text, preserving all content and meaning:

{combined}

Output only the cleaned text without any additional commentary."""
            
            return self.extract_text(prompt, context="Document cleanup task")
        except Exception as e:
            print(f"[AIDocumentExtractor] AI cleanup failed, returning raw text: {e}")
            return combined


if __name__ == "__main__":
    # Quick test
    extractor = AIDocumentExtractor()
    test_text = extractor.extract_text("Extract this: Hello, world!")
    print(f"Test result: {test_text}")
