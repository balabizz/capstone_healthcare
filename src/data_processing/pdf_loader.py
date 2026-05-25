"""PDF document loading and extraction."""

from pathlib import Path
from typing import List
from pypdf import PdfReader


class PDFLoader:
    """Load and extract text from PDF documents."""

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """
        Load text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content from the PDF
        """
        try:
            reader = PdfReader(file_path)
            text = ""
            for page_num, page in enumerate(reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error loading PDF {file_path}: {str(e)}")

    @staticmethod
    def load_multiple_pdfs(directory: str) -> dict:
        """
        Load multiple PDFs from a directory.
        
        Args:
            directory: Path to directory containing PDF files
            
        Returns:
            Dictionary with filename as key and extracted text as value
        """
        pdf_contents = {}
        pdf_dir = Path(directory)
        
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory {directory} not found")
        
        for pdf_file in pdf_dir.glob("*.pdf"):
            try:
                pdf_contents[pdf_file.name] = PDFLoader.load_pdf(str(pdf_file))
            except Exception as e:
                print(f"Warning: Failed to load {pdf_file.name}: {str(e)}")
        
        return pdf_contents
