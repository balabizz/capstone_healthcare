"""PDF document loading and extraction."""

from pathlib import Path
from typing import List
from pypdf import PdfReader


class PDFLoader:
    """Load and extract text from PDF documents."""

    @staticmethod
    def load_pages(data, max_pages=200, max_characters=1000000):
        """Extract uploaded PDF bytes while preserving 1-based page numbers."""
        from io import BytesIO
        if not data.startswith(b'%PDF-'):
            raise ValueError('The file is not a PDF.')
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise ValueError('Encrypted PDFs are not supported. Provide an unencrypted reference PDF.')
            if len(reader.pages) > max_pages:
                raise ValueError(f'PDF exceeds the {max_pages}-page limit.')
            pages, total = [], 0
            for number, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or '').strip()
                total += len(text)
                if total > max_characters:
                    raise ValueError('PDF extracted text exceeds the ingestion limit.')
                pages.append({'page': number, 'text': text})
            if not any(p['text'] for p in pages):
                raise ValueError('No readable text found. Scanned PDFs need OCR before ingestion.')
            return pages
        except ValueError:
            raise
        except Exception as error:
            raise ValueError('Could not read this PDF. Check that it is valid and unencrypted.') from error

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
