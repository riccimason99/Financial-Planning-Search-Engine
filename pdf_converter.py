"""
PDF to Text Converter for IR Search Engine

Extracts text from PDF files and saves as markdown for indexing.
"""

import os
import sys
from pathlib import Path

try:
    import fitz
except ImportError:
    print("Error: PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)


def pdf_to_text(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
    """
    doc = fitz.open(pdf_path)
    text_parts = []
    
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        if text.strip():
            text_parts.append(f"--- Page {page_num} ---\n{text}")
    
    doc.close()
    return "\n\n".join(text_parts)


def convert_pdfs_to_markdown(pdf_directory: str, output_directory: str) -> int:
    """
    Convert all PDFs in a directory to markdown files.
    
    Args:
        pdf_directory: Directory containing PDF files
        output_directory: Directory to save markdown files
        
    Returns:
        Number of files converted
    """
    pdf_dir = Path(pdf_directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return 0
    
    print(f"Found {len(pdf_files)} PDF files")
    converted = 0
    
    for pdf_path in pdf_files:
        print(f"Converting {pdf_path.name}...", end=" ")
        
        try:
            text = pdf_to_text(str(pdf_path))
            
            md_filename = pdf_path.stem + ".md"
            md_path = output_dir / md_filename
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# {pdf_path.stem}\n\n")
                f.write(text)
            
            print(f"✓ → {md_filename}")
            converted += 1
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n✓ Converted {converted}/{len(pdf_files)} PDFs")
    return converted


def main():
    """Command-line interface for PDF conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert PDFs to Markdown for IR search engine"
    )
    parser.add_argument(
        "pdf_dir",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "-o", "--output",
        default="./documents",
        help="Output directory for markdown files (default: ./documents)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_dir):
        print(f"Error: Directory not found: {args.pdf_dir}")
        sys.exit(1)
    
    convert_pdfs_to_markdown(args.pdf_dir, args.output)


if __name__ == "__main__":
    main()
