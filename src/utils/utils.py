"""
Utility functions for extracting metadata from filenames and file paths.
"""
import re
from pathlib import Path
from typing import Dict


def extract_metadata_from_filename(file_path: str) -> Dict[str, str]:
    """
    Extract company, document_type, and filing_date from filename.
    
    Expected format: COMPANY_DOCTYPE_ACCESSION.pdf
    Example: AAPL_10-K_0000320193-25-000079.pdf
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary with keys: file_path, company, document_type, filing_date
        
    Raises:
        ValueError: If filename doesn't match expected format
    """
    filename = Path(file_path).stem
    
    # Pattern: COMPANY_DOCTYPE_ACCESSION
    # Accession format: XXXXXXXXXX-YY-ZZZZZZ where YY is the year
    pattern = r'^([A-Z]+)_([^_]+)_\d{10}-(\d{2})-\d{6}$'
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(
            f"Filename '{filename}' doesn't match expected format: "
            "COMPANY_DOCTYPE_ACCESSION.pdf (e.g., AAPL_10-K_0000320193-25-000079.pdf)"
        )
    
    company, document_type, year_suffix = match.groups()
    
    # Convert YY to full year (assuming 20YY for now)
    year = f"20{year_suffix}"
    
    # For 10-K and 10-Q, filing date is typically end of fiscal year/quarter
    # This is a simplified assumption - you may want to adjust based on actual filing dates
    if document_type == "10-K":
        filing_date = f"{year}-12-31"  # Assume fiscal year end
    elif document_type == "10-Q":
        filing_date = f"{year}-09-30"  # Assume Q3 end as example
    else:
        filing_date = f"{year}-12-31"  # Default assumption
    
    return {
        "file_path": file_path,
        "company": company,
        "document_type": document_type,
        "filing_date": filing_date
    }
