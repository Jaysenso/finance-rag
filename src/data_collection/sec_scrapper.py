from sec_api import QueryApi, PdfGeneratorApi
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("SEC_API_KEY")
queryApi = QueryApi(api_key=api_key)
pdfApi = PdfGeneratorApi(api_key=api_key)

COMPANIES = [ 
    "AAPL",
    "AMZN",
    "AVGO",
    "BRK.B",
    "JPM",
    "META",
    "MSFT",
    "NVDA",
    "TSLA",
    ]

def download_pdfs_for_company(ticker):
    """Download SEC filings as PDFs"""
    
    print(f"\n{'='*60}")
    print(f"Downloading PDFs for {ticker}")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir = Path("data/pdf") / ticker
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filing_types = ["10-K", "10-Q", "8-K"]
    
    for filing_type in filing_types:
        print(f"\nSearching for {filing_type} filings...")
        
        # Search for filings
        query = {
            "query": f'ticker:{ticker} AND formType:"{filing_type}" AND filedAt:[2023-01-01 TO 2026-12-31]',
            "from": "0",
            "size": "5",  
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        filings = queryApi.get_filings(query)
        
        for filing in filings['filings']:
            try:
                url = filing['linkToFilingDetails']
                accession = filing['accessionNo'].replace('/', '-')
                
                print(f"  Converting {accession}...")
                
                # Generate PDF
                pdf_content = pdfApi.get_pdf(url)
                
                # Save PDF
                filename = f"{ticker}_{filing_type}_{accession}.pdf"
                file_path = output_dir / filing_type
                file_path.mkdir(exist_ok=True)
                
                with open(file_path / filename, "wb") as f:
                    f.write(pdf_content)
                
                print(f"Saved: {filename}")
            
                time.sleep(1) 
                
            except Exception as e:
                print(f"Error: {e}")
                continue


def main():
    """Download PDFs for all companies"""
    
    for idx, ticker in enumerate(COMPANIES, 1):
        print(f"\n[{idx}/{len(COMPANIES)}] Processing {ticker}...")
        download_pdfs_for_company(ticker)
        time.sleep(0.2)
    
    print("\n" + "="*60)
    print("PDF Download Complete!")
    print("="*60)


if __name__ == "__main__":
    main()