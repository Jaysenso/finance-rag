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
    "GOOG",
    "LLY",     # 10 - $919B
    "WMT",     # 12 - $817B
    "V",       # 13 - $637B
    "ORCL",    # 14 - $635B
    "XOM",     # 15 - $503B
    "MA",      # 16 - $490B
    "JNJ",     # 17 - $472B
    "NFLX",    # 18 - $471B
    "PLTR",    # 19 - $415B
    "ABBV",    # 20 - $411B
    
    "COST",    # 21 - $409B
    "AMD",     # 22 - $402B
    "BAC",     # 23 - $384B
    "HD",      # 24 - $361B
    "PG",      # 25 - $345B
    "GE",      # 26 - $322B
    "CVX",     # 27 - $315B
    "CSCO",    # 28 - $307B
    "KO",      # 29 - $306B
    "UNH",     # 30 - $292B
    
    "IBM",     # 31 - $286B
    "MU",      # 32 - $277B
    "WFC",     # 33 - $267B
    "MS",      # 34 - $260B
    "CAT",     # 35 - $260B
    "AXP",     # 36 - $246B
    "PM",      # 37 - $242B
    "TMUS",    # 38 - $242B
    "GS",      # 39 - $237B
    "RTX",     # 40 - $235B
    
    "CRM",     # 41 - $232B
    "MRK",     # 42 - $231B
    "ABT",     # 43 - $227B
    "MCD",     # 44 - $219B
    "TMO",     # 45 - $217B
    "PEP",     # 46 - $199B
    "ISRG",    # 47 - $195B
    "UBER",    # 48 - $190B
    "DIS",     # 49 - $190B
    "QCOM"     # 51 - $186B 
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
                date = filing['filedAt']
                
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