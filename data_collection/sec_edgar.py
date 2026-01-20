"""
SEC EDGAR filings downloader (free)
"""
import requests
from datetime import datetime
import re
import json

class SECEdgarCollector:
    BASE_URL = "https://www.sec.gov"
    
    def __init__(self, user_agent="Your Name your.email@gmail.com"):
        self.headers = {'User-Agent': user_agent}
    
    def get_company_filings(self, ticker: str, filing_type="10-K", limit=5):
        """Get recent filings for a company"""
        try:
            # First get CIK from ticker
            cik = self._get_cik(ticker)
            if not cik:
                return []
            
            # Get filings
            url = f"{self.BASE_URL}/cgi-bin/browse-edgar"
            params = {
                'action': 'getcompany',
                'CIK': cik,
                'type': filing_type,
                'count': limit
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            
            if response.status_code == 200:
                # Parse HTML response
                import re
                filings = []
                pattern = r'<td nowrap="nowrap">(\d{4}-\d{2}-\d{2})</td>'
                dates = re.findall(pattern, response.text)
                
                for i, date_str in enumerate(dates[:limit]):
                    filings.append({
                        'ticker': ticker,
                        'filing_type': filing_type,
                        'filing_date': date_str,
                        'sequence': i + 1
                    })
                
                return filings
            
            return []
            
        except Exception as e:
            print(f"Error getting filings: {e}")
            return []
    
    def _get_cik(self, ticker: str):
        """Get CIK number from ticker"""
        try:
            # Download SEC company tickers JSON
            url = f"{self.BASE_URL}/files/company_tickers.json"
            response = requests.get(url, headers=self.headers)
            data = response.json()
            
            # Search for ticker
            for cik, company_info in data.items():
                if company_info.get('ticker') == ticker.upper():
                    return str(cik).zfill(10)
            
            return None
            
        except Exception as e:
            print(f"Error getting CIK: {e}")
            return None
    
    def download_filing_text(self, ticker: str, filing_date: str):
        """Download filing as text (simplified)"""
        try:
            cik = self._get_cik(ticker)
            if not cik:
                return ""
            
            # Construct filing URL (simplified)
            year = filing_date[:4]
            url = f"{self.BASE_URL}/Archives/edgar/data/{cik}/{filing_date.replace('-', '')}/"
            
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.text[:5000]  # First 5000 chars
            
            return ""
            
        except Exception as e:
            print(f"Error downloading filing: {e}")
            return ""

if __name__ == "__main__":
    collector = SECEdgarCollector()
    filings = collector.get_company_filings("AAPL", "10-K", 2)
    print(f"Found {len(filings)} filings for AAPL")
    for filing in filings:
        print(f"Date: {filing['filing_date']}")