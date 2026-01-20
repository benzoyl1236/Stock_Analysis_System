"""
Free market data collection using Yahoo Finance
"""
import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
import time
from database.schema import db_manager, PriceData, Company

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FreeMarketDataCollector:
    def __init__(self):
        pass
    
    def fetch_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetch stock data using Yahoo Finance (free)"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                logger.warning(f"No data for {ticker}")
                return pd.DataFrame()
            
            df['Ticker'] = ticker
            df.reset_index(inplace=True)
            logger.info(f"Fetched {len(df)} records for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_company_info(self, ticker: str) -> dict:
        """Get company information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'US'),
                'market_cap': info.get('marketCap', 0),
                'description': info.get('longBusinessSummary', '')
            }
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {e}")
            return {}
    
    def save_price_data(self, df: pd.DataFrame, ticker: str):
        """Save price data to database"""
        if df.empty:
            return
        
        session = db_manager.get_session()
        try:
            for _, row in df.iterrows():
                price_data = PriceData(
                    ticker=ticker,
                    date=row['Date'],
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    volume=row['Volume'],
                    adjusted_close=row.get('Adj Close', row['Close'])
                )
                session.add(price_data)
            
            session.commit()
            logger.info(f"Saved {len(df)} records for {ticker}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving data: {e}")
        finally:
            session.close()
    
    def update_company(self, ticker: str):
        """Update company info in database"""
        info = self.fetch_company_info(ticker)
        if not info:
            return
        
        session = db_manager.get_session()
        try:
            company = session.query(Company).filter_by(ticker=ticker).first()
            
            if company:
                company.name = info['name']
                company.sector = info['sector']
                company.industry = info['industry']
                company.country = info['country']
                company.market_cap = info['market_cap']
            else:
                company = Company(
                    ticker=ticker,
                    name=info['name'],
                    sector=info['sector'],
                    industry=info['industry'],
                    country=info['country'],
                    market_cap=info['market_cap']
                )
                session.add(company)
            
            session.commit()
            logger.info(f"Updated company {ticker}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating company: {e}")
        finally:
            session.close()

if __name__ == "__main__":
    collector = FreeMarketDataCollector()
    
    # Test with a few stocks
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    for ticker in test_tickers:
        df = collector.fetch_stock_data(ticker, period="1mo")
        if not df.empty:
            collector.save_price_data(df, ticker)
        
        collector.update_company(ticker)
        time.sleep(1)  # Rate limiting