"""
Database schema for stock data
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime

Base = declarative_base()

class Company(Base):
    """Company information"""
    __tablename__ = 'companies'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), unique=True, index=True)
    name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(200))
    country = Column(String(50))
    market_cap = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)

class PriceData(Base):
    """Daily price data"""
    __tablename__ = 'price_data'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), index=True)
    date = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    adjusted_close = Column(Float)

class AnalysisResult(Base):
    """Analysis results"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), index=True)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    analysis_type = Column(String(50))
    result_data = Column(Text)  # JSON results
    score = Column(Float)
    recommendation = Column(String(20))

# Database initialization
def init_database():
    db_url = 'sqlite:///database/stock_data.db'
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

class DatabaseManager:
    def __init__(self):
        self.db_url = 'sqlite:///database/stock_data.db'
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        return self.Session()

db_manager = DatabaseManager()