"""
Technical analysis indicators
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class FreeTechnicalAnalyzer:
    def __init__(self):
        pass
    
    def analyze_stock(self, ticker: str, period: str = "6mo"):
        """Comprehensive technical analysis"""
        try:
            # Get price data
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                return {}
            
            # Calculate indicators
            indicators = {}
            
            # Moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(df['Close'])
            
            # MACD
            macd, signal = self._calculate_macd(df['Close'])
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_histogram'] = macd - signal
            
            # Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(df['Close'])
            indicators['bb_upper'] = bb_upper
            indicators['bb_lower'] = bb_lower
            
            # Volume analysis
            indicators['volume_sma'] = df['Volume'].rolling(window=20).mean().iloc[-1]
            indicators['current_volume'] = df['Volume'].iloc[-1]
            indicators['volume_ratio'] = indicators['current_volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
            
            # Price momentum
            current_price = df['Close'].iloc[-1]
            indicators['price_change_1d'] = (current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
            indicators['price_change_5d'] = (current_price - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100
            indicators['price_change_1mo'] = (current_price - df['Close'].iloc[-22]) / df['Close'].iloc[-22] * 100
            
            # Support and resistance
            indicators['resistance'] = df['High'].rolling(window=20).max().iloc[-1]
            indicators['support'] = df['Low'].rolling(window=20).min().iloc[-1]
            
            # Generate signals
            signals = self._generate_signals(df, indicators)
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'indicators': indicators,
                'signals': signals,
                'price_data': {
                    'high': df['High'].iloc[-1],
                    'low': df['Low'].iloc[-1],
                    'open': df['Open'].iloc[-1],
                    'volume': df['Volume'].iloc[-1]
                },
                'moving_averages': {
                    'sma_20': df['SMA_20'].iloc[-1],
                    'sma_50': df['SMA_50'].iloc[-1],
                    'sma_200': df['SMA_200'].iloc[-1]
                }
            }
            
        except Exception as e:
            print(f"Error in technical analysis for {ticker}: {e}")
            return {}
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        return macd.iloc[-1] if not macd.empty else 0, signal_line.iloc[-1] if not signal_line.empty else 0
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band.iloc[-1] if not upper_band.empty else 0, lower_band.iloc[-1] if not lower_band.empty else 0
    
    def _generate_signals(self, df, indicators):
        """Generate trading signals"""
        signals = []
        
        current_price = df['Close'].iloc[-1]
        
        # Moving average signals
        if current_price > df['SMA_20'].iloc[-1]:
            signals.append("Price above 20-day MA (bullish)")
        else:
            signals.append("Price below 20-day MA (bearish)")
        
        if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
            signals.append("All MAs in bullish alignment")
        
        # RSI signals
        rsi = indicators['rsi']
        if rsi > 70:
            signals.append("RSI > 70 (overbought)")
        elif rsi < 30:
            signals.append("RSI < 30 (oversold)")
        else:
            signals.append("RSI neutral")
        
        # MACD signals
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        if macd > macd_signal:
            signals.append("MACD bullish")
        else:
            signals.append("MACD bearish")
        
        # Volume signals
        if indicators['volume_ratio'] > 1.5:
            signals.append("High volume (significant move)")
        
        # Support/Resistance
        if current_price > indicators['resistance'] * 0.98:
            signals.append("Near resistance level")
        elif current_price < indicators['support'] * 1.02:
            signals.append("Near support level")
        
        # Overall bias
        bullish_count = sum(1 for s in signals if 'bullish' in s.lower())
        bearish_count = sum(1 for s in signals if 'bearish' in s.lower())
        
        if bullish_count > bearish_count:
            overall = "Bullish"
        elif bearish_count > bullish_count:
            overall = "Bearish"
        else:
            overall = "Neutral"
        
        signals.append(f"Overall bias: {overall}")
        
        return signals

if __name__ == "__main__":
    analyzer = FreeTechnicalAnalyzer()
    
    # Test analysis
    result = analyzer.analyze_stock("AAPL")
    
    print(f"Technical Analysis for {result['ticker']}")
    print(f"Current Price: ${result['current_price']:.2f}")
    
    print("\nIndicators:")
    for key, value in result['indicators'].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f}")
    
    print("\nSignals:")
    for signal in result['signals']:
        print(f"  • {signal}")