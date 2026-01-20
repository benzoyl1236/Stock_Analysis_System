"""
Free valuation models using Yahoo Finance data
"""
import yfinance as yf
import numpy as np
from datetime import datetime

class FreeValuationModels:
    def __init__(self):
        pass
    
    def dcf_valuation(self, ticker: str):
        """Simplified DCF using Yahoo Finance data"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get free cash flow
            cash_flow = stock.cashflow
            if cash_flow.empty:
                return {}
            
            # Extract FCF (simplified)
            fcf_items = cash_flow.loc[['Free Cash Flow', 'Operating Cash Flow']]
            if not fcf_items.empty:
                fcf = float(fcf_items.iloc[0, 0])
            else:
                fcf = info.get('freeCashflow', 0)
            
            if fcf <= 0:
                return {}
            
            # Simple DCF assumptions
            growth_rate = 0.05  # 5%
            terminal_growth = 0.025  # 2.5%
            discount_rate = 0.08  # 8%
            years = 5
            
            # Project FCF
            projected_fcfs = [fcf * ((1 + growth_rate) ** i) for i in range(1, years + 1)]
            
            # Terminal value
            terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            
            # Present values
            pv_fcfs = [fcf / ((1 + discount_rate) ** i) for i, fcf in enumerate(projected_fcfs, 1)]
            pv_terminal = terminal_value / ((1 + discount_rate) ** years)
            
            enterprise_value = sum(pv_fcfs) + pv_terminal
            
            # Get shares outstanding
            shares = info.get('sharesOutstanding', 1)
            
            # Fair value per share
            fair_value = enterprise_value / shares if shares > 0 else 0
            current_price = info.get('currentPrice', 0)
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'fair_value': fair_value,
                'upside': ((fair_value - current_price) / current_price * 100) if current_price > 0 else 0,
                'fcf': fcf,
                'growth_rate': growth_rate,
                'discount_rate': discount_rate,
                'is_undervalued': fair_value > current_price
            }
            
        except Exception as e:
            print(f"Error in DCF for {ticker}: {e}")
            return {}
    
    def comparable_analysis(self, ticker: str, peers: list):
        """Comparable company analysis"""
        try:
            target = yf.Ticker(ticker)
            target_info = target.info
            
            peer_data = []
            for peer in peers:
                peer_stock = yf.Ticker(peer)
                peer_info = peer_stock.info
                
                if peer_info:
                    peer_data.append({
                        'ticker': peer,
                        'pe': peer_info.get('trailingPE'),
                        'pb': peer_info.get('priceToBook'),
                        'ps': peer_info.get('priceToSalesTrailing12Months'),
                        'ev_to_ebitda': peer_info.get('enterpriseToEbitda'),
                        'market_cap': peer_info.get('marketCap')
                    })
            
            # Calculate averages
            valid_peers = [p for p in peer_data if all(v is not None for v in p.values())]
            
            if not valid_peers:
                return {}
            
            avg_pe = np.mean([p['pe'] for p in valid_peers if p['pe']])
            avg_pb = np.mean([p['pb'] for p in valid_peers if p['pb']])
            avg_ps = np.mean([p['ps'] for p in valid_peers if p['ps']])
            
            # Implied valuations
            eps = target_info.get('trailingEps', 0)
            book_value = target_info.get('bookValue', 0)
            sales_per_share = target_info.get('revenuePerShare', 0)
            
            implied_pe = eps * avg_pe if eps else 0
            implied_pb = book_value * avg_pb if book_value else 0
            implied_ps = sales_per_share * avg_ps if sales_per_share else 0
            
            # Average implied price
            implied_prices = [p for p in [implied_pe, implied_pb, implied_ps] if p > 0]
            avg_implied = np.mean(implied_prices) if implied_prices else 0
            
            current_price = target_info.get('currentPrice', 0)
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'implied_price': avg_implied,
                'implied_pe_price': implied_pe,
                'implied_pb_price': implied_pb,
                'implied_ps_price': implied_ps,
                'peer_count': len(valid_peers),
                'avg_pe': avg_pe,
                'avg_pb': avg_pb,
                'avg_ps': avg_ps,
                'upside': ((avg_implied - current_price) / current_price * 100) if current_price > 0 else 0
            }
            
        except Exception as e:
            print(f"Error in comparable analysis: {e}")
            return {}
    
    def graham_formula(self, ticker: str):
        """Benjamin Graham intrinsic value formula"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            eps = info.get('trailingEps', 0)
            book_value = info.get('bookValue', 0)
            
            if eps <= 0 or book_value <= 0:
                return {}
            
            # Graham Number: sqrt(22.5 * EPS * BVPS)
            graham_number = np.sqrt(22.5 * eps * book_value)
            
            # Graham Formula: V = EPS * (8.5 + 2g)
            # Assume growth rate from analyst estimates or historical
            growth_rate = min(info.get('earningsGrowth', 0.05), 0.15)  # Cap at 15%
            intrinsic_value = eps * (8.5 + (2 * growth_rate))
            
            current_price = info.get('currentPrice', 0)
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'graham_number': graham_number,
                'intrinsic_value': intrinsic_value,
                'margin_of_safety_price': intrinsic_value * 0.75,  # 25% margin of safety
                'eps': eps,
                'book_value': book_value,
                'growth_assumption': growth_rate,
                'is_undervalued_graham': current_price < graham_number,
                'is_undervalued_intrinsic': current_price < intrinsic_value * 0.75
            }
            
        except Exception as e:
            print(f"Error in Graham formula: {e}")
            return {}

if __name__ == "__main__":
    valuation = FreeValuationModels()
    
    # Test DCF
    dcf = valuation.dcf_valuation("AAPL")
    print("DCF Analysis:", dcf)
    
    # Test Comparable
    comps = valuation.comparable_analysis("AAPL", ["MSFT", "GOOGL", "NVDA"])
    print("\nComparable Analysis:", comps)
    
    # Test Graham
    graham = valuation.graham_formula("AAPL")
    print("\nGraham Analysis:", graham)