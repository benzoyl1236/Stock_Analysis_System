"""
Portfolio optimization using free methods
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class FreePortfolioOptimizer:
    def __init__(self, risk_free_rate=0.04):
        self.risk_free_rate = risk_free_rate
    
    def optimize_portfolio(self, tickers: list, initial_weights=None):
        """Basic portfolio optimization"""
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
            
            if data.empty:
                return {}
            
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Calculate mean returns and covariance
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Calculate portfolio metrics for equal weighting
            if initial_weights is None:
                initial_weights = np.array([1/len(tickers)] * len(tickers))
            
            portfolio_return = np.sum(mean_returns * initial_weights) * 252
            portfolio_volatility = np.sqrt(np.dot(initial_weights.T, np.dot(cov_matrix * 252, initial_weights)))
            portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Generate random portfolios for efficient frontier
            num_portfolios = 10000
            results = np.zeros((3, num_portfolios))
            weights_record = []
            
            for i in range(num_portfolios):
                weights = np.random.random(len(tickers))
                weights /= np.sum(weights)
                weights_record.append(weights)
                
                portfolio_ret = np.sum(mean_returns * weights) * 252
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                
                results[0,i] = portfolio_ret
                results[1,i] = portfolio_vol
                results[2,i] = (portfolio_ret - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            # Find optimal portfolios
            max_sharpe_idx = np.argmax(results[2])
            max_sharpe_return = results[0, max_sharpe_idx]
            max_sharpe_vol = results[1, max_sharpe_idx]
            max_sharpe_weights = weights_record[max_sharpe_idx]
            
            # Find minimum volatility portfolio
            min_vol_idx = np.argmin(results[1])
            min_vol_return = results[0, min_vol_idx]
            min_vol_vol = results[1, min_vol_idx]
            min_vol_weights = weights_record[min_vol_idx]
            
            # Calculate diversification metrics
            diversification_ratio = self._calculate_diversification_ratio(cov_matrix, max_sharpe_weights)
            
            # Calculate individual stock metrics
            stock_metrics = {}
            for i, ticker in enumerate(tickers):
                stock_metrics[ticker] = {
                    'weight_max_sharpe': max_sharpe_weights[i],
                    'weight_min_vol': min_vol_weights[i],
                    'annual_return': mean_returns[i] * 252,
                    'annual_volatility': np.sqrt(cov_matrix.iloc[i,i] * 252),
                    'sharpe_ratio': (mean_returns[i] * 252 - self.risk_free_rate) / np.sqrt(cov_matrix.iloc[i,i] * 252) 
                    if np.sqrt(cov_matrix.iloc[i,i] * 252) > 0 else 0
                }
            
            return {
                'tickers': tickers,
                'current_portfolio': {
                    'weights': initial_weights.tolist(),
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': portfolio_sharpe
                },
                'optimized_portfolios': {
                    'max_sharpe': {
                        'weights': max_sharpe_weights.tolist(),
                        'expected_return': max_sharpe_return,
                        'volatility': max_sharpe_vol,
                        'sharpe_ratio': results[2, max_sharpe_idx]
                    },
                    'min_volatility': {
                        'weights': min_vol_weights.tolist(),
                        'expected_return': min_vol_return,
                        'volatility': min_vol_vol,
                        'sharpe_ratio': results[2, min_vol_idx]
                    }
                },
                'stock_metrics': stock_metrics,
                'diversification_ratio': diversification_ratio,
                'efficient_frontier': {
                    'returns': results[0].tolist(),
                    'volatilities': results[1].tolist(),
                    'sharpe_ratios': results[2].tolist()
                }
            }
            
        except Exception as e:
            print(f"Error optimizing portfolio: {e}")
            return {}
    
    def _calculate_diversification_ratio(self, cov_matrix, weights):
        """Calculate portfolio diversification ratio"""
        weighted_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        average_vol = np.sqrt(np.mean(np.diag(cov_matrix)))
        
        return average_vol / weighted_vol if weighted_vol > 0 else 1
    
    def calculate_portfolio_metrics(self, portfolio: dict):
        """Calculate performance metrics for existing portfolio"""
        try:
            tickers = list(portfolio.keys())
            weights = list(portfolio.values())
            
            if not tickers:
                return {}
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
            
            if data.empty:
                return {}
            
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Portfolio returns
            portfolio_returns = returns.dot(weights)
            
            # Calculate metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = total_return * 252 / len(portfolio_returns)
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            peak = cumulative.expanding(min_periods=1).max()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()
            
            # Sortino ratio (only downside deviation)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': (portfolio_returns > 0).sum() / len(portfolio_returns) if len(portfolio_returns) > 0 else 0,
                'avg_win': portfolio_returns[portfolio_returns > 0].mean() if (portfolio_returns > 0).any() else 0,
                'avg_loss': portfolio_returns[portfolio_returns < 0].mean() if (portfolio_returns < 0).any() else 0,
                'profit_factor': abs(portfolio_returns[portfolio_returns > 0].sum() / portfolio_returns[portfolio_returns < 0].sum()) 
                if (portfolio_returns < 0).any() and portfolio_returns[portfolio_returns < 0].sum() != 0 else 0
            }
            
        except Exception as e:
            print(f"Error calculating portfolio metrics: {e}")
            return {}

if __name__ == "__main__":
    optimizer = FreePortfolioOptimizer()
    
    # Test portfolio optimization
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    result = optimizer.optimize_portfolio(test_tickers)
    
    print("Portfolio Optimization Results:")
    print(f"Max Sharpe Portfolio Sharpe Ratio: {result['optimized_portfolios']['max_sharpe']['sharpe_ratio']:.2f}")
    
    print("\nMax Sharpe Weights:")
    for ticker, weight in zip(test_tickers, result['optimized_portfolios']['max_sharpe']['weights']):
        print(f"  {ticker}: {weight:.2%}")