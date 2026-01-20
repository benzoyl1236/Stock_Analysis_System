"""
Main pipeline to run the entire analysis system
"""
import os
import sys
from datetime import datetime
import json
from reporting.report_generator import FreeReportGenerator
from portfolio.optimizer import FreePortfolioOptimizer
from data_collection.market_data import FreeMarketDataCollector

def main():
    """Main function to run the analysis pipeline"""
    print("=" * 60)
    print("STOCK ANALYSIS SYSTEM - FREE VERSION")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs("reports", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Initialize modules
    collector = FreeMarketDataCollector()
    reporter = FreeReportGenerator()
    optimizer = FreePortfolioOptimizer()
    
    while True:
        print("\nOptions:")
        print("1. Analyze single stock")
        print("2. Analyze portfolio")
        print("3. Update stock data")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            # Single stock analysis
            ticker = input("Enter stock ticker (e.g., AAPL): ").upper().strip()
            print(f"\nAnalyzing {ticker}...")
            
            # Generate report
            report = reporter.generate_stock_report(ticker)
            
            if report:
                # Save report
                reporter.save_report(report)
                
                # Display summary
                rec = report['investment_recommendation']
                print(f"\n✓ Analysis complete!")
                print(f"Recommendation: {rec['recommendation']}")
                print(f"Confidence: {rec['confidence']}")
                print(f"Score: {rec['score']:.1f}/100")
            else:
                print(f"✗ Could not analyze {ticker}")
        
        elif choice == "2":
            # Portfolio analysis
            tickers_input = input("Enter portfolio tickers (comma-separated, e.g., AAPL,MSFT,GOOGL): ")
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            
            if len(tickers) < 2:
                print("Please enter at least 2 tickers")
                continue
            
            print(f"\nAnalyzing portfolio: {', '.join(tickers)}")
            
            # Optimize portfolio
            result = optimizer.optimize_portfolio(tickers)
            
            if result:
                print("\n✓ Portfolio Optimization Results:")
                print(f"Max Sharpe Ratio Portfolio:")
                print(f"  Expected Return: {result['optimized_portfolios']['max_sharpe']['expected_return']:.1%}")
                print(f"  Volatility: {result['optimized_portfolios']['max_sharpe']['volatility']:.1%}")
                print(f"  Sharpe Ratio: {result['optimized_portfolios']['max_sharpe']['sharpe_ratio']:.2f}")
                
                print("\n  Weights:")
                for t, w in zip(tickers, result['optimized_portfolios']['max_sharpe']['weights']):
                    print(f"    {t}: {w:.1%}")
                
                # Save portfolio results
                with open(f"reports/portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                    json.dump(result, f, indent=2)
            else:
                print("✗ Could not optimize portfolio")
        
        elif choice == "3":
            # Update stock data
            ticker = input("Enter ticker to update data: ").upper().strip()
            print(f"Updating data for {ticker}...")
            
            # Fetch and save data
            df = collector.fetch_stock_data(ticker, period="1y")
            if not df.empty:
                collector.save_price_data(df, ticker)
                collector.update_company(ticker)
                print(f"✓ Updated data for {ticker}")
            else:
                print(f"✗ Could not fetch data for {ticker}")
        
        elif choice == "4":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()