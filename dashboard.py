"""
Streamlit dashboard for the stock analysis system
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from reporting.report_generator import FreeReportGenerator
from portfolio.optimizer import FreePortfolioOptimizer

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("📈 Free Stock Analysis System")
st.markdown("---")

# Initialize modules
report_generator = FreeReportGenerator()
portfolio_optimizer = FreePortfolioOptimizer()

# Sidebar
st.sidebar.header("Analysis Options")
option = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Single Stock Analysis", "Portfolio Optimization", "Compare Stocks"]
)

if option == "Single Stock Analysis":
    st.header("Single Stock Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
    
    with col2:
        if st.button("Analyze Stock", type="primary"):
            with st.spinner(f"Analyzing {ticker}..."):
                report = report_generator.generate_stock_report(ticker)
                
                if report:
                    # Display results
                    rec = report['investment_recommendation']
                    
                    st.success(f"Analysis Complete!")
                    
                    # Metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Recommendation", rec['recommendation'])
                    
                    with col2:
                        st.metric("Confidence", rec['confidence'])
                    
                    with col3:
                        st.metric("Score", f"{rec['score']:.1f}/100")
                    
                    with col4:
                        val = report['sections']['valuation']['summary']
                        st.metric("Upside", f"{val.get('average_upside', 0):.1f}%")
                    
                    # Tabs for detailed analysis
                    tab1, tab2, tab3, tab4 = st.tabs(["Valuation", "Financial Health", "Technical", "Risks"])
                    
                    with tab1:
                        st.subheader("Valuation Analysis")
                        val_data = report['sections']['valuation']['summary']
                        
                        if val_data:
                            df_valuation = pd.DataFrame(val_data['individual_valuations'])
                            st.dataframe(df_valuation)
                            
                            st.metric("Current Price", f"${val_data['current_price']:.2f}")
                            st.metric("Fair Value", f"${val_data['average_fair_value']:.2f}")
                            st.metric("Upside/Downside", f"{val_data['average_upside']:.1f}%")
                    
                    with tab2:
                        st.subheader("Financial Health")
                        health = report['sections']['financial_health']
                        
                        st.metric("Health Score", f"{health['health_score']:.2f}/1.0")
                        st.metric("Health Status", health['health_status'])
                        
                        # Display ratios
                        if 'ratios' in health:
                            ratios_df = pd.DataFrame([health['ratios']]).T
                            ratios_df.columns = ['Value']
                            st.dataframe(ratios_df)
                    
                    with tab3:
                        st.subheader("Technical Analysis")
                        technical = report['sections']['technical_analysis']
                        
                        if technical:
                            st.write("Signals:")
                            for signal in technical.get('signals', []):
                                st.write(f"• {signal}")
                    
                    with tab4:
                        st.subheader("Risk Assessment")
                        risks = report['risk_assessment']
                        
                        st.metric("Overall Risk", risks['overall_risk'])
                        
                        for level in ['high', 'medium', 'low']:
                            if risks['risk_levels'][level]:
                                st.write(f"**{level.upper()} Risks:**")
                                for risk in risks['risk_levels'][level]:
                                    st.write(f"• {risk}")
                
                else:
                    st.error(f"Could not analyze {ticker}")

elif option == "Portfolio Optimization":
    st.header("Portfolio Optimization")
    
    tickers_input = st.text_input("Enter Tickers (comma-separated)", "AAPL,MSFT,GOOGL")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if st.button("Optimize Portfolio", type="primary"):
        if len(tickers) < 2:
            st.warning("Please enter at least 2 tickers")
        else:
            with st.spinner("Optimizing portfolio..."):
                result = portfolio_optimizer.optimize_portfolio(tickers)
                
                if result:
                    st.success("Portfolio Optimized!")
                    
                    # Display optimized portfolios
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Max Sharpe Portfolio")
                        max_sharpe = result['optimized_portfolios']['max_sharpe']
                        
                        st.metric("Expected Return", f"{max_sharpe['expected_return']:.1%}")
                        st.metric("Volatility", f"{max_sharpe['volatility']:.1%}")
                        st.metric("Sharpe Ratio", f"{max_sharpe['sharpe_ratio']:.2f}")
                        
                        # Display weights
                        weights_df = pd.DataFrame({
                            'Ticker': tickers,
                            'Weight': max_sharpe['weights']
                        })
                        st.dataframe(weights_df)
                    
                    with col2:
                        st.subheader("Min Volatility Portfolio")
                        min_vol = result['optimized_portfolios']['min_volatility']
                        
                        st.metric("Expected Return", f"{min_vol['expected_return']:.1%}")
                        st.metric("Volatility", f"{min_vol['volatility']:.1%}")
                        st.metric("Sharpe Ratio", f"{min_vol['sharpe_ratio']:.2f}")
                    
                    # Efficient frontier visualization
                    st.subheader("Efficient Frontier")
                    frontier_df = pd.DataFrame({
                        'Return': result['efficient_frontier']['returns'],
                        'Volatility': result['efficient_frontier']['volatilities'],
                        'Sharpe': result['efficient_frontier']['sharpe_ratios']
                    })
                    
                    st.scatter_chart(frontier_df, x='Volatility', y='Return', color='Sharpe')

elif option == "Compare Stocks":
    st.header("Compare Multiple Stocks")
    
    tickers_input = st.text_input("Enter Tickers to Compare (comma-separated)", "AAPL,MSFT,GOOGL,NVDA")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if st.button("Compare Stocks", type="primary"):
        if len(tickers) < 1:
            st.warning("Please enter at least 1 ticker")
        else:
            comparison_data = []
            
            for ticker in tickers:
                with st.spinner(f"Analyzing {ticker}..."):
                    report = report_generator.generate_stock_report(ticker)
                    
                    if report:
                        val = report['sections']['valuation']['summary']
                        health = report['sections']['financial_health']
                        rec = report['investment_recommendation']
                        
                        comparison_data.append({
                            'Ticker': ticker,
                            'Current Price': val.get('current_price', 0),
                            'Fair Value': val.get('average_fair_value', 0),
                            'Upside %': val.get('average_upside', 0),
                            'Health Score': health.get('health_score', 0),
                            'Recommendation': rec.get('recommendation', 'N/A'),
                            'Overall Score': rec.get('score', 0)
                        })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison)
                
                # Sort by score
                st.subheader("Ranked by Overall Score")
                df_sorted = df_comparison.sort_values('Overall Score', ascending=False)
                st.dataframe(df_sorted)

# Footer
st.markdown("---")
st.markdown("**Free Stock Analysis System** • Built with Python & Yahoo Finance API")
st.markdown("*Data may be delayed by 15 minutes*")