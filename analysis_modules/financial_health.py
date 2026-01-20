"""
Financial health analysis using free data
"""
import yfinance as yf
import numpy as np
from datetime import datetime

class FreeFinancialHealthAnalyzer:
    def __init__(self):
        self.ratios_benchmark = {
            'current_ratio': {'min': 1.5, 'good': 2.0},
            'debt_to_equity': {'max': 0.5, 'good': 0.3},
            'interest_coverage': {'min': 3.0, 'good': 5.0},
            'roe': {'min': 0.15, 'good': 0.20},
            'net_margin': {'min': 0.10, 'good': 0.15}
        }
    
    def analyze_company(self, ticker: str):
        """Comprehensive financial health analysis"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get financial statements
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            cash_flow = stock.cash_flow
            
            if balance_sheet.empty or income_stmt.empty:
                return {}
            
            # Extract latest data
            latest_date = balance_sheet.columns[0]
            
            # Balance sheet items
            total_assets = balance_sheet.get('Total Assets', {}).get(latest_date, 0)
            current_assets = balance_sheet.get('Current Assets', {}).get(latest_date, 0) or \
                           balance_sheet.get('Total Current Assets', {}).get(latest_date, 0)
            current_liabilities = balance_sheet.get('Current Liabilities', {}).get(latest_date, 0) or \
                                balance_sheet.get('Total Current Liabilities', {}).get(latest_date, 0)
            total_liabilities = balance_sheet.get('Total Liabilities', {}).get(latest_date, 0)
            total_equity = balance_sheet.get('Total Equity', {}).get(latest_date, 0) or \
                         balance_sheet.get("Stockholders' Equity", {}).get(latest_date, 0)
            
            # Income statement items
            revenue = income_stmt.get('Total Revenue', {}).get(latest_date, 0) or \
                     income_stmt.get('Revenue', {}).get(latest_date, 0)
            net_income = income_stmt.get('Net Income', {}).get(latest_date, 0)
            operating_income = income_stmt.get('Operating Income', {}).get(latest_date, 0)
            interest_expense = abs(income_stmt.get('Interest Expense', {}).get(latest_date, 0) or 0)
            
            # Cash flow items
            operating_cash_flow = cash_flow.get('Operating Cash Flow', {}).get(latest_date, 0) or \
                                cash_flow.get('Cash Flow From Continuing Operating Activities', {}).get(latest_date, 0)
            capex = abs(cash_flow.get('Capital Expenditure', {}).get(latest_date, 0) or \
                       cash_flow.get('Capital Expenditures', {}).get(latest_date, 0) or 0)
            
            # Calculate ratios
            ratios = {}
            
            # Liquidity ratios
            if current_liabilities > 0:
                ratios['current_ratio'] = current_assets / current_liabilities
            
            # Leverage ratios
            if total_equity > 0:
                ratios['debt_to_equity'] = total_liabilities / total_equity
                ratios['debt_to_assets'] = total_liabilities / total_assets if total_assets > 0 else 0
            
            # Profitability ratios
            if revenue > 0:
                ratios['gross_margin'] = info.get('grossMargins', 0)
                ratios['operating_margin'] = operating_income / revenue if revenue > 0 else 0
                ratios['net_margin'] = net_income / revenue if revenue > 0 else 0
            
            if total_equity > 0:
                ratios['return_on_equity'] = net_income / total_equity
            
            if total_assets > 0:
                ratios['return_on_assets'] = net_income / total_assets
            
            # Coverage ratios
            if interest_expense > 0:
                ratios['interest_coverage'] = operating_income / interest_expense
            
            # Cash flow ratios
            if capex > 0:
                ratios['capex_coverage'] = operating_cash_flow / capex
            
            free_cash_flow = operating_cash_flow - capex
            ratios['free_cash_flow'] = free_cash_flow
            ratios['fcf_margin'] = free_cash_flow / revenue if revenue > 0 else 0
            
            # Assess health
            health_score = self._calculate_health_score(ratios)
            
            # Identify risks
            risks = self._identify_risks(ratios, info)
            
            return {
                'ticker': ticker,
                'company_name': info.get('longName', ticker),
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'ratios': ratios,
                'health_score': health_score['score'],
                'health_status': health_score['status'],
                'free_cash_flow': free_cash_flow,
                'revenue': revenue,
                'net_income': net_income,
                'total_equity': total_equity,
                'total_assets': total_assets,
                'risks': risks,
                'recommendations': self._generate_recommendations(ratios, health_score)
            }
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            return {}
    
    def _calculate_health_score(self, ratios: dict):
        """Calculate overall financial health score"""
        score = 0
        max_score = 0
        
        for ratio_name, benchmark in self.ratios_benchmark.items():
            if ratio_name in ratios:
                ratio_value = ratios[ratio_name]
                max_score += 1
                
                if 'min' in benchmark:
                    if ratio_value >= benchmark['good']:
                        score += 1
                    elif ratio_value >= benchmark['min']:
                        score += 0.5
                elif 'max' in benchmark:
                    if ratio_value <= benchmark['good']:
                        score += 1
                    elif ratio_value <= benchmark['max']:
                        score += 0.5
        
        final_score = score / max_score if max_score > 0 else 0
        
        if final_score >= 0.8:
            status = 'Excellent'
        elif final_score >= 0.6:
            status = 'Good'
        elif final_score >= 0.4:
            status = 'Fair'
        else:
            status = 'Poor'
        
        return {'score': final_score, 'status': status}
    
    def _identify_risks(self, ratios: dict, info: dict):
        """Identify financial risks"""
        risks = []
        
        # Liquidity risk
        if ratios.get('current_ratio', 0) < 1.0:
            risks.append('High liquidity risk (current ratio < 1.0)')
        elif ratios.get('current_ratio', 0) < 1.5:
            risks.append('Moderate liquidity risk (current ratio < 1.5)')
        
        # Leverage risk
        if ratios.get('debt_to_equity', 0) > 1.0:
            risks.append('High leverage risk (debt/equity > 1.0)')
        elif ratios.get('debt_to_equity', 0) > 0.5:
            risks.append('Moderate leverage risk (debt/equity > 0.5)')
        
        # Profitability risk
        if ratios.get('net_margin', 0) < 0.05:
            risks.append('Low profitability (net margin < 5%)')
        
        # Cash flow risk
        if ratios.get('free_cash_flow', 0) < 0:
            risks.append('Negative free cash flow')
        
        # Growth risk
        revenue_growth = info.get('revenueGrowth', 0)
        if revenue_growth < 0:
            risks.append('Declining revenue')
        
        return risks
    
    def _generate_recommendations(self, ratios: dict, health_score: dict):
        """Generate recommendations"""
        recommendations = []
        
        if health_score['score'] < 0.4:
            recommendations.append("Avoid - poor financial health")
        elif health_score['score'] < 0.6:
            recommendations.append("Caution - monitor closely")
        
        if ratios.get('current_ratio', 0) < 1.5:
            recommendations.append("Improve working capital management")
        
        if ratios.get('debt_to_equity', 0) > 0.5:
            recommendations.append("Reduce debt levels")
        
        if ratios.get('free_cash_flow', 0) > 0:
            recommendations.append("Strong cash flow generation")
        
        if len(recommendations) == 0:
            recommendations.append("Financially healthy company")
        
        return recommendations

if __name__ == "__main__":
    analyzer = FreeFinancialHealthAnalyzer()
    
    # Test analysis
    result = analyzer.analyze_company("AAPL")
    
    print(f"Financial Health Analysis for {result.get('company_name')}")
    print(f"Health Score: {result.get('health_score'):.2f} ({result.get('health_status')})")
    
    print("\nKey Ratios:")
    for key, value in result.get('ratios', {}).items():
        print(f"  {key}: {value:.2f}")
    
    print("\nRisks:")
    for risk in result.get('risks', []):
        print(f"  • {risk}")
    
    print("\nRecommendations:")
    for rec in result.get('recommendations', []):
        print(f"  • {rec}")