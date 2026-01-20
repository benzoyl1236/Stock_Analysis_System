"""
Generate analysis reports
"""
import json
from datetime import datetime
from analysis_modules.valuation import FreeValuationModels
from analysis_modules.financial_health import FreeFinancialHealthAnalyzer
from analysis_modules.technical_analysis import FreeTechnicalAnalyzer
from portfolio.optimizer import FreePortfolioOptimizer

class FreeReportGenerator:
    def __init__(self):
        self.valuation = FreeValuationModels()
        self.health = FreeFinancialHealthAnalyzer()
        self.technical = FreeTechnicalAnalyzer()
        self.portfolio = FreePortfolioOptimizer()
    
    def generate_stock_report(self, ticker: str):
        """Generate comprehensive stock analysis report"""
        try:
            report = {
                'ticker': ticker,
                'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sections': {}
            }
            
            # 1. Valuation Analysis
            dcf = self.valuation.dcf_valuation(ticker)
            comps = self.valuation.comparable_analysis(ticker, ["MSFT", "GOOGL", "AMZN", "NVDA", "META"])
            graham = self.valuation.graham_formula(ticker)
            
            report['sections']['valuation'] = {
                'dcf': dcf,
                'comparable_analysis': comps,
                'graham_analysis': graham,
                'summary': self._summarize_valuation(dcf, comps, graham)
            }
            
            # 2. Financial Health Analysis
            health = self.health.analyze_company(ticker)
            report['sections']['financial_health'] = health
            
            # 3. Technical Analysis
            technical = self.technical.analyze_stock(ticker)
            report['sections']['technical_analysis'] = technical
            
            # 4. Investment Recommendation
            recommendation = self._generate_recommendation(
                report['sections']['valuation']['summary'],
                health,
                technical
            )
            
            report['investment_recommendation'] = recommendation
            
            # 5. Risk Assessment
            report['risk_assessment'] = self._assess_risks(health, technical, dcf)
            
            return report
            
        except Exception as e:
            print(f"Error generating report for {ticker}: {e}")
            return {}
    
    def _summarize_valuation(self, dcf: dict, comps: dict, graham: dict) -> dict:
        """Summarize valuation analysis"""
        valuations = []
        
        if dcf.get('fair_value', 0) > 0:
            valuations.append(('DCF', dcf['fair_value'], dcf['upside']))
        
        if comps.get('implied_price', 0) > 0:
            valuations.append(('Comparables', comps['implied_price'], comps['upside']))
        
        if graham.get('intrinsic_value', 0) > 0:
            valuations.append(('Graham', graham['intrinsic_value'], 
                             ((graham['intrinsic_value'] * 0.75 - graham.get('current_price', 0)) / 
                              graham.get('current_price', 1) * 100) if graham.get('current_price', 0) > 0 else 0))
        
        if not valuations:
            return {}
        
        avg_fair_value = sum(v[1] for v in valuations) / len(valuations)
        avg_upside = sum(v[2] for v in valuations) / len(valuations)
        
        current_price = dcf.get('current_price', 0) or comps.get('current_price', 0) or graham.get('current_price', 0)
        
        return {
            'current_price': current_price,
            'average_fair_value': avg_fair_value,
            'average_upside': avg_upside,
            'is_undervalued': avg_fair_value > current_price,
            'valuation_range': {
                'min': min(v[1] for v in valuations),
                'max': max(v[1] for v in valuations)
            },
            'individual_valuations': [
                {'method': v[0], 'value': v[1], 'upside': v[2]} for v in valuations
            ]
        }
    
    def _generate_recommendation(self, valuation_summary: dict, health: dict, technical: dict) -> dict:
        """Generate investment recommendation"""
        score = 0
        max_score = 0
        
        # Valuation score (40% weight)
        if valuation_summary.get('is_undervalued', False):
            score += 40
        elif valuation_summary.get('average_upside', 0) > 10:
            score += 30
        elif valuation_summary.get('average_upside', 0) > 0:
            score += 20
        else:
            score += 10
        max_score += 40
        
        # Financial health score (40% weight)
        health_score = health.get('health_score', 0)
        score += health_score * 40
        max_score += 40
        
        # Technical score (20% weight)
        technical_signals = technical.get('signals', [])
        bullish_count = sum(1 for s in technical_signals if 'bullish' in s.lower())
        bearish_count = sum(1 for s in technical_signals if 'bearish' in s.lower())
        
        if bullish_count > bearish_count:
            score += 20
        elif bearish_count > bullish_count:
            score += 5
        else:
            score += 12
        max_score += 20
        
        # Final score
        final_score = score / max_score * 100
        
        # Recommendation based on score
        if final_score >= 80:
            recommendation = "STRONG BUY"
            confidence = "High"
        elif final_score >= 65:
            recommendation = "BUY"
            confidence = "Medium"
        elif final_score >= 50:
            recommendation = "HOLD"
            confidence = "Low"
        elif final_score >= 35:
            recommendation = "SELL"
            confidence = "Medium"
        else:
            recommendation = "STRONG SELL"
            confidence = "High"
        
        return {
            'score': final_score,
            'recommendation': recommendation,
            'confidence': confidence,
            'rationale': self._generate_rationale(valuation_summary, health, technical)
        }
    
    def _generate_rationale(self, valuation_summary: dict, health: dict, technical: dict) -> str:
        """Generate rationale for recommendation"""
        rationales = []
        
        # Valuation rationale
        if valuation_summary.get('is_undervalued', False):
            rationales.append("Stock appears undervalued based on multiple valuation methods.")
        else:
            rationales.append("Stock appears fairly valued or overvalued.")
        
        # Financial health rationale
        health_status = health.get('health_status', '')
        if health_status == 'Excellent':
            rationales.append("Excellent financial health with strong fundamentals.")
        elif health_status == 'Good':
            rationales.append("Good financial health with solid fundamentals.")
        elif health_status == 'Fair':
            rationales.append("Fair financial health, monitor closely.")
        else:
            rationales.append("Poor financial health, significant risks present.")
        
        # Technical rationale
        signals = technical.get('signals', [])
        if any('bullish' in s.lower() for s in signals):
            rationales.append("Technical indicators show bullish momentum.")
        elif any('bearish' in s.lower() for s in signals):
            rationales.append("Technical indicators show bearish momentum.")
        
        return " ".join(rationales)
    
    def _assess_risks(self, health: dict, technical: dict, dcf: dict) -> dict:
        """Assess overall risks"""
        risks = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        # Financial risks
        health_risks = health.get('risks', [])
        for risk in health_risks:
            if 'High' in risk or 'Negative' in risk:
                risks['high'].append(risk)
            elif 'Moderate' in risk:
                risks['medium'].append(risk)
            else:
                risks['low'].append(risk)
        
        # Valuation risks
        if dcf.get('fcf', 0) <= 0:
            risks['high'].append("Negative or zero free cash flow")
        
        if dcf.get('growth_rate', 0) > 0.15:
            risks['medium'].append("High growth assumptions in valuation")
        
        # Technical risks
        rsi = technical.get('indicators', {}).get('rsi', 50)
        if rsi > 80:
            risks['high'].append("Extremely overbought (RSI > 80)")
        elif rsi > 70:
            risks['medium'].append("Overbought (RSI > 70)")
        
        # Count risks
        risk_counts = {level: len(risks[level]) for level in risks}
        total_risks = sum(risk_counts.values())
        
        return {
            'risk_levels': risks,
            'risk_counts': risk_counts,
            'total_risks': total_risks,
            'overall_risk': 'High' if risk_counts['high'] > 2 else 'Medium' if total_risks > 3 else 'Low'
        }
    
    def save_report(self, report: dict, filename: str = None):
        """Save report to file"""
        if not filename:
            ticker = report.get('ticker', 'unknown')
            filename = f"reports/{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {filename}")
        
        # Also generate a readable summary
        self._generate_summary_text(report, filename.replace('.json', '.txt'))
    
    def _generate_summary_text(self, report: dict, filename: str):
        """Generate readable text summary"""
        with open(filename, 'w') as f:
            f.write(f"STOCK ANALYSIS REPORT\n")
            f.write(f"=====================\n\n")
            f.write(f"Ticker: {report.get('ticker')}\n")
            f.write(f"Generated: {report.get('generated_date')}\n\n")
            
            # Recommendation
            rec = report.get('investment_recommendation', {})
            f.write(f"RECOMMENDATION: {rec.get('recommendation', 'N/A')}\n")
            f.write(f"Confidence: {rec.get('confidence', 'N/A')}\n")
            f.write(f"Score: {rec.get('score', 0):.1f}/100\n\n")
            f.write(f"Rationale: {rec.get('rationale', 'N/A')}\n\n")
            
            # Valuation Summary
            val = report.get('sections', {}).get('valuation', {}).get('summary', {})
            f.write(f"VALUATION SUMMARY\n")
            f.write(f"-----------------\n")
            f.write(f"Current Price: ${val.get('current_price', 0):.2f}\n")
            f.write(f"Fair Value: ${val.get('average_fair_value', 0):.2f}\n")
            f.write(f"Upside: {val.get('average_upside', 0):.1f}%\n")
            f.write(f"Undervalued: {'Yes' if val.get('is_undervalued', False) else 'No'}\n\n")
            
            # Financial Health
            health = report.get('sections', {}).get('financial_health', {})
            f.write(f"FINANCIAL HEALTH\n")
            f.write(f"----------------\n")
            f.write(f"Score: {health.get('health_score', 0):.2f}/1.0\n")
            f.write(f"Status: {health.get('health_status', 'N/A')}\n\n")
            
            # Risks
            risks = report.get('risk_assessment', {})
            f.write(f"RISK ASSESSMENT\n")
            f.write(f"---------------\n")
            f.write(f"Overall Risk: {risks.get('overall_risk', 'N/A')}\n")
            f.write(f"High Risks: {risks.get('risk_counts', {}).get('high', 0)}\n")
            f.write(f"Medium Risks: {risks.get('risk_counts', {}).get('medium', 0)}\n")
            f.write(f"Low Risks: {risks.get('risk_counts', {}).get('low', 0)}\n")

if __name__ == "__main__":
    generator = FreeReportGenerator()
    
    # Generate report for Apple
    report = generator.generate_stock_report("AAPL")
    
    if report:
        generator.save_report(report)
        print("\n=== SUMMARY ===")
        rec = report['investment_recommendation']
        print(f"Recommendation: {rec['recommendation']} ({rec['confidence']})")
        print(f"Score: {rec['score']:.1f}/100")
        
        # Print key metrics
        val_summary = report['sections']['valuation']['summary']
        print(f"\nValuation:")
        print(f"  Current: ${val_summary['current_price']:.2f}")
        print(f"  Fair Value: ${val_summary['average_fair_value']:.2f}")
        print(f"  Upside: {val_summary['average_upside']:.1f}%")
        
        health = report['sections']['financial_health']
        print(f"\nFinancial Health: {health['health_status']} ({health['health_score']:.2f})")
        
        risks = report['risk_assessment']
        print(f"\nRisk Assessment: {risks['overall_risk']}")