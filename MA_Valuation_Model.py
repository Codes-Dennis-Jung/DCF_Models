import pandas as pd
import numpy as np

class MAValuationModel:
    def __init__(self, target_financials, comparable_companies, synergies=None):
        """
        Initialize M&A valuation model with target company financials and comparable companies data
        
        Parameters:
        target_financials (dict): Financial metrics of target company
        comparable_companies (list): List of dicts containing comparable companies' metrics
        synergies (dict): Expected cost and revenue synergies (optional)
        """
        self.target = target_financials
        self.comps = comparable_companies
        self.synergies = synergies or {}
        
    def dcf_valuation(self, wacc, growth_rate, forecast_years=5):
        """Calculate DCF valuation using FCF projections"""
        fcf = self.target['fcf']
        
        # Project future cash flows
        projected_fcf = []
        for year in range(forecast_years):
            fcf *= (1 + growth_rate)
            projected_fcf.append(fcf)
            
        # Calculate terminal value using Gordon Growth model
        terminal_value = projected_fcf[-1] * (1 + growth_rate) / (wacc - growth_rate)
        
        # Discount cash flows and terminal value
        pv_fcf = sum([cf / (1 + wacc)**(i+1) for i, cf in enumerate(projected_fcf)])
        pv_terminal = terminal_value / (1 + wacc)**forecast_years
        
        enterprise_value = pv_fcf + pv_terminal
        equity_value = enterprise_value - self.target['net_debt']
        
        return {'enterprise_value': enterprise_value, 'equity_value': equity_value}
    
    def trading_multiples(self):
        """Calculate valuation using comparable company multiples"""
        # Calculate median multiples from comparable companies
        ev_ebitda = np.median([comp['ev'] / comp['ebitda'] for comp in self.comps])
        price_earnings = np.median([comp['price'] / comp['earnings'] for comp in self.comps])
        
        # Apply multiples to target metrics
        ev_from_ebitda = self.target['ebitda'] * ev_ebitda
        equity_from_pe = self.target['earnings'] * price_earnings
        
        return {
            'ev_ebitda_multiple': ev_ebitda,
            'pe_multiple': price_earnings,
            'ev_from_ebitda': ev_from_ebitda,
            'equity_from_pe': equity_from_pe
        }
    
    def precedent_transactions(self, transactions):
        """Calculate valuation based on similar M&A transactions"""
        # Calculate median transaction multiples
        transaction_ev_ebitda = np.median([t['ev'] / t['ebitda'] for t in transactions])
        transaction_ev_revenue = np.median([t['ev'] / t['revenue'] for t in transactions])
        
        # Apply to target company
        ev_from_ebitda = self.target['ebitda'] * transaction_ev_ebitda
        ev_from_revenue = self.target['revenue'] * transaction_ev_revenue
        
        return {
            'transaction_ev_ebitda': transaction_ev_ebitda,
            'transaction_ev_revenue': transaction_ev_revenue,
            'ev_from_ebitda': ev_from_ebitda,
            'ev_from_revenue': ev_from_revenue
        }
    
    def synergy_value(self, tax_rate=0.25, wacc=0.1):
        """Calculate present value of expected synergies"""
        if not self.synergies:
            return 0
            
        annual_cost_savings = self.synergies.get('cost_savings', 0)
        annual_revenue_synergies = self.synergies.get('revenue_synergies', 0)
        implementation_costs = self.synergies.get('implementation_costs', 0)
        
        # Calculate after-tax synergies
        after_tax_synergies = (annual_cost_savings + annual_revenue_synergies) * (1 - tax_rate)
        
        # Assume perpetual synergies and discount to present value
        pv_synergies = after_tax_synergies / wacc
        
        # Subtract one-time implementation costs
        net_synergy_value = pv_synergies - implementation_costs
        
        return net_synergy_value
    
    def football_field(self):
        """Generate valuation ranges for different methods"""
        valuations = {
            'DCF': self.dcf_valuation(0.1, 0.02),
            'Trading Multiples': self.trading_multiples(),
            'Transaction Multiples': self.precedent_transactions([]),
            'Synergy Value': self.synergy_value()
        }
        
        return pd.DataFrame(valuations)

# Example usage
target_data = {
    'fcf': 100,
    'ebitda': 120,
    'earnings': 80,
    'revenue': 500,
    'net_debt': 200
}

comps_data = [
    {'ev': 1000, 'ebitda': 100, 'price': 800, 'earnings': 60},
    {'ev': 1500, 'ebitda': 140, 'price': 1200, 'earnings': 90},
    {'ev': 2000, 'ebitda': 180, 'price': 1600, 'earnings': 120}
]

synergies_data = {
    'cost_savings': 20,
    'revenue_synergies': 10,
    'implementation_costs': 50
}

# Initialize and run model
model = MAValuationModel(target_data, comps_data, synergies_data)