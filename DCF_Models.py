import numpy as np
import pandas as pd

def calculate_dcf(
    revenue_growth_rates,
    ebit_margins,
    tax_rate,
    nwc_percent,
    capex_percent,
    discount_rate,
    terminal_growth_rate,
    initial_revenue,
    years=5
):
    """
    Calculate company valuation using DCF method.
    
    Parameters:
    revenue_growth_rates: List of yearly growth rates
    ebit_margins: List of EBIT margins
    tax_rate: Corporate tax rate
    nwc_percent: Net working capital as % of revenue
    capex_percent: Capital expenditure as % of revenue
    discount_rate: WACC or required rate of return
    terminal_growth_rate: Long-term growth rate
    initial_revenue: Starting revenue
    years: Forecast period
    """
    
    # Initialize arrays
    revenues = np.zeros(years)
    ebit = np.zeros(years)
    nopat = np.zeros(years)
    nwc = np.zeros(years + 1)
    capex = np.zeros(years)
    fcf = np.zeros(years)
    
    # Project revenues
    revenues[0] = initial_revenue * (1 + revenue_growth_rates[0])
    for i in range(1, years):
        revenues[i] = revenues[i-1] * (1 + revenue_growth_rates[min(i, len(revenue_growth_rates)-1)])
    
    # Calculate EBIT
    for i in range(years):
        ebit[i] = revenues[i] * ebit_margins[min(i, len(ebit_margins)-1)]
    
    # Calculate NOPAT
    nopat = ebit * (1 - tax_rate)
    
    # Calculate NWC changes
    nwc[0] = initial_revenue * nwc_percent
    for i in range(years):
        nwc[i+1] = revenues[i] * nwc_percent
    nwc_changes = np.diff(nwc)
    
    # Calculate CAPEX
    capex = revenues * capex_percent
    
    # Calculate free cash flows
    fcf = nopat - nwc_changes - capex
    
    # Calculate terminal value
    terminal_fcf = fcf[-1] * (1 + terminal_growth_rate)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
    
    # Discount cash flows
    discount_factors = np.array([(1 + discount_rate) ** -i for i in range(1, years + 1)])
    pv_fcf = fcf * discount_factors
    pv_terminal_value = terminal_value * discount_factors[-1]
    
    # Calculate enterprise value
    enterprise_value = np.sum(pv_fcf) + pv_terminal_value
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Revenue': revenues,
        'EBIT': ebit,
        'NOPAT': nopat,
        'NWC Change': nwc_changes,
        'CAPEX': capex,
        'FCF': fcf,
        'PV of FCF': pv_fcf
    })
    
    return {
        'enterprise_value': enterprise_value,
        'terminal_value': terminal_value,
        'summary': summary
    }

###### USAGE ######
model_inputs = {
    'revenue_growth_rates': [0.15, 0.12, 0.10, 0.08, 0.06],
    'ebit_margins': [0.25, 0.26, 0.27, 0.27, 0.28],
    'tax_rate': 0.25,
    'nwc_percent': 0.12,
    'capex_percent': 0.08,
    'discount_rate': 0.10,
    'terminal_growth_rate': 0.02,
    'initial_revenue': 1000000
}

valuation = calculate_dcf(**model_inputs)
print(f"Enterprise Value: ${valuation['enterprise_value']:,.2f}")

    
def sensitivity_analysis(base_inputs, variables, ranges):
    """
    Perform sensitivity analysis on DCF model by varying input parameters.
    
    Parameters:
    base_inputs: Dict of base case inputs for DCF model
    variables: List of variables to analyze
    ranges: Dict of percentage changes to apply to each variable
    """
    results = {}
    base_value = calculate_dcf(**base_inputs)['enterprise_value']
    
    for var in variables:
        var_results = []
        base_var_value = base_inputs[var]
        
        for pct_change in ranges[var]:
            if isinstance(base_var_value, list):
                # Handle list inputs (growth rates, margins)
                modified_value = [x * (1 + pct_change) for x in base_var_value]
            else:
                # Handle scalar inputs
                modified_value = base_var_value * (1 + pct_change)
            
            # Create modified inputs
            modified_inputs = base_inputs.copy()
            modified_inputs[var] = modified_value
            
            # Calculate new value
            new_value = calculate_dcf(**modified_inputs)['enterprise_value']
            pct_change_in_value = (new_value - base_value) / base_value
            
            var_results.append({
                'change_in_input': f"{pct_change:+.1%}",
                'enterprise_value': new_value,
                'value_change': f"{pct_change_in_value:+.1%}"
            })
            
        results[var] = pd.DataFrame(var_results)
    
    return results

###### USAGE ######
base_inputs = {
    'revenue_growth_rates': [0.15, 0.12, 0.10, 0.08, 0.06],
    'ebit_margins': [0.25, 0.26, 0.27, 0.27, 0.28],
    'tax_rate': 0.25,
    'nwc_percent': 0.12,
    'capex_percent': 0.08,
    'discount_rate': 0.10,
    'terminal_growth_rate': 0.02,
    'initial_revenue': 1000000
}

# Define variables to analyze and their ranges
variables_to_analyze = ['discount_rate', 'terminal_growth_rate']
sensitivity_ranges = {
    'discount_rate': np.array([-0.20, -0.10, 0, 0.10, 0.20]),
    'terminal_growth_rate': np.array([-0.50, -0.25, 0, 0.25, 0.50])
}

# Run sensitivity analysis
sensitivity_results = sensitivity_analysis(base_inputs, variables_to_analyze, sensitivity_ranges)

# Print results
for var, results_df in sensitivity_results.items():
    print(f"\nSensitivity Analysis for {var}:")
    print(results_df)

def reverse_dcf(
    current_price,
    shares_outstanding,
    net_debt,
    initial_revenue,
    ebit_margins,
    tax_rate,
    nwc_percent,
    capex_percent,
    discount_rate,
    terminal_growth_rate,
    years=5,
    iteration_range=(0, 0.5, 0.001)
):
    """
    Calculate implied growth rate given current stock price.
    
    Parameters:
    current_price: Current stock price
    shares_outstanding: Number of shares outstanding
    net_debt: Total debt minus cash
    Other parameters same as calculate_dcf()
    iteration_range: Tuple of (min_growth, max_growth, step) for iteration
    """
    market_cap = current_price * shares_outstanding
    target_ev = market_cap + net_debt
    
    min_growth, max_growth, step = iteration_range
    growth_rates = np.arange(min_growth, max_growth, step)
    
    closest_growth = None
    min_difference = float('inf')
    
    for growth in growth_rates:
        growth_array = [growth] * years
        inputs = {
            'revenue_growth_rates': growth_array,
            'ebit_margins': ebit_margins,
            'tax_rate': tax_rate,
            'nwc_percent': nwc_percent,
            'capex_percent': capex_percent,
            'discount_rate': discount_rate,
            'terminal_growth_rate': terminal_growth_rate,
            'initial_revenue': initial_revenue,
            'years': years
        }
        
        calculated_ev = calculate_dcf(**inputs)['enterprise_value']
        difference = abs(calculated_ev - target_ev)
        
        if difference < min_difference:
            min_difference = difference
            closest_growth = growth
    
    return closest_growth

def reverse_dcf_sensitivity(
    base_inputs,
    variables,
    ranges
):
    """
    Sensitivity analysis for reverse DCF.
    
    Parameters:
    base_inputs: Dict containing all reverse DCF inputs
    variables: List of variables to analyze
    ranges: Dict of percentage changes to apply
    """
    results = {}
    base_growth = reverse_dcf(**base_inputs)
    
    for var in variables:
        var_results = []
        base_var_value = base_inputs[var]
        
        for pct_change in ranges[var]:
            modified_inputs = base_inputs.copy()
            modified_inputs[var] = base_var_value * (1 + pct_change)
            
            implied_growth = reverse_dcf(**modified_inputs)
            
            var_results.append({
                'change_in_input': f"{pct_change:+.1%}",
                'implied_growth': f"{implied_growth:.1%}",
                'growth_delta': f"{(implied_growth - base_growth):+.1%}"
            })
        
        results[var] = pd.DataFrame(var_results)
    
    return results

###### USAGE ######
reverse_inputs = {
    'current_price': 50,
    'shares_outstanding': 1000000,
    'net_debt': 500000,
    'initial_revenue': 1000000,
    'ebit_margins': [0.25, 0.26, 0.27, 0.27, 0.28],
    'tax_rate': 0.25,
    'nwc_percent': 0.12,
    'capex_percent': 0.08,
    'discount_rate': 0.10,
    'terminal_growth_rate': 0.02
}

# Find implied growth rate
implied_growth = reverse_dcf(**reverse_inputs)
print(f"Implied Growth Rate: {implied_growth:.1%}")

# Run sensitivity analysis
sensitivity_vars = ['discount_rate', 'terminal_growth_rate']
sensitivity_ranges = {
    'discount_rate': [-0.20, -0.10, 0, 0.10, 0.20],
    'terminal_growth_rate': [-0.50, -0.25, 0, 0.25, 0.50]
}

sensitivity_results = reverse_dcf_sensitivity(
    reverse_inputs,
    sensitivity_vars,
    sensitivity_ranges
)
