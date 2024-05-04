# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'

from IPython.display import display

# DSC 80 preferred styles
pio.templates["dsc80"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc80"


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def clean_loans(loans):
    loans['issue_d'] = pd.to_datetime(loans['issue_d'])
    loans['term'] = loans['term'].str.extract('(\d+)').astype(int)
    loans ['emp_title'] = loans['emp_title'].str.lower().str.strip()
    loans.loc[loans['emp_title'] == 'rn', 'emp_title'] = 'registered nurse'
    loans['term_end'] = loans.apply(lambda row : row['issue_d'] + pd.DateOffset(months=row['term']),axis=1)
    return loans


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def correlations(df, pairs):
    output = []
    for i in pairs:
        correlation = df[i[0]].corr(df[i[1]])
        output.append(correlation)

    series = pd.Series (output, index = ['r_'+pair[0]+"_"+pair[1] for pair in pairs] )
    return series
    
# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(loans):
    loans_copy = loans.copy()
    bins = [580,670,740,800,850]
    labels = ['[580, 670)','[670, 740)','[740, 800)','[800, 850)']

    term_colors = {'36 months': 'purple', '60 months': 'gold'}
    loans_copy = loans_copy.sort_values(by='fico_range_low')
    loans_copy['credit_score_bins'] = pd.cut(loans_copy['fico_range_low'], bins = bins, labels=labels, include_lowest=True).astype(str)
    fig = px.box(loans_copy, x='credit_score_bins', y='int_rate', color='term',
                 category_orders={'term': ['36 months', '60 months']},
                 color_discrete_map=term_colors,
                 labels={'int_rate': 'Interest Rate (%)', 'term': 'Loan Length (Months)', 'credit_score_bins': 'Credit Score Range'},
                 title='Interest Rate vs. Credit Score',
                 )
    
    # Show the plot
    return fig



# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def ps_test(loans, N):
    loans['has_ps']  = loans['desc'].notna()
    observed_diff = loans.groupby('has_ps')['int_rate'].mean().diff().iloc[-1]

    permu_diff = np.zeros(N)

    for i in range(N):
        permutation = np.random.permutation(loans['has_ps'])
        shuffled_data = loans.assign(has_ps=permutation)

        permutation_diff=shuffled_data.groupby('has_ps')['int_rate'].mean().diff().iloc[-1]
        permu_diff[i] = permutation_diff
    
    p_value = (observed_diff<=permu_diff).mean()
    return p_value


def missingness_mechanism():
    return 2
    
def argument_for_nmar():
    '''
    Put your justification here in this multi-line string.
    Make sure to return your string!
    '''
    string = 'P-vlaue is less than 0.05, Reject Ho. The significant of this is that the interst rates given to applications with personal statements are larger than people with no personal statments on average. Since the interest rate is higher with ps, then people would not like to submit it. Therefore will be nmar'
    return string


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

'annual_inc = gross income'
def tax_owed(income, brackets):
    # Initialize tax owed
    tax_owed = 0
    
    # Iterate through each bracket

    if len(brackets) == 1:
        rate, limit = brackets[0]
        taxable_amount = max(0, income - limit)
        tax_owed += taxable_amount * rate
    else:
        for i in range(1, len(brackets)):
            rate, limit = brackets[i-1]
            next_limit = brackets[i][1]
            
            # Check if income is within the current bracket
            if income > limit:
                # Calculate the taxable amount in the current bracket
                taxable_amount = min(income, next_limit) - limit
                
                # Add the tax for the current bracket to the total tax owed
                tax_owed += taxable_amount * rate
    
    return tax_owed
        





# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(state_taxes_raw): 
    state_clean = state_taxes_raw.dropna(how='all').reset_index(drop=True)
    state_clean.loc[state_clean['State'].str.contains('\(', na=True),'State'] =np.nan
    state_clean['State'] = state_clean['State'].fillna(method='ffill')
    state_clean['Rate'] = np.round(state_clean['Rate'].str.replace('%', '').str.replace('none','0').astype(float)/100,2)
    state_clean['Lower Limit'] =state_clean['Lower Limit'].str.replace('$', '',regex=True).str.replace(',','').fillna(0).astype(int)

    return state_clean

# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def state_brackets(state_taxes):
    table = state_taxes.set_index('State').apply(
        lambda x :
        [list(zip(x.filter(like='Rate').tolist()
                 , x.filter(like='Lower Limit').tolist()
            ))]
        ,axis=1,result_type='expand')
    table.columns = ['bracket_list']

    return table.groupby('State')['bracket_list'].sum().reset_index().set_index('State')
def combine_loans_and_state_taxes(loans, state_taxes):
    # Start by loading in the JSON file.
    # state_mapping is a dictionary; use it!
    import json
    state_mapping_path = Path('data') / 'state_mapping.json'
    with open(state_mapping_path, 'r') as f:
        state_mapping = json.load(f)
    
    state_taxes = state_brackets(state_taxes).reset_index()
    # Now it's your turn:
    state_taxes['State'] = state_taxes['State'].apply(lambda x: state_mapping.get(x))
    new_df = pd.merge(loans,state_taxes, left_on='addr_state', right_on='State', how='left')
    return new_df.drop(columns='addr_state')



# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(loans_with_state_taxes):
    FEDERAL_BRACKETS = [
     (0.1, 0), 
     (0.12, 11000), 
     (0.22, 44725), 
     (0.24, 95375), 
     (0.32, 182100),
     (0.35, 231251),
     (0.37, 578125)
    ]
    loans_with_state_taxes['federal_tax_owed'] =loans_with_state_taxes.apply(lambda x: tax_owed(x['annual_inc'], FEDERAL_BRACKETS), axis=1)
    loans_with_state_taxes['state_tax_owed'] = loans_with_state_taxes.apply(lambda x : tax_owed(x['annual_inc'],x['bracket_list']), axis=1)
    loans_with_state_taxes['disposable_income'] = loans_with_state_taxes['annual_inc'] - loans_with_state_taxes['federal_tax_owed'] - loans_with_state_taxes['state_tax_owed']
    return loans_with_state_taxes

# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def aggregate_and_combine(loans, keywords, quantitative_column, categorical_column):
    output_df = pd.DataFrame()
    for i in keywords:
        keywords_filter = loans[loans['emp_title'].str.contains(i)]

        table_avg= round(keywords_filter.groupby(categorical_column)[quantitative_column].mean(),2)
        output_df[f'{i}_mean_{quantitative_column}'] = table_avg
        output_df.at['Overall', f'{i}_mean_{quantitative_column}'] = round(keywords_filter[quantitative_column].mean(),2)

    return output_df

# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def exists_paradox(loans, keywords, quantitative_column, categorical_column):
    table = aggregate_and_combine(loans,keywords,quantitative_column,categorical_column)
    return ( (table.iloc[:-1,0] > table.iloc[:-1,1]).any() and (table.iloc[-1,0]< table.iloc[-1,1] ) ).item()


def paradox_example(loans):
    return {
        'loans': loans,
        'keywords': ['administrator', 'sales'],
        'quantitative_column': 'annual_inc',
        'categorical_column': 'home_ownership'
    }
