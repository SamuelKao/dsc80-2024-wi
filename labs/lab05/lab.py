# lab.py


from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def after_purchase():
    return ['NMAR','MD','MAR','MAR','MAR']


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multiple_choice():
    return ['MAR', 'MAR','MD', 'NMAR', 'MCAR']


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------



def first_round():
    '''
    payments['date_of_birth'] = pd.to_datetime(payments['date_of_birth'], format='%d-%b-%Y')
    payments['age'] = 2024 - payments['date_of_birth'].dt.year
    missing_credit_card = payments[payments['credit_card_number'].isnull()]
    not_missing_credit_card = payments[payments['credit_card_number'].notnull()]
    fig = create_kde_plotly(payments, 'credit_card_number', 'age', 'not missing', 'age')
    '''
    return [0.283,'NR']


    #create_kde_plotly(df, group_col, group1, group2, vals_col, title='')

'''
1. get observe_stats by 
2. use np.random.permutation 
3. get p-value


'''

def second_round():
    return [0.02,'R','D']

# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def verify_child(heights):

    p_values = pd.Series(index=heights.columns[2:])
    for i in heights.columns[2:]:
        missing_height = heights[i].isna()
        not_missing_height = heights[i].notna()
        missing_sample = stats.ks_2samp(heights.loc[missing_height,'father'],heights.loc[not_missing_height,'father']).pvalue
        p_values[i] = missing_sample

    return p_values



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

def cond_single_imputation(new_heights):
    '''
    using single-valued mean imputation
    return a series
    '''
    new_heights = new_heights.sort_values(by='father',ascending = False)
    #new_heights['father quartile'] = pd.qcut(new_heights['father'],4, ['Q1', 'Q2', 'Q3', 'Q4'])
    #mean_imputation = new_heights.groupby('father quartile')['child'].mean()

    '''new_heights['child'] = new_heights['child'].fillna(mean_imputation)

    new_heights = new_heights.drop(columns= ['father quartile'])'''
    new_heights['father quartile'] = pd.qcut(new_heights['father'],4)
    mean_imputation = new_heights.groupby('father quartile')['child'].transform('mean')
    new_heights['child'] = new_heights['child'].fillna(mean_imputation)
    return new_heights['child']
# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    '''
    Create a histogram of observed child heights using 10 bins

    '''
    observed_values, bin_edge = np.histogram(child.dropna(),bins=10,density =True)
    hist_normalized = observed_values / observed_values.sum()
    
    select_bin = np.random.choice (len(bin_edge)-1,size = N,p=hist_normalized)
    imputed_values = np.random.uniform(bin_edge[select_bin], bin_edge[select_bin + 1], size=N)

    return imputed_values
def impute_height_quant(child):
    missing = child.index[child.isna()]

    imputation = quantitative_distribution(child,len(missing))
    child.loc[missing] = imputation


    return child
# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def answers():
    return [1,2,2,1],['https://en.wikipedia.org/robots.txt','https://www.indeed.com/robots.txt']
