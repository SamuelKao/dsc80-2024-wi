# lab.py


import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from pathlib import Path
from sklearn.preprocessing import Binarizer, QuantileTransformer, FunctionTransformer

import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def best_transformation():
    return 1


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------

def ordinal_col (col,ordering):
    return col.map({value: index for index, value in enumerate(ordering)})

def create_ordinal(df):
    output = pd.DataFrame()
    cut_ordering = ['Fair','Good','Very Good','Premium','Ideal']
    output['ordinal_cut'] = ordinal_col(df['cut'],cut_ordering)
    
    color_ordering = ['J','I','H','G','F','E','D']
    output['ordinal_color'] = ordinal_col(df['color'],color_ordering)

    clarity_ordering = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
    output['ordinal_clarity'] = ordinal_col(df['clarity'],clarity_ordering)

    return output
    



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_one_hot(df):
    #if the data column in the ordinal data column then 
    copy_df = df.copy()
    copy_df = create_ordinal(copy_df)
    output_df = pd.DataFrame()
    
    for col in copy_df.columns:
        unique_value = copy_df[col].unique()
        for value in unique_value:
            output_df[f'one_hot_{col}_{value}'] = (copy_df[col]==value).astype(int)
    return output_df
def create_proportions(df):
    output = pd.DataFrame()
    copy_df = create_ordinal(df)
    for col in copy_df.columns:
        value_count = copy_df[col].value_counts()
        proportion = copy_df[col].map(lambda x: value_count[x]/len(copy_df))
        output[f'proportion_{col}'] = proportion
    return output


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def create_quadratics(df):
    output = pd.DataFrame()
    quant_col = df.select_dtypes(include =['float64','int64']).columns
    for i in range(len(quant_col)):
        for j in range(i+1,len(quant_col)):
            col1, col2 = quant_col[i], quant_col[j]

            output[f'{col1} * {col2}'] = df[col1]*df[col2]
    output = output.filter(regex='x|y|z')
    return output
# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def comparing_performance():
    # create a model per variable => (variable, R^2, RMSE) table
    return [0.8493305264354858, 1548.533193,'x','carat * x', 'ordinal_color', 1440.2479185766772]


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


class TransformDiamonds(object):
    
    def __init__(self, diamonds):
        self.data = diamonds
        
    # Question 6.1
    def transform_carat(self, data):
        output = Binarizer(threshold=1).fit_transform(data[['carat']])
        return output
    # Question 6.2
    def transform_to_quantile(self, data):

        trans = QuantileTransformer(n_quantiles=100)
        trans.fit(self.data[['carat']])
        percentile = trans.transform(data[['carat']])
        return percentile
    # Question 6.3
    def transform_to_depth_pct(self, data):
        def depth (x):
            try:
                return 100 * 2 * x[:,2] / (x[:,0] + x[:,1])
            except ZeroDivisionError:
                return np.nan
        trans = FunctionTransformer(depth,validate=False)
        return trans.transform(data[['x','y','z']].values)
