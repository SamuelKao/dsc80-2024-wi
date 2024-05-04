# lab.py


import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def simple_pipeline(data):


    plt = Pipeline([
        ('log-scales',FunctionTransformer(np.log)),
        ('Linear Regression', LinearRegression())
    ])

    plt.fit(data[['c2']],data['y'])
    prediction = plt.predict(data[['c2']])
    return plt,prediction

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multi_type_pipeline(data):
    prec = ColumnTransformer(transformers=[
                                ('log_c2',FunctionTransformer(np.log),['c2']),
                                ('ohe_group', OneHotEncoder(),['group']),
                                ],
                            remainder= 'passthrough')

    plt = Pipeline([
        ('predecessor', prec),
        ('predict_y', LinearRegression()) 
        ])
    
    plt.fit(data.drop('y',axis=1),data['y'])

    prediction = plt.predict(data.drop('y',axis=1))
    return plt, prediction
    


    
# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


# Imports
from sklearn.base import BaseEstimator, TransformerMixin

class StdScalerByGroup(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # X might not be a pandas DataFrame (e.g. a np.array)
        df = pd.DataFrame(X)
        # Compute and store the means/standard-deviations for each column (e.g. 'c1' and 'c2'), 
        # for each group (e.g. 'A', 'B', 'C').  
        # (Our solution uses a dictionary)
        self.grps_ = {}
        for i in df.columns[1:]:
            for group in df.iloc[:,0].unique():
                group_data = df[df.iloc[:,0]==group][i]
                key = (i, group)
                self.grps_[key] = {'mean': group_data.mean(), 'std': group_data.std()}

        return self
    
    def transform(self, X, y=None):

        try:
            getattr(self, "grps_")
        except AttributeError:
            raise RuntimeError("You must fit the transformer before tranforming the data!")
        
        # Hint: Define a helper function here!
        
        df = pd.DataFrame(X)
        def standardize_group (group_data,mean,std):
            return (group_data-mean) / std
        result_df = df.copy()
        for i in df.columns[1:]:
            for group in df.iloc[:,0].unique():
                key = (i,group)
                mean_val = self.grps_[key]['mean']
                std_val = self.grps_[key]['std']
                result_df.loc[df.iloc[:,0] == group, i] = standardize_group(df[df.iloc[:,0] == group][i], mean_val, std_val)
        return result_df.iloc[:,1:]


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def eval_toy_model():
    return [(2.755108697451811,0.39558507345910754),(2.3148336164355263,0.5733249315673331),(2.315733947782385,0.5729929650348397)]


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

def rmse(train,train_prediction,test,test_prediction):
    from sklearn.metrics import mean_squared_error
    train = mean_squared_error(train,train_prediction,squared=False)
    test = mean_squared_error(test,test_prediction,squared=False)
    return train,test

def tree_reg_perf(galton):
    # Add your imports here
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.tree import DecisionTreeRegressor
    
    df = pd.DataFrame()
    feature_col = ['father','mother','children','childNum','gender']
    x = galton[feature_col]
    y = galton['childHeight']
    x_trains,x_tests,y_trains,y_test = train_test_split(x,y,test_size=0.25,random_state=100)
    test_err = []
    train_err = []
    for i in range(1,21):   
        knn =  DecisionTreeRegressor(max_depth=i)
        knn.fit(x_trains,y_trains)
        test_prediction = knn.predict(x_tests)
        train_prediction = knn.predict(x_trains)
        
        train_rmse,test_rmse = rmse(y_trains,train_prediction,y_test,test_prediction)

        test_err.append(test_rmse)
        train_err.append(train_rmse)
    df['train_err'] = train_err
    df['test_err'] = test_err
    df.index = range(1,21)

    return df 


def knn_reg_perf(galton):
    # Add your imports here
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.neighbors import KNeighborsRegressor
    df = pd.DataFrame()
    feature_col = ['father','mother','children','childNum','gender']
    x = galton[feature_col]
    y = galton['childHeight']
    x_trains,x_tests,y_trains,y_test = train_test_split(x,y,test_size=0.25)
    test_err = []
    train_err = []
    for i in range(1,21):   
        knn =  KNeighborsRegressor(n_neighbors=i)
        knn.fit(x_trains,y_trains)
        test_prediction = knn.predict(x_tests)
        train_prediction = knn.predict(x_trains)
        
        train_rmse,test_rmse = rmse(y_trains,train_prediction,y_test,test_prediction)


        test_err.append(test_rmse)
        train_err.append(train_rmse)
    df['train_err'] = train_err
    df['test_err'] = test_err
    df.index = range(1,21)
    return df 

# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def titanic_model(titanic):
    # Add your import(s) here
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler



    def extract_name (df):
        df['Title'] = df['Name'].str.extract(r'(Mr|Miss|Mrs)\.')
        return df[['Title']]
    
    def standardize_age(df):
        df['Standardized_Age'] = df.groupby('Pclass')['Age'].transform(lambda x: (x - x.mean()) / x.std())
        return df[['Standardized_Age']]
    transform_name = FunctionTransformer(extract_name,validate=False)

    pl = Pipeline([
        ('Name', transform_name),
        ('Nameohe',OneHotEncoder()),
    ])
    pl_group = Pipeline([
    
        ('ageohe',StdScalerByGroup())
    ])
    preprocessor = ColumnTransformer(
        [
            ('Name',pl,['Name']),
            ('Age',pl_group,['Pclass','Age']),
            ('std', StandardScaler(), ['Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']),
            ('ohe', OneHotEncoder(), ['Sex'])
        ],
        remainder='passthrough'
    )
    
    classifier = RandomForestClassifier(n_estimators=1,random_state=42)
    titanic_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classifier)
    ])

    titanic_model.fit(titanic.drop('Survived',axis=1),titanic['Survived'])
   
    return titanic_model

