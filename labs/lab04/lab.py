# lab.py


import pandas as pd
import numpy as np
import io
from pathlib import Path
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def prime_time_logins(login):
    login = login.copy()
    login['Time'] = pd.to_datetime(login['Time'])

    a = login.loc[:,'Time'].apply(lambda x: 1 if x.hour>=16 and (x.hour<20 ) else 0)
    login['Time'] = a
    output = login.groupby('Login Id').sum()
    return output

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def count_frequency(login):
    

    login['Time'] = pd.to_datetime(login['Time'])
    today = pd.Timestamp('2024-01-31 23:59:00')

    
    login['DaysOnSite'] = (today - login.groupby('Login Id')['Time'].transform('min')).dt.days

    # Create a custom aggregator function
    def custom_aggregator(data):
        total_logins = data['Time'].count()
        total_days = data['DaysOnSite'].max()
        return total_logins / total_days

    # Use the custom aggregator with groupby to get logins per day for each user
    output = login.groupby('Login Id').apply(custom_aggregator)
    login.drop('DaysOnSite', axis=1, inplace=True)

    return  output
# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def cookies_null_hypothesis():
    return [2]
                         
def cookies_p_value(N):
    np.random.seed(42)
    
    null_samples = np.random.choice([0, 1], size=(N, 250), p=[0.96, 0.04])
    count = np.sum(null_samples, axis=1)
    p_value = np.sum(count >= 10) / N
    
    return p_value


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def car_null_hypothesis():
    return [1,4]

def car_alt_hypothesis():
    return [2,6]

def car_test_statistic():
    return [1,4]

def car_p_value():
    return 3


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def superheroes_test_statistic():
    return [1,2]
    
def bhbe_col(heroes):
    #eye color is blue and hair is blond
    hair_eye_color = ( heroes.loc[:,'Eye color'].str.lower().str.contains('blue')) & ( heroes.loc[:,'Hair color'].str.lower().str.contains('blond'))

    return hair_eye_color

#The proportion of 'good' character among blond-haired, blue-eyed character
def superheroes_observed_statistic(heroes):
    #eye color is blue and hair is blond
    hero = bhbe_col(heroes)

    good = heroes['Alignment'].str.lower() == 'good'
    observed_proportion = good[hero].sum() / hero.sum()
    return observed_proportion #0.849

def simulate_bhbe_null(heroes, N): 
    probability = heroes['Alignment'].value_counts(normalize=True)['good']

    shuffled_alignments = np.random.multinomial(len(heroes['Alignment']),[probability,1-probability],N)
    return shuffled_alignments[:,0] / len(heroes)
    
def superheroes_p_value(heroes):
    
    observe = superheroes_observed_statistic(heroes)
    simulation = simulate_bhbe_null(heroes,100000)
    p_value = (simulation >= observe).mean()
    hypothesis_result = 'Reject' if p_value < 0.01 else 'Fail to reject'
    
    return [p_value, hypothesis_result]




# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def diff_of_means(data, col='orange'):
    average_diff = abs(data.groupby('Factory')[col].mean().diff()).sum() /2
    return average_diff
def simulate_null(data, col='orange'):
    shuffle_col = np.random.permutation(data[col])
    df_shuffled = data.copy()
    df_shuffled[col] = shuffle_col

    

    return diff_of_means(df_shuffled,col)


def color_p_value(data, col='orange'):
    null_distribution = [simulate_null(data,col) for _ in range(1000)]
    null_distribution = np.array(null_distribution)

    test_stats = (diff_of_means(data,col) <= null_distribution).mean()

    
    return test_stats



# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def ordered_colors():
    return [('yellow',0.000),('orange',0.042),('red',0.247) ,('green',0.448),('purple',0.974)]


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def same_color_distribution():
    return ( 0.006,'Reject')


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def perm_vs_hyp():
    return ['P','P','H','H','P']
