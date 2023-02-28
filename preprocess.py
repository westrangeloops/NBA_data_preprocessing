import numpy as np
import pandas as pd
import os
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"

# write your code here

def clean_data(path):
    df_raw = pd.read_csv(path)
    # parse b_day and draft_year as datetime
    df_raw['b_day'] = pd.to_datetime(df_raw['b_day'], format='%m/%d/%y')
    df_raw['draft_year'] = pd.to_datetime(df_raw['draft_year'], format='%Y')
    # Replace missing values in teams with no teams:
    df_raw['team'].fillna('No Team', inplace=True)
    # height in meters
    df_raw['height'] = df_raw['height'].apply(lambda x: float(x.split('/')[1]))
    # weight in kg
    df_raw['weight'] = df_raw['weight'].apply(lambda x: float(x.split('/')[1].replace('kg.', '')))
    # remove $ from salary
    df_raw['salary'] = df_raw['salary'].apply(lambda x: float(x[1:]))
    # Americanize the word
    df_raw['country'] = df_raw['country'].apply(lambda x: x if x == 'USA' else 'Not-USA')
    # 'Undrafted to 0'
    df_raw['draft_round'] = df_raw['draft_round'].apply(lambda x: '0' if x == 'Undrafted' else x)
    return df_raw

def feature_data(raw_data):
    raw_data['version'] = raw_data['version'].apply(lambda x: x.replace('NBA', '').replace('k', '0'))
    raw_data['version'] = pd.to_datetime(raw_data['version'], format='%Y')
    raw_data['age'] = pd.DatetimeIndex(raw_data['version']).year - pd.DatetimeIndex(raw_data['b_day']).year
    raw_data['experience'] = pd.DatetimeIndex(raw_data['version']).year - pd.DatetimeIndex(raw_data['draft_year']).year
    raw_data['bmi'] = raw_data['weight'] / (raw_data['height'])**2
    raw_data.drop(['version', 'b_day', 'draft_year', 'weight', 'height', 'full_name', 'college', 'jersey', 'draft_peak'], axis=1, inplace=True)
    return raw_data

def multicol_data(df_1):
    df_plot = sns.heatmap(df_1.corr(method='pearson'), cmap="YlGnBu", annot=True)
    plt.savefig('corr.png')
    df_1.drop('age', axis=1, inplace=True)
    return df_1
def transform_data(df_2):
    # Label
    y = df_2['salary']
    df_2.drop(['salary'], axis=1, inplace=True)
    # X
    num_df2 = df_2.select_dtypes('number')
    cat_df2 = df_2.select_dtypes('object')
    # Num scale
    columns_names = num_df2.columns.tolist()
    scaler = StandardScaler()
    df_st = scaler.fit_transform(num_df2)
    num_df2 = pd.DataFrame(df_st, columns=columns_names)
    # Encoder
    encode = OneHotEncoder()
    cat_enc = encode.fit_transform(cat_df2)
    column_names = encode.categories_
    list_of_columns = np.concatenate(column_names).ravel().tolist()
    cat_df2 = pd.DataFrame.sparse.from_spmatrix(cat_enc, columns=list_of_columns)
    # concatenate
    df = pd.concat([num_df2, cat_df2], axis=1, join='inner')
    return df, y


raw_data = clean_data(data_path)
df_1 = feature_data(raw_data)
df_2 = multicol_data(df_1)
X, y = transform_data(df_2)
