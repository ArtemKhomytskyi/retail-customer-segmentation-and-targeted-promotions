# utils.py
# Main imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import missingno as msno
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px

import ast
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def plot_cluster_profile(df, feature_columns, cluster_col='pred_cluster'):
    """
    Plot mean values of features per cluster.
    """
    mean_values = df.groupby(cluster_col)[feature_columns].mean().reset_index()
    mean_values_long = pd.melt(mean_values, id_vars=[cluster_col], var_name='variable', value_name='mean_value')
    
    sns.barplot(x='variable', y='mean_value', hue=cluster_col, data=mean_values_long)
    plt.xticks(rotation=45)
    plt.title('Cluster Profile')
    plt.tight_layout()
    plt.show()

def cap_outliers(df, columns):
    """
    Cap outliers in specified columns using IQR method.
    """
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        IQR = q3 - q1
        lower_bound = q1 - 1.5 * IQR
        upper_bound = q3 + 1.5 * IQR
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

def find_time_slot(hour):
    """
    Map an hour to a time slot.
    """
    time_slots = {
        1: range(6, 9), # "early_morning"
        2: range(9, 13), # "morning"
        3: range(13, 18), # "afternoon"
        4: range(18, 22), # "evening"
        5: range(22, 24) # "night"
    }
    for slot, hour_range in time_slots.items():
        if hour in hour_range:
            return slot
    return 0 # "late_night"

def calculate_outliers(data):
    outlier_indices = {}
    outlier_proportions = {}

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            IQR = q3 - q1

            lower_bound = q1 - 1.5 * IQR
            upper_bound = q3 + 1.5 * IQR

            outliers_min = data[data[column] < lower_bound].index.to_series()
            outliers_max = data[data[column] > upper_bound].index.to_series()

            outliers = pd.concat([outliers_min, outliers_max], axis=0).unique()
            outlier_indices[column] = outliers

            outlier_proportion = len(outliers) / len(data[column])
            outlier_proportions[column] = outlier_proportion

    return outlier_indices, outlier_proportions

def assosiations_cl(df_basket, cluster, type):
    print(f"\n=== Association Rules for Cluster {cluster} ===")
    
    cluster_basket = df_basket[df_basket[type] == cluster]
    
    # Конвертим list_of_goods из строки в список
    cluster_basket['list_of_goods'] = cluster_basket['list_of_goods'].apply(clean_list_of_goods)
    
    # Готовим transaction list
    transactions = cluster_basket['list_of_goods'].tolist()
    
    # TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Apriori
    frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)
    
    # Association Rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    
    # Печатаем топ 10 правил
    return rules

def clean_list_of_goods(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        return x.strip("[]").replace("'", "").split(", ")
    else:
        return []
