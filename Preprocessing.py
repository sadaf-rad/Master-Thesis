# Preprocessing.py
import pandas as pd
import networkx as nx

def preprocess_data():
    # Load transactions dataset
    data_t = pd.read_csv("sarafu_txns_20200125-20210615.csv", on_bad_lines='skip', encoding='utf-8', delimiter=",")
    data_t['timestamp'] = pd.to_datetime(data_t['timeset'], errors='coerce')
    data_t['month'] = data_t['timestamp'].dt.to_period('M')  # Extract month
    data_t = data_t.dropna(subset=['timestamp'])

    # Load users dataset
    data_u = pd.read_csv("sarafu_users_20210615.csv", on_bad_lines='skip', encoding='utf-8', delimiter=",")

    # Create a directed graph from transactions
    G = nx.from_pandas_edgelist(data_t, source='source', target='target', edge_attr=['weight'], create_using=nx.DiGraph())
    return G, data_t, data_u  # Return graph and data for further use

