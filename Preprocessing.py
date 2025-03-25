import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Load transactions dataset 
data_t = pd.read_csv("sarafu_txns_20200125-20210615.csv", on_bad_lines='skip',  encoding='utf-8',delimiter=",")

# Converting the Timestep
data_t['timestamp'] = pd.to_datetime(data_t['timeset'], errors='coerce')
data_t['month'] = data_t['timestamp'].dt.to_period('M')  # Extract month
data_t = data_t.dropna(subset=['timestamp'])

# Load users dataset
data_u = pd.read_csv("sarafu_users_20210615.csv", on_bad_lines='skip', encoding='utf-8', delimiter=",")

print(" Datasets loaded successfully!")

print(f"Raw Dataset Date Range: {data_t['timeset'].min()} to {data_t['timeset'].max()}")

# Extract date details
data_t['year'] = data_t['timestamp'].dt.year
data_t['month'] = data_t['timestamp'].dt.month

print(f"Data range: {data_t['timestamp'].min()} to {data_t['timestamp'].max()}")

print(data_t.groupby(['year', 'month']).size())

# Creating The Directed Graph
G = nx.from_pandas_edgelist(data_t, source='source', target='target', edge_attr=['weight'], create_using=nx.DiGraph())

<<<<<<< HEAD
print(f" Graph Created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
=======
print(f"Graph Created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
>>>>>>> e9ca15db039ed4d2f3786ef504e343a3604253ab
