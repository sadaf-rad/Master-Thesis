import pandas as pd
import networkx as nx
import pickle

def preprocess_early_behavior():
    print("Loading transaction data...")
    data_t = pd.read_csv("sarafu_txns_20200125-20210615.csv", on_bad_lines='skip', encoding='utf-8', delimiter=",")
    
    print("Fixing timestamp...")
    data_t['timestamp'] = pd.to_datetime(data_t['timeset'], errors='coerce')
    data_t = data_t.dropna(subset=['timestamp'])

    # Filter to 2020 only
    data_2020 = data_t[data_t['timestamp'].dt.year == 2020].copy()
    data_2020['month'] = data_2020['timestamp'].dt.to_period('M')

    print("Finding first activity per user...")
    senders = data_2020[['source', 'timestamp']].rename(columns={'source': 'user'})
    receivers = data_2020[['target', 'timestamp']].rename(columns={'target': 'user'})
    all_users = pd.concat([senders, receivers])
    first_activity = all_users.groupby('user')['timestamp'].min().reset_index()
    first_activity.columns = ['user', 'first_active_date']

    # Only users who joined in 2020
    first_2020 = first_activity[first_activity['first_active_date'].dt.year == 2020]

    print("Extracting first 60 days of activity from each user's first transaction...")
    data_2020['user'] = data_2020['source']
    merged = data_2020.merge(first_2020, on='user', how='inner')
    merged['days_since_first'] = (merged['timestamp'] - merged['first_active_date']).dt.days
    df_60days = merged[(merged['days_since_first'] >= 0) & (merged['days_since_first'] < 60)]

    ### BASIC FEATURES
    print("Calculating basic features...")
    basic_features = df_60days.groupby('user').agg(
        tx_count=('timestamp', 'count'),
        active_days=('timestamp', lambda x: x.dt.date.nunique()),
        first_tx=('timestamp', 'min'),
        last_tx=('timestamp', 'max'),
        partners=('target', 'nunique')
    ).reset_index()
    basic_features['duration'] = (basic_features['last_tx'] - basic_features['first_tx']).dt.days

    ### GRAPH FEATURES (Degree + k-shell)
    print("Building transaction graph...")
    G = nx.DiGraph()
    for _, row in df_60days.iterrows():
        G.add_edge(row['source'], row['target'])

    print("Computing degree and k-shell...")
    degrees = dict(G.degree())
    kshells = nx.core_number(G.to_undirected())

    graph_features = pd.DataFrame({
        'user': list(degrees.keys()),
        'degree': list(degrees.values()),
        'k_shell': [kshells.get(u, 0) for u in degrees.keys()]
    })

    ### MERGE ALL FEATURES
    print("Combining features and first transaction date...")
    final_df = basic_features.merge(graph_features, on='user', how='left')
    final_df = final_df.merge(first_2020[['user', 'first_active_date']], on='user', how='left')

    ### SAVE OUTPUT
    print("Saving as CSV and PKL...")
    final_df.to_csv('early_user_features.csv', index=False)
    with open('early_user_features.pkl', 'wb') as f:
        pickle.dump(final_df, f)

    print(" Done! Total users with extracted features:", len(final_df))

if __name__ == "__main__":
    preprocess_early_behavior()
