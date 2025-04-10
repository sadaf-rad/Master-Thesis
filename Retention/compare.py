import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def plot_individual_user_lines():
    print(" Loading transaction and early feature data...")
    df_tx = pd.read_csv("sarafu_txns_20200125-20210615.csv", on_bad_lines='skip', encoding='utf-8')
    df_users = pd.read_csv("/home/s3986160/master-thesis/Retention/early_user_features.csv")

    df_tx['timestamp'] = pd.to_datetime(df_tx['timeset'], errors='coerce')
    df_users['first_active_date'] = pd.to_datetime(df_users['first_active_date'])
    df_tx = df_tx.dropna(subset=['timestamp'])
    df_tx['user'] = df_tx['source']

    # Merge first active date and filter relevant window
    merged = df_tx.merge(df_users[['user', 'first_active_date']], on='user', how='inner')
    merged['days_since_first'] = (merged['timestamp'] - merged['first_active_date']).dt.days
    merged = merged[(merged['days_since_first'] >= 0) & (merged['days_since_first'] < 180)].copy()
    merged['time_window'] = (merged['days_since_first'] // 30).astype(int)

    # Aggregate features per user per window
    print(" Aggregating basic user features...")
    features = merged.groupby(['user', 'time_window']).agg(
        tx_count=('timestamp', 'count'),
        active_days=('timestamp', lambda x: x.dt.date.nunique()),
        partners=('target', 'nunique')
    ).reset_index()

    # Degree and k-shell per time window
    print(" Computing degree and k-shell...")
    deg_list = []
    for w in sorted(merged['time_window'].unique()):
        sub = merged[merged['time_window'] == w]
        G = nx.DiGraph()
        G.add_edges_from(zip(sub['source'], sub['target']))
        degrees = dict(G.degree())
        kshell = nx.core_number(G.to_undirected()) if len(G) > 0 else {}

        for u in degrees:
            deg_list.append({
                'user': u,
                'time_window': w,
                'degree': degrees[u],
                'k_shell': kshell.get(u, 0)
            })

    deg_df = pd.DataFrame(deg_list)
    full_df = features.merge(deg_df, on=['user', 'time_window'], how='left').fillna(0)

    print(" Plotting individual user trends (line plots)...")
    metrics = ['tx_count', 'active_days', 'partners', 'degree', 'k_shell']
    labels = {
        'tx_count': 'Transaction Count',
        'active_days': 'Active Days',
        'partners': 'Unique Partners',
        'degree': 'Degree',
        'k_shell': 'K-Shell'
    }

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=full_df,
            x='time_window',
            y=metric,
            hue='user',
            linewidth=1,
            legend='full'  # Add legend per user
        )
        plt.title(f'{labels[metric]} per User Over Time')
        plt.xlabel('Time Window (Months Since First Transaction)')
        plt.ylabel(labels[metric])
        plt.xticks(ticks=range(6), labels=["0–30d", "30–60d", "60–90d", "90–120d", "120–150d", "150–180d"])
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize='small', title='User')
        plt.tight_layout()
        path = f"/home/s3986160/master-thesis/Retention/lineplot_per_user_{metric}.png"
        plt.savefig(path, dpi=300)
        plt.show()
        print(f" Saved: {path}")
   

# Run the function
plot_individual_user_lines()
