import pickle
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from Preprocessing import preprocess_data_and_save


def load_or_create_graphs():
    try:
        with open('monthly_graphs.pkl', 'rb') as f:
            monthly_graphs = pickle.load(f)
        print("Loaded graphs from pickle.")
    except FileNotFoundError:
        print("Pickle not found. Reprocessing data...")
        _, _, monthly_graphs = preprocess_data_and_save()
    return monthly_graphs


def analyze_cyclic_acyclic_scc(monthly_graphs):
    result = []

    for month, G in tqdm(monthly_graphs.items(), desc="Analyzing SCCs"):
        sccs = list(nx.strongly_connected_components(G))

        cyclic_users = set()
        for comp in sccs:
            if len(comp) > 1:  # SCC with 2+ nodes implies at least one cycle
                cyclic_users.update(comp)

        all_users = set(G.nodes())
        acyclic_users = all_users - cyclic_users

        result.append({
            'month': str(month),
            'cyclic_users': len(cyclic_users),
            'acyclic_users': len(acyclic_users),
            'total_users': len(all_users)
        })

    return pd.DataFrame(result)


def plot_user_counts(df):
    # Plot cyclic only
    plt.figure(figsize=(10, 5))
    plt.plot(df['month'], df['cyclic_users'], marker='o', label='Cyclic Users', color='green')
    plt.xticks(rotation=45)
    plt.xlabel("Month")
    plt.ylabel("Number of Users")
    plt.title("Cyclic Users Over Time")
    plt.tight_layout()
    plt.savefig("cyclic_users_plot.png")
    plt.show(block=True)

    # Plot acyclic only
    plt.figure(figsize=(10, 5))
    plt.plot(df['month'], df['acyclic_users'], marker='o', label='Acyclic Users', color='blue')
    plt.xticks(rotation=45)
    plt.xlabel("Month")
    plt.ylabel("Number of Users")
    plt.title("Acyclic Users Over Time")
    plt.tight_layout()
    plt.savefig("acyclic_users_plot.png")
    plt.show(block=True)

    # Combined plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['month'], df['cyclic_users'], marker='o', label='Cyclic Users', color='green')
    plt.plot(df['month'], df['acyclic_users'], marker='o', label='Acyclic Users', color='blue')
    plt.xticks(rotation=45)
    plt.xlabel("Month")
    plt.ylabel("Number of Users")
    plt.title("Cyclic vs Acyclic Users Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison_plot.png")
    plt.show(block=True)


def save_to_csv(df, filename="cyclic_acyclic_user_stats.csv"):
    df.to_csv(filename, index=False)
    print(f"Saved user stats to {filename}")


if __name__ == "__main__":
    graphs = load_or_create_graphs()
    stats_df = analyze_cyclic_acyclic_scc(graphs)
    save_to_csv(stats_df)
    plot_user_counts(stats_df)
