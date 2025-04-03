import pickle
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import community.community_louvain as community_louvain  # make sure this import works
from Preprocessing import preprocess_data_and_save

def load_or_create_graphs():
    try:
        with open('monthly_graphs.pkl', 'rb') as f:
            monthly_graphs = pickle.load(f)
        print(" Loaded graphs from pickle.")
    except FileNotFoundError:
        print(" Pickle not found. Reprocessing data...")
        _, _, monthly_graphs = preprocess_data_and_save()
    return monthly_graphs

def analyze_cyclic_distribution_across_communities(monthly_graphs):
    results = []

    for month, G in tqdm(monthly_graphs.items(), desc="Analyzing cyclic users and communities"):
        G_u = G.to_undirected()

        if len(G_u.nodes) == 0 or len(G_u.edges) == 0:
            print(f"⚠️ Skipping {month} — graph is empty or has no edges.")
            continue

        try:
            partition = community_louvain.best_partition(G_u)
        except Exception as e:
            print(f"⚠️ Louvain failed in {month}: {e}")
            continue

        # Detect cyclic users using SCCs (on directed graph)
        sccs = list(nx.strongly_connected_components(G))
        cyclic_users = set()
        for comp in sccs:
            if len(comp) > 1:
                cyclic_users.update(comp)

        # Map cyclic users to their community
        community_ids = {user: partition.get(user) for user in cyclic_users if user in partition}
        cyclic_communities = set(community_ids.values())

        results.append({
            "month": str(month),
            "total_cyclic_users": len(cyclic_users),
            "total_communities": len(set(partition.values())),
            "communities_with_cyclic_users": len(cyclic_communities),
            "avg_cyclic_users_per_community": len(cyclic_users) / max(1, len(cyclic_communities))
        })

    return pd.DataFrame(results)

def plot_cyclic_community_distribution(df):
    if df.empty:
        print("⚠️ No data to plot — Louvain failed for all months or graphs were empty.")
        return

    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").astype(str)

    plt.figure(figsize=(12, 6))
    plt.plot(df["month"], df["avg_cyclic_users_per_community"], marker='o', color='purple')
    plt.title("Avg Cyclic Users per Community Over Time")
    plt.xlabel("Month")
    plt.ylabel("Avg Cyclic Users / Community")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("avg_cyclic_users_per_community.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df["month"], df["communities_with_cyclic_users"], marker='o', label="Communities w/ Cyclic Users", color="green")
    plt.plot(df["month"], df["total_communities"], marker='o', label="Total Communities", color="gray")
    plt.title("Cyclic Community Spread Over Time")
    plt.xlabel("Month")
    plt.ylabel("Community Count")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("cyclic_community_spread.png")
    plt.show()

def save_to_csv(df, filename="cyclic_community_distribution.csv"):
    df.to_csv(filename, index=False)
    print(f"✅ Saved results to {filename}")

if __name__ == "__main__":
    graphs = load_or_create_graphs()
    results_df = analyze_cyclic_distribution_across_communities(graphs)
    save_to_csv(results_df)
    plot_cyclic_community_distribution(results_df)
