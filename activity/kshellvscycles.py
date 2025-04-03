import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm

with open("monthly_graphs.pkl", "rb") as f:
    monthly_graphs = pickle.load(f)


df_kshell_sorted = pd.read_csv("monthly_kshell_scores_sorted.csv")


cyclic_records = []

for month, G in tqdm(monthly_graphs.items(), desc="Detecting cyclic users..."):
    sccs = list(nx.strongly_connected_components(G))
    cyclic_users = set()

    for comp in sccs:
        if len(comp) > 1:
            cyclic_users.update(comp)

    for user in G.nodes():
        cyclic_records.append({
            "month": str(month),
            "user": user,
            "cyclic": user in cyclic_users
        })

df_cyclic_flags = pd.DataFrame(cyclic_records)
df_cyclic_flags["month"] = pd.to_datetime(df_cyclic_flags["month"]).dt.to_period("M").astype(str)

# Merge with K-shell Data 
df_combined = pd.merge(df_kshell_sorted, df_cyclic_flags, on=["month", "user"])

# Filter for Cyclic Users Only 
df_cyclic_only = df_combined[df_combined["cyclic"] == True]

#  Count of Cyclic Users by K-shell 
cyclic_kshell_counts = df_cyclic_only["kshell"].value_counts().sort_index().reset_index()
cyclic_kshell_counts.columns = ["kshell", "count"]

#  Compute Proportion (within cyclic users only) 
total_cyclic_users = cyclic_kshell_counts["count"].sum()
cyclic_kshell_counts["percentage"] = (cyclic_kshell_counts["count"] / total_cyclic_users) * 100


plt.figure(figsize=(10, 6))
sns.barplot(data=cyclic_kshell_counts, x="kshell", y="percentage", color="green")
plt.title("Proportion of Cyclic Users by K-shell ")
plt.xlabel("K-shell Index")
plt.ylabel("Percentage of Cyclic Users")
plt.tight_layout()
plt.savefig("cyclic_users_only_by_kshell.png")
plt.show()
