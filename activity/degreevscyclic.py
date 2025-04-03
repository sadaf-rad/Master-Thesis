import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import networkx as nx

with open("monthly_graphs.pkl", "rb") as f:
    monthly_graphs = pickle.load(f)

with open("degree_centralities.pkl", "rb") as f:
    degree_centralities = pickle.load(f)

records = []
print("Processing cyclic status and merging with degree centrality...")

for month, G in tqdm(monthly_graphs.items()):
    # Cyclic users from SCCs
    sccs = list(nx.strongly_connected_components(G))
    cyclic_users = set()
    for comp in sccs:
        if len(comp) > 1:
            cyclic_users.update(comp)

    deg_dict = degree_centralities.get(month, {})
    
    for user in G.nodes():
        records.append({
            "month": str(month),
            "user": user,
            "degree": deg_dict.get(user, 0),
            "cyclic": user in cyclic_users
        })

# DataFrame
df = pd.DataFrame(records)
df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").astype(str)

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="cyclic", y="degree", palette="Set2")
plt.title("Are High-Degree Users Cyclic?")
plt.xlabel("Cyclic")
plt.ylabel("Degree Centrality")
plt.tight_layout()
plt.savefig("boxplot_degree_vs_cyclic.png")
plt.show()

df["degree_group"] = pd.qcut(df["degree"], q=5, labels=["Very Low", "Low", "Mid", "High", "Very High"])

cyclic_ratio = df.groupby("degree_group")["cyclic"].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(data=cyclic_ratio, x="degree_group", y="cyclic", palette="coolwarm")
plt.title("Proportion of Cyclic Users by Degree Group")
plt.xlabel("Degree Group")
plt.ylabel("Proportion Cyclic")
plt.tight_layout()
plt.savefig("cyclic_ratio_by_degree_group.png")
plt.show()

correlation = df[["degree", "cyclic"]].corr().iloc[0, 1]
print(f" Correlation between degree and cyclic status: {correlation:.2f}")
