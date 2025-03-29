import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# === Step 1: Load Monthly Graphs ===
with open("monthly_graphs.pkl", "rb") as f:
    monthly_graphs = pickle.load(f)

kshell_records = []

print("🧠 Calculating K-shell values per month...")

# === Step 2: Compute K-shell Per Month ===
for month, G in tqdm(monthly_graphs.items()):
    try:
        G_undirected = G.to_undirected()
        core_dict = nx.core_number(G_undirected)

        for node, kshell in core_dict.items():
            kshell_records.append({
                'user': node,
                'kshell': kshell,
                'month': str(month)
            })
    except Exception as e:
        print(f"⚠️ Error in {month}: {e}")

# === Step 3: Create DataFrame and Sort ===
df_kshell = pd.DataFrame(kshell_records)
df_kshell["month"] = pd.to_datetime(df_kshell["month"]).dt.to_period("M").astype(str)

# Sort by month and descending kshell
df_kshell_sorted = df_kshell.sort_values(by=["month", "kshell"], ascending=[True, False])

# Save to CSV
df_kshell_sorted.to_csv("monthly_kshell_sorted.csv", index=False)
print("✅ Saved sorted K-shell scores to 'monthly_kshell_scores_sorted.csv'")

# === Step 4: Plot K-shell Distribution Over Time ===
print("📊 Plotting K-shell distribution...")

# Count users in each kshell per month
kshell_counts = df_kshell.groupby(["month", "kshell"]).size().reset_index(name="count")

plt.figure(figsize=(14, 6))
sns.lineplot(data=kshell_counts, x="month", y="count", hue="kshell", palette="viridis")

plt.title("K-shell User Distribution Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Users")
plt.xticks(rotation=45)
plt.legend(title="K-shell")
plt.tight_layout()

# Save plot
plt.savefig("kshell_distribution_over_time.png")
plt.show()

print("📈 Plot saved as 'kshell_distribution_over_time.png'")
