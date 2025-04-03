import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_kshell = pd.read_csv("monthly_kshell_scores_sorted.csv")
df_kshell["month"] = pd.to_datetime(df_kshell["month"]).dt.to_period("M").astype(str)

with open("degree_centralities.pkl", "rb") as f:
    degree_centralities = pickle.load(f)

#  DataFrame
records = []
for month, centrality_dict in degree_centralities.items():
    for user, degree in centrality_dict.items():
        records.append({
            "user": user,
            "month": str(month),
            "degree": degree
        })

df_degree = pd.DataFrame(records)
df_degree["month"] = pd.to_datetime(df_degree["month"]).dt.to_period("M").astype(str)

df = pd.merge(df_kshell, df_degree, on=["user", "month"], how="inner")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="kshell", y="degree", alpha=0.4)
plt.title("Scatterplot: K-shell vs Degree Centrality")
plt.xlabel("K-shell")
plt.ylabel("Degree Centrality")
plt.tight_layout()
plt.savefig("scatter_kshell_vs_degree.png")
plt.show()

correlation = df[["kshell", "degree"]].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: K-shell vs Degree")
plt.tight_layout()
plt.savefig("heatmap_kshell_vs_degree.png")
plt.show()

bar_data = df.groupby("kshell")["degree"].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(data=bar_data, x="kshell", y="degree", palette="Set2")
plt.title("Bar Chart: Avg Degree by K-shell")
plt.xlabel("K-shell")
plt.ylabel("Average Degree Centrality")
plt.tight_layout()
plt.savefig("bar_avg_degree_by_kshell.png")
plt.show()

print(" Correlation between k-shell and degree:", round(correlation.iloc[0, 1], 2))
