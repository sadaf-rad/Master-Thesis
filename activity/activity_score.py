import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df_kshell = pd.read_csv("monthly_kshell_scores_sorted.csv")
df_kshell["month"] = pd.to_datetime(df_kshell["month"]).dt.to_period("M").astype(str)

with open("degree_centralities.pkl", "rb") as f:
    degree_centralities = pickle.load(f)

# DataFrame
records = []
for month, centrality_dict in degree_centralities.items():
    for user, centrality in centrality_dict.items():
        records.append({
            "user": user,
            "month": str(month),
            "degree_centrality": centrality
        })

df_degree = pd.DataFrame(records)


df = pd.merge(df_kshell, df_degree, on=["user", "month"], how="inner")

scaler = MinMaxScaler()
df[["kshell_scaled", "degree_scaled"]] = scaler.fit_transform(df[["kshell", "degree_centrality"]])

df["activity_score"] = 0.2 * df["kshell_scaled"] + 0.8 * df["degree_scaled"]

df_sorted = df.sort_values(by=["month", "activity_score"], ascending=[True, False])

df_sorted.to_csv("monthly_activity_scores.csv", index=False)
print("Saved combined activity scores to 'monthly_activity_scores.csv'")
