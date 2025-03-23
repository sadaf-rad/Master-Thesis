import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import Preprocessing
G = Preprocessing.G

#Weighte Degree Distribution
weighted_in_degrees = {node: sum(data['weight'] for _, _, data in G.in_edges(node, data=True)) for node in G.nodes()}
weighted_out_degrees = {node: sum(data['weight'] for _, _, data in G.out_edges(node, data=True)) for node in G.nodes()}

weighted_in_counts = Counter(weighted_in_degrees.values())
weighted_out_counts = Counter(weighted_out_degrees.values())

x_w_in, y_w_in = zip(*sorted(weighted_in_counts.items()))
x_w_out, y_w_out = zip(*sorted(weighted_out_counts.items()))

plt.figure(figsize=(6, 4))
plt.scatter(x_w_in, y_w_in, color='green', alpha=0.7)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Weighted Degree")
plt.ylabel("Number of Nodes")
plt.title(" Weighted In-Degree Distribution")
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(x_w_out, y_w_out, color='purple', alpha=0.7)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Weighted Degree")
plt.ylabel("Number of Nodes")
plt.title("Weighted Out-Degree Distribution")
plt.show()

# Normal Degree Distribution
in_degree_values = np.array(list(dict(G.in_degree()).values()))
out_degree_values = np.array(list(dict(G.out_degree()).values()))

in_degree_counts = Counter(in_degree_values)
out_degree_counts = Counter(out_degree_values)

x_in, y_in = zip(*sorted(in_degree_counts.items()))
x_out, y_out = zip(*sorted(out_degree_counts.items()))

plt.figure(figsize=(6, 4))
plt.scatter(x_in, y_in, color='blue', alpha=0.7)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Degree")
plt.ylabel("Number of Nodes")
plt.title("In-Degree Distribution")
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(x_out, y_out, color='red', alpha=0.7)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Degree")
plt.ylabel("Number of Nodes")
plt.title("Out-Degree Distribution")
plt.show()

# Demographics distribution for top 20% active users - in-degree

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df_txns = pd.read_csv('sarafu_txns_20200125-20210615.csv')
df_users = pd.read_csv('sarafu_users_20210615.csv')

in_degree = pd.concat([df_txns['source'], df_txns['target']], axis=0).value_counts()

top_20_percent_users = in_degree.head(int(0.2 * len(in_degree)))

df_users["xDAI_blockchain_address"] = df_users["xDAI_blockchain_address"].astype(str).str.strip().str.lower()
top_in_degree_users_fixed = pd.DataFrame({
    'blockchain_address': pd.concat([df_txns['source'], df_txns['target']], axis=0).unique()
})
top_in_degree_users_fixed["blockchain_address"] = top_in_degree_users_fixed["blockchain_address"].astype(str).str.strip().str.lower()

df_users_filtered = df_users[df_users['xDAI_blockchain_address'].isin(top_in_degree_users_fixed["blockchain_address"])]

#  Gender Distribution
gender_dist = df_users_filtered['gender'].value_counts(normalize=True) * 100

location_dist = df_users_filtered['area_name'].value_counts(normalize=True).head(10) * 100

job_sector_dist = df_users_filtered['business_type'].value_counts(normalize=True) * 100

specific_job_dist = df_users_filtered['held_roles'].value_counts(normalize=True) * 100

df_txns['timeset'] = pd.to_datetime(df_txns['timeset'], errors='coerce')
df_txns['year'] = df_txns['timeset'].dt.year
df_txns['month'] = df_txns['timeset'].dt.month

df_txns_filtered = df_txns[df_txns['source'].isin(df_users_filtered['xDAI_blockchain_address']) |
                            df_txns['target'].isin(df_users_filtered['xDAI_blockchain_address'])]

transaction_period_dist = df_txns_filtered.groupby(['year', 'month']).size().reset_index(name='count')

# 📌 Display Results
print(f"Gender Distribution of Top 20% Users:\n{gender_dist}\n")
print(f"Location Distribution of Top 20% Users:\n{location_dist}\n")
print(f"Job Sector Distribution of Top 20% Users:\n{job_sector_dist}\n")
print(f"Specific Job Titles Distribution of Top 20% Users:\n{specific_job_dist}\n")
print(f"Transaction Year and Month Distribution:\n{transaction_period_dist}\n")


# 1. Gender Distribution Plot
plt.figure(figsize=(8, 6))
sns.barplot(x=gender_dist.index, y=gender_dist.values, palette='Set2')
plt.title("Gender Distribution of Top 20% Users")
plt.xlabel("Gender")
plt.ylabel("Percentage")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=location_dist.index, y=location_dist.values, palette='Set3')
plt.title("Location Distribution of Top 20% Users (Top 10 Locations)")
plt.xlabel("Location")
plt.ylabel("Percentage")
plt.xticks(rotation=45)

# 3. Job Sector Distribution Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=job_sector_dist.index, y=job_sector_dist.values, palette='Set1')
plt.title("Job Sector Distribution of Top 20% Users")
plt.xlabel("Job Sector")
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=specific_job_dist.index, y=specific_job_dist.values, palette='Set1')
plt.title("Specific Job Titles Distribution of Top 20% Users")
plt.xlabel("Job Title")
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
sns.lineplot(x=transaction_period_dist['month'], y=transaction_period_dist['count'], hue=transaction_period_dist['year'], palette='coolwarm')
plt.title("Transaction Year and Month Distribution of Top 20% Users")
plt.xlabel("Month")
plt.ylabel("Number of Transactions")
plt.show()



# Demographics distribution for top 20% active users - out-degree


out_degree = df_txns['source'].value_counts()

top_20_percent_users_out = out_degree.head(int(0.2 * len(out_degree)))

print(f"Number of users in the top 20% based on out-degree: {len(top_20_percent_users_out)}")

top_out_degree_users_fixed = pd.DataFrame({
    'blockchain_address': df_txns['source'].unique()
})

df_users["xDAI_blockchain_address"] = df_users["xDAI_blockchain_address"].astype(str).str.strip().str.lower()
top_out_degree_users_fixed["blockchain_address"] = top_out_degree_users_fixed["blockchain_address"].astype(str).str.strip().str.lower()

missing_addresses_out = top_out_degree_users_fixed[~top_out_degree_users_fixed["blockchain_address"].isin(df_users["xDAI_blockchain_address"])]

print(f"🔍 Missing Blockchain Addresses in Users Dataset for Top 20% Out-Degree Users: {len(missing_addresses_out)}")

df_users_filtered_out = df_users[df_users['xDAI_blockchain_address'].isin(top_out_degree_users_fixed["blockchain_address"])]

print("Filtered User Dataset Sample (based on blockchain address matching - Out-Degree):")
print(df_users_filtered_out[['id', 'xDAI_blockchain_address', 'gender', 'area_name', 'held_roles']].head())

gender_dist_out = df_users_filtered_out['gender'].value_counts(normalize=True) * 100

location_dist_out = df_users_filtered_out['area_name'].value_counts(normalize=True).head(10) * 100

job_sector_dist_out = df_users_filtered_out['business_type'].value_counts(normalize=True) * 100

specific_job_dist_out = df_users_filtered_out['held_roles'].value_counts(normalize=True) * 100

df_txns['timeset'] = pd.to_datetime(df_txns['timeset'], errors='coerce')
df_txns['year'] = df_txns['timeset'].dt.year
df_txns['month'] = df_txns['timeset'].dt.month

df_txns_filtered_out = df_txns[df_txns['source'].isin(df_users_filtered_out['xDAI_blockchain_address']) |
                                df_txns['target'].isin(df_users_filtered_out['xDAI_blockchain_address'])]

transaction_period_dist_out = df_txns_filtered_out.groupby(['year', 'month']).size().reset_index(name='count')

print(f"Gender Distribution of Top 20% Out-Degree Users:\n{gender_dist_out}\n")
print(f"Location Distribution of Top 20% Out-Degree Users:\n{location_dist_out}\n")
print(f"Job Sector Distribution of Top 20% Out-Degree Users:\n{job_sector_dist_out}\n")
print(f"Specific Job Titles Distribution of Top 20% Out-Degree Users:\n{specific_job_dist_out}\n")
print(f"Transaction Year and Month Distribution for Top 20% Out-Degree Users:\n{transaction_period_dist_out}\n")


# 1. Gender Distribution Plot
plt.figure(figsize=(8, 6))
sns.barplot(x=gender_dist_out.index, y=gender_dist_out.values, palette='Set2')
plt.title("Gender Distribution of Top 20% Out-Degree Users")
plt.xlabel("Gender")
plt.ylabel("Percentage")
plt.show()

# 2. Location Distribution Plot (Top 10 locations)
plt.figure(figsize=(10, 6))
sns.barplot(x=location_dist_out.index, y=location_dist_out.values, palette='Set3')
plt.title("Location Distribution of Top 20% Out-Degree Users (Top 10 Locations)")
plt.xlabel("Location")
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.show()

# 3. Job Sector Distribution Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=job_sector_dist_out.index, y=job_sector_dist_out.values, palette='Set1')
plt.title("Job Sector Distribution of Top 20% Out-Degree Users")
plt.xlabel("Job Sector")
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.show()

# 4. Specific Job Titles Distribution Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=specific_job_dist_out.index, y=specific_job_dist_out.values, palette='Set1')
plt.title("Specific Job Titles Distribution of Top 20% Out-Degree Users")
plt.xlabel("Job Title")
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.show()

# 5. Transaction Year and Month Distribution Plot
plt.figure(figsize=(8, 6))
sns.lineplot(x=transaction_period_dist_out['month'], y=transaction_period_dist_out['count'], hue=transaction_period_dist_out['year'], palette='coolwarm')
plt.title("Transaction Year and Month Distribution of Top 20% Out-Degree Users")
plt.xlabel("Month")
plt.ylabel("Number of Transactions")
plt.show()
