import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Preprocessing
# Load the transaction data and user data
data_t = pd.read_csv("sarafu_txns_20200125-20210615.csv", on_bad_lines='skip', encoding='utf-8', delimiter=",")
df_users = pd.read_csv("sarafu_users_20210615.csv", on_bad_lines='skip', encoding='utf-8', delimiter=",")

# Reset index to avoid duplicate index issues
data_t.reset_index(drop=True, inplace=True)

# Create the 'user' column by concatenating 'source' and 'target'
data_t['user'] = pd.concat([data_t['source'], data_t['target']], axis=0, ignore_index=True)

# Step 1: Calculate transaction counts per user over the entire network (aggregate over all weeks)
user_transaction_counts = data_t['user'].value_counts()

# Step 2: Apply Log-Transformation if the data is skewed (log transformation of the transaction counts)
log_transaction_counts = np.log1p(user_transaction_counts)  # log(x+1) transformation

# Step 3: Calculate Z-scores for all users based on log-transformed transaction count
mean_transactions = log_transaction_counts.mean()
std_transactions = log_transaction_counts.std()
user_z_scores = (log_transaction_counts - mean_transactions) / std_transactions

# Step 4: Define the dynamic threshold (90th percentile of Z-scores)
threshold = user_z_scores.quantile(0.9)  # 90th percentile for a higher threshold
print(f"Dynamic Threshold (90th percentile): {threshold:.2f}")

# Step 5: Filter active users who have Z-scores greater than the threshold
active_users = user_z_scores[user_z_scores > threshold]

# Print the number of active users
print(f"\nNumber of Active Users: {len(active_users)}")

# Filter the demographic data for the active users
active_user_demographics = df_users[df_users['xDAI_blockchain_address'].isin(active_users.index)]

# Print Active Users' Blockchain Addresses
print("\nActive Users' Blockchain Addresses:")
print(active_user_demographics['xDAI_blockchain_address'])

# Create a pivot table for job and location
pivot_job_location = active_user_demographics.pivot_table(index='xDAI_blockchain_address', 
                                                          columns='business_type', 
                                                          aggfunc='size', fill_value=0)

# Calculate correlation between job and location
job_location_corr = active_user_demographics[['business_type', 'area_name']].groupby(['business_type', 'area_name']).size().unstack().fillna(0)

# Plot heatmap for job-location correlation
plt.figure(figsize=(10, 6))
sns.heatmap(job_location_corr, annot=True, cmap="YlGnBu", cbar=True)
plt.title("Correlation Between Job (Business Type) and Location")
plt.tight_layout()
plt.savefig("jl.png")  # Save with a short name (job-location)
plt.close()

# Create a pivot table for location and gender
pivot_location_gender = active_user_demographics.pivot_table(index='xDAI_blockchain_address', 
                                                             columns='area_name', 
                                                             aggfunc='size', fill_value=0)

# Calculate correlation between location and gender
location_gender_corr = active_user_demographics[['area_name', 'gender']].groupby(['area_name', 'gender']).size().unstack().fillna(0)

# Plot heatmap for location-gender correlation
plt.figure(figsize=(10, 6))
sns.heatmap(location_gender_corr, annot=True, cmap="YlGnBu", cbar=True)
plt.title("Correlation Between Location and Gender")
plt.tight_layout()
plt.savefig("lg.png")  # Save with a short name (location-gender)
plt.close()

# Create a pivot table for gender and job (business type)
pivot_gender_job = active_user_demographics.pivot_table(index='xDAI_blockchain_address', 
                                                        columns='gender', 
                                                        aggfunc='size', fill_value=0)

# Calculate correlation between gender and job (business type)
gender_job_corr = active_user_demographics[['gender', 'business_type']].groupby(['gender', 'business_type']).size().unstack().fillna(0)

# Plot heatmap for gender-job correlation
plt.figure(figsize=(10, 6))
sns.heatmap(gender_job_corr, annot=True, cmap="YlGnBu", cbar=True)
plt.title("Correlation Between Gender and Job (Business Type)")
plt.tight_layout()
plt.savefig("gj.png")  # Save with a short name (gender-job)
plt.close()

# Create a pivot table for gender, location, and job (business type)
pivot_all = active_user_demographics.pivot_table(index='xDAI_blockchain_address', 
                                                 columns=['gender', 'area_name', 'business_type'], 
                                                 aggfunc='size', fill_value=0)

# Plot the heatmap for the correlation between all three demographics
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_all.corr(), annot=True, cmap="YlGnBu", cbar=True)
plt.title("Correlation Between Gender, Location, and Job (Business Type)")
plt.tight_layout()
plt.savefig("glj.png")  # Save with a short name (gender-location-job)
plt.close()

# Display the active users for each week along with their demographic information
print(f"\nActive Users and Their Demographics:")
print(active_user_demographics[['xDAI_blockchain_address', 'gender', 'area_name', 'business_type']])

# Find the week with the highest number of active females, males, etc.
demographics_counts_per_week = active_user_demographics.groupby(['gender', 'week']).size().unstack().fillna(0)
highest_female_week = demographics_counts_per_week['Female'].idxmax()
highest_male_week = demographics_counts_per_week['Male'].idxmax()

# Display the week with the most active users by gender
print(f"\nWeek with most active females: Week {highest_female_week}")
print(f"Week with most active males: Week {highest_male_week}")

# Also check by location
demographics_location_per_week = active_user_demographics.groupby(['area_name', 'week']).size().unstack().fillna(0)
highest_location_week = demographics_location_per_week.idxmax()

# Display the location with the most active users
print(f"\nWeek with most active location: Week {highest_location_week}")
