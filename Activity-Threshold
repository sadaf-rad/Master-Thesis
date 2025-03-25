import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the transaction data and user data
data_t = pd.read_csv("sarafu_txns_20200125-20210615.csv", on_bad_lines='skip', encoding='utf-8', delimiter=",")
df_users = pd.read_csv("sarafu_users_20210615.csv", on_bad_lines='skip', encoding='utf-8', delimiter=",")

# Convert timestamp to datetime and extract the week
data_t['timestamp'] = pd.to_datetime(data_t['timeset'], errors='coerce')
data_t['week'] = data_t['timestamp'].dt.isocalendar().week

# Function to calculate the Z-score for each user per week and return active users
def get_active_users_for_all_weeks():
    active_users_per_week = {}
    thresholds_per_week = {}

    # Loop through all weeks in the dataset
    for week_number in data_t['week'].unique():
        week_data = data_t[data_t['week'] == week_number]
        
        # Calculate transaction count per user for the specific week
        week_user_transaction_counts = pd.concat([week_data['source'], week_data['target']], axis=0).value_counts()
        
        # Calculate Z-score for each user based on their transaction count in that week
        mean_transactions = week_user_transaction_counts.mean()
        std_transactions = week_user_transaction_counts.std()
        
        # Calculate Z-scores
        week_user_z_scores = (week_user_transaction_counts - mean_transactions) / std_transactions
        
        # Set dynamic threshold: consider top 20% of Z-scores as active users (percentile-based)
        percentile_threshold = week_user_z_scores.quantile(0.8)  # Top 20% Z-scores
        
        # Store dynamic threshold for reference
        thresholds_per_week[week_number] = percentile_threshold
        
        # Get active users based on dynamic percentile threshold
        active_users = week_user_z_scores[week_user_z_scores > percentile_threshold]
        
        # Store active users for the week
        active_users_per_week[week_number] = active_users
    
    return active_users_per_week, thresholds_per_week

# Call the function to get active users for all weeks
all_active_users, all_thresholds = get_active_users_for_all_weeks()

# ------------------------------- Step 1: Print Dynamic Thresholds -------------------------------

# Print the dynamic thresholds for each week
print("Dynamic Thresholds per Week:")
for week, threshold in all_thresholds.items():
    print(f"Week {week}: Threshold = {threshold:.2f}")

# ------------------------------- Step 2: Active Users Demographics -------------------------------
# Flatten the dictionary to create a DataFrame
flattened_data = []
for week, users in all_active_users.items():
    for user, z_score in users.items():
        flattened_data.append([week, user, z_score])

# Create DataFrame from the flattened data
active_users_df = pd.DataFrame(flattened_data, columns=['Week', 'User', 'User Z-Scores'])

# ------------------------------- Step 3: Demographics for Active Users -------------------------------

# Filter the demographic data for the active users
active_user_demographics = df_users[df_users['xDAI_blockchain_address'].isin(active_users_df['User'])]

# ------------------------------- Step 4: Correlation Between Gender and Location -------------------------------
# Create a pivot table for gender and location
pivot_gender_location = active_user_demographics.pivot_table(index='xDAI_blockchain_address', 
                                                             columns='gender', 
                                                             aggfunc='size', fill_value=0)

# Calculate correlation between gender and location
gender_location_corr = active_user_demographics[['gender', 'area_name']].groupby(['gender', 'area_name']).size().unstack().fillna(0)
plt.figure(figsize=(10, 6))
sns.heatmap(gender_location_corr, annot=True, cmap="YlGnBu", cbar=True)
plt.title("Correlation Between Gender and Location for Active Users")
plt.tight_layout()
plt.savefig("gender_location_correlation.png")
plt.close()

# ------------------------------- Step 5: Correlation Among Gender, Location, and Business Type -------------------------------

# Create a pivot table for all three demographic features: gender, location, and business type
pivot_gender_location_job = active_user_demographics.pivot_table(index='xDAI_blockchain_address', 
                                                                columns=['gender', 'area_name', 'business_type'], 
                                                                aggfunc='size', fill_value=0)

# Plot the heatmap for the correlation between all three demographics
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_gender_location_job.corr(), annot=True, cmap="YlGnBu", cbar=True)
plt.title("Correlation Between Gender, Location, and Business Type for Active Users")
plt.tight_layout()
plt.savefig("gender_location_job_correlation.png")
plt.close()

# ------------------------------- Step 6: Display Active Users and Their Demographics -------------------------------
# Display the active users for each week along with their demographic information
for week, users in all_active_users.items():
    print(f"\nActive Users for Week {week}:")
    active_users = active_user_demographics[active_user_demographics['xDAI_blockchain_address'].isin(users.index)]
    print(active_users[['xDAI_blockchain_address', 'gender', 'area_name', 'business_type']])

# ------------------------------- Step 7: Identify Weeks with Highest Active Users in Demographics -------------------------------
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
