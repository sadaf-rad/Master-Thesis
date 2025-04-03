import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing

#Demographic Distribution based on Entropy for top 20% users
#2020-2021
data_t = pd.read_csv("sarafu_txns_20200125-20210615.csv", on_bad_lines='skip', encoding='utf-8', delimiter=",", low_memory=False)

df_users = pd.read_csv("sarafu_users_20210615.csv", on_bad_lines='skip', encoding='utf-8', delimiter=",", low_memory=False)

data_t['timestamp'] = pd.to_datetime(data_t['timeset'], errors='coerce')
data_t['month'] = data_t['timestamp'].dt.to_period('M')

data_t = data_t.dropna(subset=['timestamp'])

data_t_2020 = data_t[data_t['timestamp'].dt.year == 2020]
data_t_2021 = data_t[data_t['timestamp'].dt.year == 2021]

transaction_counts_2020 = data_t_2020['month'].value_counts()
transaction_counts_2021 = data_t_2021['month'].value_counts()

top_months_2020 = transaction_counts_2020.head()
top_months_2021 = transaction_counts_2021.head()

top_month_addresses_2020 = pd.concat([data_t_2020[data_t_2020['month'].isin(top_months_2020.index)]['source'],
                                      data_t_2020[data_t_2020['month'].isin(top_months_2020.index)]['target']]).unique()

top_month_addresses_2021 = pd.concat([data_t_2021[data_t_2021['month'].isin(top_months_2021.index)]['source'],
                                      data_t_2021[data_t_2021['month'].isin(top_months_2021.index)]['target']]).unique()

address_transaction_count_2020 = pd.concat([data_t_2020[data_t_2020['source'].isin(top_month_addresses_2020)]['source'],
                                            data_t_2020[data_t_2020['target'].isin(top_month_addresses_2020)]['target']]).value_counts()
address_transaction_count_2021 = pd.concat([data_t_2021[data_t_2021['source'].isin(top_month_addresses_2021)]['source'],
                                            data_t_2021[data_t_2021['target'].isin(top_month_addresses_2021)]['target']]).value_counts()

top_20_percent_addresses_2020 = address_transaction_count_2020.head(int(len(address_transaction_count_2020) * 0.2)).index
top_20_percent_addresses_2021 = address_transaction_count_2021.head(int(len(address_transaction_count_2021) * 0.2)).index

df_top_users_2020 = df_users[df_users['old_POA_blockchain_address'].isin(top_20_percent_addresses_2020) |
                              df_users['xDAI_blockchain_address'].isin(top_20_percent_addresses_2020)]

df_top_users_2021 = df_users[df_users['old_POA_blockchain_address'].isin(top_20_percent_addresses_2021) |
                              df_users['xDAI_blockchain_address'].isin(top_20_percent_addresses_2021)]

merged_data_2020 = pd.merge(df_top_users_2020, data_t_2020, how='inner', left_on=['old_POA_blockchain_address', 'xDAI_blockchain_address'],
                             right_on=['source', 'target'])

merged_data_2021 = pd.merge(df_top_users_2021, data_t_2021, how='inner', left_on=['old_POA_blockchain_address', 'xDAI_blockchain_address'],
                             right_on=['source', 'target'])

# Visualize the demographics based on entropy
def calculate_entropy(data):
    """Calculate the entropy of categorical data."""
    value_counts = data.value_counts()
    probabilities = value_counts / len(data)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Function to calculate entropy
def visualize_entropy_for_demographics(df_top_users, year):
    print(f"\nEntropy for {year}:")

    # Gender Entropy
    if 'gender' in df_top_users.columns:
        entropy_gender = calculate_entropy(df_top_users['gender'])
        print(f"  Entropy based on gender: {entropy_gender}")

    # Location Entropy
    if 'area_name' in df_top_users.columns:
        entropy_location = calculate_entropy(df_top_users['area_name'])
        print(f"  Entropy based on location: {entropy_location}")

    # Job Entropy
    if 'business_type' in df_top_users.columns:
        entropy_job = calculate_entropy(df_top_users['business_type'])
        print(f"  Entropy based on job (business type): {entropy_job}")

    # Visualizing the demographics
    demographics = ['gender', 'area_name', 'business_type']
    for demographic in demographics:
        if demographic in df_top_users.columns:
            plt.figure(figsize=(8, 6))
            df_top_users[demographic].value_counts().plot(kind='bar', title=f'{demographic.capitalize()} Distribution for Top 20% Users in {year}')
            plt.xlabel(demographic.capitalize())
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.show()

# Visualizing entropy for 2020
visualize_entropy_for_demographics(df_top_users_2020, 2020)

# Visualizing entropy for 2021
visualize_entropy_for_demographics(df_top_users_2021, 2021)


