import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing 

# Demographic Distribution Based on Burstiness for Top 20% users

data_t = pd.read_csv("sarafu_txns_20200125-20210615.csv", on_bad_lines='skip', encoding='utf-8', delimiter=",", low_memory=False)

df_users = pd.read_csv("sarafu_users_20210615.csv", on_bad_lines='skip', encoding='utf-8', delimiter=",", low_memory=False)

data_t['timestamp'] = pd.to_datetime(data_t['timeset'], errors='coerce')
data_t['month'] = data_t['timestamp'].dt.to_period('M')

data_t = data_t.dropna(subset=['timestamp'])

data_t_2020 = data_t[data_t['timestamp'].dt.year == 2020]
data_t_2021 = data_t[data_t['timestamp'].dt.year == 2021]

transaction_counts_2020 = data_t_2020['month'].value_counts()
transaction_counts_2021 = data_t_2021['month'].value_counts()

top_month_2020 = transaction_counts_2020.idxmax()
top_month_2021 = transaction_counts_2021.idxmax()

top_month_2020_name = top_month_2020.strftime('%B %Y')
top_month_2021_name = top_month_2021.strftime('%B %Y')

top_month_addresses_2020 = pd.concat([data_t_2020[data_t_2020['month'] == top_month_2020]['source'],
                                      data_t_2020[data_t_2020['month'] == top_month_2020]['target']]).unique()

top_month_addresses_2021 = pd.concat([data_t_2021[data_t_2021['month'] == top_month_2021]['source'],
                                      data_t_2021[data_t_2021['month'] == top_month_2021]['target']]).unique()

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

peak_activity_2020 = transaction_counts_2020.max()
peak_activity_2021 = transaction_counts_2021.max()

avg_activity_2020 = transaction_counts_2020.mean()

burstiness_2020 = peak_activity_2020 / avg_activity_2020
burstiness_2021 = peak_activity_2021 / avg_activity_2021

print(f"Burstiness for 2020: {burstiness_2020}")
print(f"Burstiness for 2021: {burstiness_2021}")

def visualize_burstiness(df_top_users, year, top_month_name, burstiness):
    print(f"\nVisualizing Demographics for {year} - {top_month_name} with Burstiness: {burstiness}")

    # Gender Burstiness
    if 'gender' in df_top_users.columns:
        df_top_users['gender'].value_counts().plot(kind='bar', title=f'Gender Distribution for Top 20% Users in {top_month_name}')
        plt.xlabel("Gender")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    # Location Burstiness
    if 'area_name' in df_top_users.columns:
        df_top_users['area_name'].value_counts().plot(kind='bar', title=f'Location Distribution for Top 20% Users in {top_month_name}')
        plt.xlabel("Location")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    # Job Burstiness
    if 'business_type' in df_top_users.columns:
        df_top_users['business_type'].value_counts().plot(kind='bar', title=f'Business Type Distribution for Top 20% Users in {top_month_name}')
        plt.xlabel("Business Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

visualize_burstiness(df_top_users_2020, 2020, top_month_2020_name, burstiness_2020)

visualize_burstiness(df_top_users_2021, 2021, top_month_2021_name, burstiness_2021)
