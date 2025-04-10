import pandas as pd

def label_user_retention():
    print(" Loading transaction data...")
    df = pd.read_csv("sarafu_txns_20200125-20210615.csv", on_bad_lines='skip', encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    if 'source' not in df.columns:
        if 'ifrom' in df.columns:
            df.rename(columns={'ifrom': 'source'}, inplace=True)
        else:
            raise KeyError("Missing 'source' or 'from' column in transaction data.")

    print(" Filtering transactions from Jan to June 2021...")
    df_2021 = df[(df['timestamp'] >= '2021-01-01') & (df['timestamp'] < '2021-07-01')].copy()
    df_2021['month'] = df_2021['timestamp'].dt.to_period('M')

    #  Identify users by blockchain address
    df_2021['user'] = df_2021['source']
    user_months = df_2021.groupby(['user', 'month']).size().reset_index(name='tx_count')

    print(" Building activity matrix...")
    activity_matrix = user_months.pivot(index='user', columns='month', values='tx_count').fillna(0)
    activity_matrix = activity_matrix.sort_index(axis=1)
    activity_binary = activity_matrix.applymap(lambda x: 1 if x > 0 else 0)

    #  Fill any missing months with 0s
    expected_months = pd.period_range("2021-01", "2021-06", freq='M')
    for m in expected_months:
        if m not in activity_binary.columns:
            activity_binary[m] = 0
    activity_binary = activity_binary[expected_months]

    print(" Labeling users as churned or retained...")

    def label_retention(row):
        activity = row.values.astype(int)
        total_active = activity.sum()
        binary_str = ''.join(map(str, activity))
        has_4_consec_inactive = '0000' in binary_str
        if total_active < 3 and has_4_consec_inactive:
            return 'churned'
        else:
            return 'retained'

    activity_binary['label'] = activity_binary.apply(label_retention, axis=1)

    print(" Loading early behavior features and merging labels...")
    early_df = pd.read_csv('/home/s3986160/master-thesis/Retention/early_user_features.csv')
    labeled_df = early_df.merge(activity_binary['label'], on='user', how='left')
    labeled_df['label'] = labeled_df['label'].fillna('unknown')

    print(" Saving to labeled_user_features.csv...")
    labeled_df.to_csv('labeled_user_features.csv', index=False)

    print(" Done! Users labeled and saved.")

if __name__ == "__main__":
    label_user_retention()
