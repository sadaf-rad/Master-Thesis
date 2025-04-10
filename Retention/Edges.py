import pandas as pd
import matplotlib.pyplot as plt

def analyze_churned_users_return_behavior():
    print(" Loading labeled user data...")
    labeled_df = pd.read_csv('/home/s3986160/master-thesis/Retention/labeled_user_features.csv')

    print(" Loading full transaction data...")
    df = pd.read_csv("sarafu_txns_20200125-20210615.csv", on_bad_lines='skip', encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    if 'source' not in df.columns:
        if 'ï»¿from' in df.columns:
            df.rename(columns={'ï»¿from': 'source'}, inplace=True)
        else:
            raise KeyError("Missing 'source' or 'from' column in transaction data.")

    df['user'] = df['source']
    df['month'] = df['timestamp'].dt.to_period('M')

    # Filter for Jan–June 2021
    df = df[(df['timestamp'] >= '2021-01-01') & (df['timestamp'] < '2021-07-01')]

    print(" Creating user-month activity matrix...")
    user_months = df.groupby(['user', 'month']).size().reset_index(name='tx_count')
    activity_matrix = user_months.pivot(index='user', columns='month', values='tx_count').fillna(0)
    activity_matrix = activity_matrix.sort_index(axis=1)
    activity_binary = activity_matrix.applymap(lambda x: 1 if x > 0 else 0)

    expected_months = pd.period_range("2021-01", "2021-06", freq='M')
    for m in expected_months:
        if m not in activity_binary.columns:
            activity_binary[m] = 0
    activity_binary = activity_binary[expected_months]

    print(" Checking reactivation after 4-month inactivity...")

    def reactivated_after_inactivity(row):
        activity = ''.join(str(int(x)) for x in row)
        if '0000' in activity:
            first_0000 = activity.index('0000')
            rest = activity[first_0000 + 4:]
            return '1' in rest
        return False

    activity_binary['reactivated_after_churn'] = activity_binary.apply(
        reactivated_after_inactivity, axis=1
    )

    # Merge churn label
    label_map = labeled_df.set_index('user')['label']
    activity_binary['label'] = activity_binary.index.map(label_map).fillna('unknown')

    # Save extended data
    final_df = activity_binary.reset_index()[['user', 'label', 'reactivated_after_churn']]
    merged = labeled_df.merge(final_df[['user', 'reactivated_after_churn']], on='user', how='left')
    merged['reactivated_after_churn'] = merged['reactivated_after_churn'].fillna(False)

    print(" Saved updated labeled data to: /home/s3986160/master-thesis/Retention/labeled_user_features_with_final_return_flag.csv")
    merged.to_csv('/home/s3986160/master-thesis/Retention/labeled_user_features_with_final_return_flag.csv', index=False)

    # Optional bar chart
    print(" Generating bar chart...")
    return_counts = merged[merged['label'] == 'churned']['reactivated_after_churn'].value_counts()

    # Always safe: Map values True/False to readable labels
    return_counts.index = return_counts.index.map({False: 'Did Not Return', True: 'Returned'})
    return_counts = return_counts.reindex(['Did Not Return', 'Returned']).fillna(0)

    return_counts.plot(kind='bar', color='skyblue', title='Reactivation of Churned Users')
    plt.ylabel('Number of Users')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('/home/s3986160/master-thesis/Retention/churned_users_reactivation_barplot.png')
    print("📈 Saved bar chart to churned_users_reactivation_barplot.png")

if __name__ == "__main__":
    analyze_churned_users_return_behavior()
