import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution_per_feature():
    print("Loading labeled feature dataset...")
    df = pd.read_csv('/home/s3986160/master-thesis/Retention/labeled_user_features_with_final_return_flag.csv')

    print("🔍 Filtering for 'retained' and 'churned' users only...")
    df = df[df['label'].isin(['retained', 'churned'])]

    # Drop non-feature columns
    exclude_columns = ['user', 'label', 'reactivated_after_churn']
    numeric_features = df.select_dtypes(include='number').columns.difference(exclude_columns)

    print(f" Generating distribution plots for {len(numeric_features)} features...")

    for feature in numeric_features:
        print(f" Plotting distribution for: {feature}")

        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=feature, hue='label', kde=True, stat='density', common_norm=False, palette='pastel', bins=30)
        plt.title(f'Distribution of {feature} by Retention Label')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(f'/home/s3986160/master-thesis/Retention/distribution_{feature}_by_label.png')
        plt.close()

    print("Distribution plots saved to /Retention/ folder.")

if __name__ == "__main__":
    plot_distribution_per_feature()
