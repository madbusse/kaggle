import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.api.types import is_numeric_dtype


def visualize_relationships(train_path='train.csv'):
    # Load and preprocess
    df = pd.read_csv(train_path)
    if 'crmid' in df.columns:
        df = df.drop(columns=['crmid'])
    # Parse date into components
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df.drop(columns=['date'])

    features = [c for c in df.columns if c != 'consent']

    # For each feature, create a plot
    for feat in features:
        plt.figure()
        if is_numeric_dtype(df[feat]):
            # Numeric: boxplot of feature by consent
            data0 = df[df['consent'] == 0][feat].dropna()
            data1 = df[df['consent'] == 1][feat].dropna()
            plt.boxplot([data0, data1], labels=['No Consent', 'Consent'])
            plt.ylabel(feat)
            plt.title(f'{feat} distribution by Consent')
        else:
            # Categorical: bar chart of mean consent per category
            means = df.groupby(feat)['consent'].mean()
            plt.bar(means.index.astype(str), means.values)
            plt.ylabel('Mean Consent')
            plt.title(f'{feat} vs Mean Consent')
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        # Save each figure for reference
        filename = f'{feat}_vs_consent.png'
        plt.savefig(filename)
        print(f'Saved plot for {feat} to {filename}')
        plt.close()

    print('All feature visualizations completed.')


if __name__ == '__main__':
    visualize_relationships()
