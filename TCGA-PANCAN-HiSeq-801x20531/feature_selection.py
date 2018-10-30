import pandas as pd
import numpy as np
from sklearn import feature_selection as fs
from sklearn import preprocessing

filename = 'data.csv'
df = pd.read_csv('loaded_' + filename)

y = df.select_dtypes(include=['object'])

print(df.dtypes)
# num_df = df[df.dtypes != 'float64']
x = df.select_dtypes(include=['float64'])
x.dropna(axis=0, inplace=False)

# print(num_df.head())

scaler = preprocessing.StandardScaler()
x = pd.DataFrame(data=scaler.fit_transform(x).reshape(-1, len(x.columns)),
                 columns=x.columns)


"""
Select a threshold and remove features that have a variance lower than that threshold.
"""


def filter_by_variance(data, threshold=0.0):
    filter = fs.VarianceThreshold(threshold)
    filter.fit_transform(data)


"""
Remove features that have poor correlation across replicates.
"""


def filter_by_replicate_correlation(data):

    features = list(data.columns[1:])

    for feature in features:
        replicates = {}
        for c in data.Metadata_broad_sample.unique():
            replicates[c] = []

        for _ in data.index.values:
            row = data.loc[_]

            replicates[row['Metadata_broad_sample']].extend([row[feature]])

        replicate_data = pd.DataFrame(data=dict([(k, pd.Series(v)) for k, v in replicates.items()]))

        correlation_matrix = replicate_data.corr()
        correlation_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

        correlation_matrix = np.extract(~np.isnan(correlation_matrix),
                                        correlation_matrix)

        cor = np.mean(correlation_matrix)
        print('Replicate Correlation:', cor)

        if abs(cor) is np.nan or abs(cor) < 0.005:
            print('Excluded:', feature)
            data.drop(feature, axis=1, inplace=True)
        else:
            print('Included:', feature)


"""
The feature-feature correlation matrix is computed,
and pairs with a correlation exceeding a given 
threshold are identified iteratively.
"""


def filter_by_correlated_features(data):
    numeric_data = data

    feature_feature_correlation = numeric_data.corr()

    for feature in numeric_data.columns:
        if np.mean(feature_feature_correlation[feature]) > 0.95:
            print('Excluded:', feature)
            data.drop(feature, axis=1, inplace=True)


filter_by_variance(x)
print('Filtered by variance.')


filter_by_correlated_features(x)
print('Filtered by correlated features.')

x = pd.concat([y, x], axis=1)

print(x.head())
print('Sample Size:', df.shape[0])

try:
    x.drop(['Unnamed: 0'], axis=1, inplace=True)
except (ValueError, KeyError) as e:
    pass


x.to_csv('fs_' + filename, index=False)









