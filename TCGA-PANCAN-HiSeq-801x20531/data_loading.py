import pandas as pd

filename = 'data.csv'
df = pd.read_csv(filename)
df2 = pd.read_csv('labels.csv')

# print(df.head())
# print(df.shape)
# print('*'*20)
#
# print(df2.head())
# print(df2.shape)

# for _ in df.index.values:
#     row = df.loc[_]

df.insert(1, list(df2.columns.values)[1], df2.Class)
df.drop(list(df.columns.values)[0], axis=1, inplace=True)

print(df.head())

df.to_csv('loaded_' + filename, index=False)


