import pandas as pd
import matplotlib as plot

df = pd.read_csv('https://raw.githubusercontent.com/ysdyt/pandas_tutorial/master/data/lunch_box.csv', sep=',')

print(df.head())

print(df.tail())


print('dataframeの行数・列数の確認==>\n', df.shape, '\n')
print('indexの確認==>\n', df.index, '\n')
print('columnの確認==>\n', df.columns, '\n')
print('dataframeの各列のデータ型を確認==>\n', df.dtypes, '\n')

print(df[['name', 'kcal']][100:106], '\n')

print(df.loc[100], '\n')


print(df.iloc[[1,2,4],[0,2]], '\n')


print(df[ df['kcal'] > 450 ], '\n')

a = df[['name', 'kcal']].query('kcal > 450 and name == "豚肉の生姜焼"')
print(a, '\n')

rem = df['remarks'].unique()
print(rem, '\n')

print(len(df['datetime'].unique()), '\n')

print(df.describe())

df.set_index('datetime', inplace=True)
print(df.head(), '\n')

df.rename(columns={'y': 'sales'}, inplace=True)
print(df.head(), '\n')

df.sort_values(['sales', 'temperature'], ascending=True).head()
print(df.head(), '\n')

df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
print(df.index)

print(df.sort_index().head())
