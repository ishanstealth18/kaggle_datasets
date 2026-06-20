import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

original_df = pd.read_csv('data_science_Queryresults.csv')
print(original_df.info())

original_df.dropna(axis=0, inplace=True)

original_df['Tags'] = original_df['Tags'].str.replace('<', ' ')
original_df['Tags'] = original_df['Tags'].str.replace('>', ' ')
original_df['Tags'] = original_df['Tags'].str.strip()


original_df.loc[original_df["TagName"].str.contains("sql", case=False, na=False), "TagName"] = "sql"
original_df.loc[original_df["TagName"].str.contains("python", case=False, na=False), "TagName"] = "python"


relevant_tagnames = original_df[(original_df['TagName'].str.contains('python')) | (original_df['TagName'].str.contains('sql')) |
                                (original_df['TagName'].str.contains('artificial')) |
                                (original_df['TagName'].str.contains('statistics')) | (original_df['TagName'].str.contains('learn')) ]

#print(list(relevant_tagnames['TagName'].unique()))

#print(relevant_tagnames['TagName'].shape)

relevant_tagnames['CreationDate'] = pd.to_datetime(relevant_tagnames['CreationDate'])
relevant_tagnames['year'] = relevant_tagnames['CreationDate'].dt.year
relevant_tagnames['month'] = relevant_tagnames['CreationDate'].dt.month
print(relevant_tagnames['month'].unique())



question_count = relevant_tagnames.groupby(['TagName', 'month'], as_index=False).agg(question_count = ('PostTypeId', 'count'))

#print(question_count.head())

sns.barplot(data=question_count, x='TagName', y='question_count')
plt.show()

sns.barplot(data=question_count, x='month', y='question_count')
plt.show()


