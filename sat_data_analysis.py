import pandas as pd
from matplotlib import pyplot as plt

sat_2012_df = pd.read_csv('2012_SAT_Results_20260624.csv')
sat_2010_df= pd.read_csv('SAT_(College_Board)_2010_School_Level_Results_20260624.csv')

print(sat_2012_df.info())
print(sat_2010_df.info())

sat_2010_df = sat_2010_df.rename(columns={'School Name': 'SCHOOL NAME'})


merge1 = sat_2012_df.merge(sat_2010_df, on='DBN', how='left')
merge2 = sat_2012_df.merge(sat_2010_df, on="SCHOOL NAME", how='left')

combined_df = merge1.combine_first(merge2)

# drop unnecessary columns
combined_df.drop(columns=['DBN_x', 'DBN_y', 'SCHOOL NAME', 'SCHOOL NAME_y'], axis=1, inplace=True)
#print(combined_df.info())

# rename column names
combined_df = combined_df.rename(columns={'Critical Reading Mean': 'Reading_Score_2010', 'Mathematics Mean': 'Maths_Score_2010',
                   'Num of SAT Test Takers': 'SAT_Test_Takers_2012', 'Number of Test Takers': 'SAT_Test_Takers_2010', 'SAT Critical Reading Avg. Score':
                                 'Reading_Score_2012', 'SAT Math Avg. Score': 'Maths_Score_2012', 'SAT Writing Avg. Score': 'Writing_Score_2012',
                                 'Writing Mean': 'Writing_Score_2010'})



# check distribution
#combined_df['Reading_Score_2010'].hist()
#combined_df['Writing_Score_2010'].hist()
#plt.show()

combined_df['SAT_Test_Takers_2012'] = pd.to_numeric(combined_df['SAT_Test_Takers_2012'], errors='coerce')
combined_df['SAT_Test_Takers_2010'] = pd.to_numeric(combined_df['SAT_Test_Takers_2010'], errors='coerce')
combined_df['Reading_Score_2012'] = pd.to_numeric(combined_df['Reading_Score_2012'], errors='coerce')
combined_df['Maths_Score_2012'] = pd.to_numeric(combined_df['Maths_Score_2012'], errors='coerce')
combined_df['Writing_Score_2012'] = pd.to_numeric(combined_df['Writing_Score_2012'], errors='coerce')

#combined_df['Writing_Score_2012'].hist()
#plt.show()


# impute NaN values
combined_df['Reading_Score_2010'] = combined_df['Reading_Score_2010'].fillna(combined_df['Reading_Score_2010'].median())
combined_df['Maths_Score_2010'] = combined_df['Maths_Score_2010'].fillna(combined_df['Maths_Score_2010'].mode()[0])
combined_df['SAT_Test_Takers_2010'] = combined_df['SAT_Test_Takers_2010'].fillna(combined_df['SAT_Test_Takers_2010'].mode()[0])
combined_df['Writing_Score_2010'] = combined_df['Writing_Score_2010'].fillna(combined_df['Writing_Score_2010'].median())
combined_df['SAT_Test_Takers_2012'] = combined_df['SAT_Test_Takers_2012'].fillna(combined_df['SAT_Test_Takers_2012'].median())
combined_df['Reading_Score_2012'] = combined_df['Reading_Score_2012'].fillna(combined_df['Reading_Score_2012'].median())
combined_df['Maths_Score_2012'] = combined_df['Maths_Score_2012'].fillna(combined_df['Maths_Score_2012'].median())
combined_df['Writing_Score_2012'] = combined_df['Writing_Score_2012'].fillna(combined_df['Writing_Score_2012'].median())


# convert numeric datatype

combined_df['SAT_Test_Takers_2012'] = combined_df['SAT_Test_Takers_2012'].astype(int)
combined_df['SAT_Test_Takers_2010'] = combined_df['SAT_Test_Takers_2010'].astype(int)
combined_df['Reading_Score_2012'] = combined_df['Reading_Score_2012'].astype(float)
combined_df['Maths_Score_2012'] = combined_df['Maths_Score_2012'].astype(float)
combined_df['Writing_Score_2012'] = combined_df['Writing_Score_2012'].astype(float)


print(combined_df.info())

corr_cols_2010 = ['Reading_Score_2010', 'Maths_Score_2010', 'Writing_Score_2010']
matrix_2010 = combined_df[corr_cols_2010].corr()
#print(matrix_2010)

corr_cols_2012 = ['Reading_Score_2012', 'Maths_Score_2012', 'Writing_Score_2012']
matrix_2012 = combined_df[corr_cols_2012].corr()

# scatter plots
import seaborn as sns
#sns.pairplot(combined_df[corr_cols_2010], kind='reg')
#sns.pairplot(combined_df[corr_cols_2012], kind='reg')
#plt.show()

# Performance by school adn year
score_per_year_df = pd.DataFrame({
    'Year': [2010, 2012]
})
score_per_year_df['Mean_Reading_Score'] = [combined_df['Reading_Score_2010'].mean(),combined_df['Reading_Score_2012'].mean()]
score_per_year_df['Mean_Writing_Score'] = [combined_df['Writing_Score_2010'].mean(),combined_df['Writing_Score_2012'].mean()]
score_per_year_df['Mean_Maths_Score'] = [combined_df['Maths_Score_2010'].mean(),combined_df['Maths_Score_2012'].mean()]

print(score_per_year_df)

# plot charts for performance per year
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# 3. Add global main title
fig.suptitle("Mean Score Trends by Year", y=1.05, fontsize=16, fontweight="bold")

# 4. Plot Reading Scores (Column 0)
sns.barplot(data=score_per_year_df, x="Year", y="Mean_Reading_Score", ax=axes[0], hue='Year')
axes[0].set_title("Reading Mean")
axes[0].set_ylabel("Mean Score")

# 5. Plot Writing Scores (Column 1)
sns.barplot(data=score_per_year_df, x="Year", y="Mean_Writing_Score", ax=axes[1], hue='Year')
axes[1].set_title("Writing Mean")
axes[1].set_ylabel("")  # Hide Y label since it shares with column 0

# 6. Plot Math Scores (Column 2)
sns.barplot(data=score_per_year_df, x="Year", y="Mean_Maths_Score", ax=axes[2], hue='Year')
axes[2].set_title("Maths Mean")
axes[2].set_ylabel("")  # Hide Y label since it shares with column 0

# 7. Clean up layout boundaries
plt.tight_layout()
#plt.show()


# Top performing schools 2010
score_school_2010_df = combined_df[['SCHOOL NAME_x', 'Reading_Score_2010', 'Writing_Score_2010', 'Maths_Score_2010']].copy()

reading_scores_2010_df = score_school_2010_df.groupby(['SCHOOL NAME_x'], as_index=False)['Reading_Score_2010'].mean()
writing_scores_2010_df = score_school_2010_df.groupby(['SCHOOL NAME_x'], as_index=False)['Writing_Score_2010'].mean()
maths_scores_2010_df = score_school_2010_df.groupby(['SCHOOL NAME_x'], as_index=False)['Maths_Score_2010'].mean()

sorted_reading_scores_2010_df = reading_scores_2010_df.sort_values(by=['Reading_Score_2010'])
sorted_writing_scores_2010_df = writing_scores_2010_df.sort_values(by=['Writing_Score_2010'])
sorted_maths_scores_2010_df = maths_scores_2010_df.sort_values(by=['Maths_Score_2010'])

top_2010_scores_df = pd.concat([sorted_reading_scores_2010_df, sorted_writing_scores_2010_df, sorted_maths_scores_2010_df], axis=1)
top_2010_scores_df = top_2010_scores_df.loc[:, ~top_2010_scores_df.columns.duplicated()]
top_2010_scores_df = top_2010_scores_df.head(5)
top_2010_scores_df['SCHOOL NAME_x'] = ['WORLD_CULTURES', 'PROSPECT_HEIGHTS', 'KINGSBRIDGE', 'EAST_SIDE', 'INTERNATIONAL_COMMUNITY']

#print(top_2010_scores_df['SCHOOL NAME_x'])


fig, axes = plt.subplots(1,3, figsize=(15, 5), sharey=True)

sns.barplot(data=top_2010_scores_df, x='SCHOOL NAME_x', y='Reading_Score_2010',ax=axes[0], hue='SCHOOL NAME_x', legend=True)
axes[0].set_title("Reading Mean")
axes[0].set_ylabel("Mean Score")
axes[0].set(xticklabels=[])
axes[0].tick_params(bottom=False)


sns.barplot(data=top_2010_scores_df, x='SCHOOL NAME_x', y='Writing_Score_2010',ax=axes[1], hue='SCHOOL NAME_x', legend=True)
axes[1].set_title("Writing Mean")
axes[1].set_ylabel("Mean Score")
axes[1].set(xticklabels=[])
axes[1].tick_params(bottom=False)

sns.barplot(data=top_2010_scores_df, x='SCHOOL NAME_x', y='Maths_Score_2010',ax=axes[2], hue='SCHOOL NAME_x', legend=True)
axes[2].set_title("Maths Mean")
axes[2].set_ylabel("Mean Score")
axes[2].set(xticklabels=[])
axes[2].tick_params(bottom=False)


plt.tight_layout()
plt.show()


