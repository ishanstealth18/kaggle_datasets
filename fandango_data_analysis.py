import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

score_df = pd.read_csv('fandango_score_comparison.csv')
scrape_df = pd.read_csv('fandango_scrape.csv')
movie_ratings = pd.read_csv('movie_ratings_16_17.csv')

combined_df = pd.merge(score_df, scrape_df, how='inner')

print(combined_df.info())

# see data distribution for Fandango stars and Fandango ratings
before_analysis_data = combined_df[['FILM', 'Fandango_Stars', 'Fandango_Ratingvalue', 'Fandango_votes', 'Fandango_Difference']]
after_analysis_data = movie_ratings[['movie', 'year', 'fandango']]

print(after_analysis_data.info())

before_analysis_data = before_analysis_data[before_analysis_data['FILM'].str.contains('2015')]

print(before_analysis_data['Fandango_Ratingvalue'].unique())

print(after_analysis_data['fandango'].unique())


plt.style.use('fivethirtyeight')
ax_before = sns.kdeplot(data=before_analysis_data, x = 'Fandango_Ratingvalue', fill=True, label='Before Analysis')
ax_after = sns.kdeplot(data=after_analysis_data, x = 'fandango', fill=True, label='After Analysis')

tick_positions = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
ax_before.set_xticks(tick_positions)
ax_after.set_xticks(tick_positions)

plt.title('KDE Distribution')
plt.legend()
plt.xlabel('Ratings')
plt.ylabel('No of Records')
plt.show()

# Frequency tables
print(before_analysis_data['Fandango_Ratingvalue'].value_counts(dropna=True))
print(after_analysis_data['fandango'].value_counts(dropna=True))

# frequency table after normalization
print(before_analysis_data['Fandango_Ratingvalue'].value_counts(normalize=True, dropna=True))
print(after_analysis_data['fandango'].value_counts(normalize=True, dropna=True))

# Compare stats for both data

before_analysis_mean = before_analysis_data['Fandango_Ratingvalue'].mean()
before_analysis_median = before_analysis_data['Fandango_Ratingvalue'].median()
before_analysis_mode = before_analysis_data['Fandango_Ratingvalue'].mode()

after_analysis_mean = after_analysis_data['fandango'].mean()
after_analysis_median = after_analysis_data['fandango'].median()
after_analysis_mode = after_analysis_data['fandango'].mode()


stats = ('Mean', 'Median', 'Mode')
x = np.arange(len(stats))

y1 = [before_analysis_mean, before_analysis_median, before_analysis_mode]
y2 = [after_analysis_mean, after_analysis_median, after_analysis_mode]
width = 0.40

plt.bar(x-0.2, y1, width, label='2015')
plt.bar(x+0.2, y2, width, label='2016')

plt.ylabel('Stars')
plt.show()



