import numpy as np
import pandas as pd

original_df= pd.read_csv('StarWars.csv', encoding='windows-1254')


# Renaming column names for better understanding
original_df.rename(columns={'Have you seen any of the 6 films in the Star Wars franchise?': 'Movies Watched?'}, inplace=True)
original_df.rename(columns={'Do you consider yourself to be a fan of the Star Wars film franchise?': 'Star Wars Fan'}, inplace=True)
original_df.rename(columns={'Which of the following Star Wars films have you seen? Please select all that apply.': 'Star Wars: Episode I  The Phantom Menace'}, inplace=True)
original_df.rename(columns={'Unnamed: 4': 'Star Wars: Episode II  Attack of the Clones'}, inplace=True)
original_df.rename(columns={'Unnamed: 5': 'Star Wars: Episode III  Revenge of the Sith'}, inplace=True)
original_df.rename(columns={'Unnamed: 6': 'Star Wars: Episode IV  A New Hope'}, inplace=True)
original_df.rename(columns={'Unnamed: 7': 'Star Wars: Episode V The Empire Strikes Back'}, inplace=True)
original_df.rename(columns={'Unnamed: 8': 'Star Wars: Episode VI Return of the Jedi'}, inplace=True)



original_df.rename(columns={'Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.': 'Rank: Star Wars: Episode I'}, inplace=True)
original_df.rename(columns={'Unnamed: 10': 'Rank: Star Wars: Episode II'}, inplace=True)
original_df.rename(columns={'Unnamed: 11': 'Rank: Star Wars: Episode III'}, inplace=True)
original_df.rename(columns={'Unnamed: 12': 'Rank: Star Wars: Episode IV'}, inplace=True)
original_df.rename(columns={'Unnamed: 13': 'Rank: Star Wars: Episode V'}, inplace=True)
original_df.rename(columns={'Unnamed: 14': 'Rank: Star Wars: Episode VI'}, inplace=True)


original_df.rename(columns={'Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.': 'Han Solo'}, inplace=True)
original_df.rename(columns={'Unnamed: 16': 'Luke Skywalker'}, inplace=True)
original_df.rename(columns={'Unnamed: 17': 'Princess Leia Organa'}, inplace=True)
original_df.rename(columns={'Unnamed: 18': 'Anakin Skywalker'}, inplace=True)
original_df.rename(columns={'Unnamed: 19': 'Obi Wan Kenobi'}, inplace=True)
original_df.rename(columns={'Unnamed: 20': 'Emperor Palpatine'}, inplace=True)
original_df.rename(columns={'Unnamed: 21': 'Darth Vader'}, inplace=True)
original_df.rename(columns={'Unnamed: 22': 'Lando Calrissian'}, inplace=True)
original_df.rename(columns={'Unnamed: 23': 'Boba Fett'}, inplace=True)
original_df.rename(columns={'Unnamed: 24': 'C-3P0'}, inplace=True)
original_df.rename(columns={'Unnamed: 25': 'R2 D2'}, inplace=True)
original_df.rename(columns={'Unnamed: 26': 'Jar Jar Binks'}, inplace=True)
original_df.rename(columns={'Unnamed: 27': 'Padme Amidala'}, inplace=True)
original_df.rename(columns={'Unnamed: 28': 'Yoda'}, inplace=True)


original_df.rename(columns={'Are you familiar with the Expanded Universe?': 'Expanded Universe Familiarity'}, inplace=True)
original_df.rename(columns={'Which character shot first?': 'First Shot Character'}, inplace=True)
original_df.rename(columns={'Do you consider yourself to be a fan of the Expanded Universe?Œæ': 'Expanded Character Fan'}, inplace=True)
original_df.rename(columns={'Do you consider yourself to be a fan of the Star Trek franchise?': 'Star Trek Franchise Fan'}, inplace=True)


# Star Wars Fan is Boolean column, so we will use Mode to impute values
original_df['Star Wars Fan'] = original_df['Star Wars Fan'].replace('Response', np.nan)
star_wars_fan_mode= original_df['Star Wars Fan'].mode()[0]
original_df.fillna({'Star Wars Fan': star_wars_fan_mode}, inplace=True)


original_df['Star Wars: Episode I  The Phantom Menace'] = original_df['Star Wars: Episode I  The Phantom Menace'].replace('Star Wars: Episode I  The Phantom Menace', 1)
original_df.fillna({'Star Wars: Episode I  The Phantom Menace': 0}, inplace=True)

original_df['Star Wars: Episode II  Attack of the Clones'] = original_df['Star Wars: Episode II  Attack of the Clones'].replace('Star Wars: Episode II  Attack of the Clones', 1)
original_df.fillna({'Star Wars: Episode II  Attack of the Clones': 0}, inplace=True)

original_df['Star Wars: Episode III  Revenge of the Sith'] = original_df['Star Wars: Episode III  Revenge of the Sith'].replace('Star Wars: Episode III  Revenge of the Sith', 1)
original_df.fillna({'Star Wars: Episode III  Revenge of the Sith': 0}, inplace=True)

original_df['Star Wars: Episode IV  A New Hope'] = original_df['Star Wars: Episode IV  A New Hope'].replace('Star Wars: Episode IV  A New Hope', 1)
original_df.fillna({'Star Wars: Episode IV  A New Hope': 0}, inplace=True)

original_df['Star Wars: Episode V The Empire Strikes Back'] = original_df['Star Wars: Episode V The Empire Strikes Back'].replace('Star Wars: Episode V The Empire Strikes Back', 1)
original_df.fillna({'Star Wars: Episode V The Empire Strikes Back': 0}, inplace=True)

original_df['Star Wars: Episode VI Return of the Jedi'] = original_df['Star Wars: Episode VI Return of the Jedi'].replace('Star Wars: Episode VI Return of the Jedi', 1)
original_df.fillna({'Star Wars: Episode VI Return of the Jedi': 0}, inplace=True)



original_df['Rank: Star Wars: Episode I'] = original_df['Rank: Star Wars: Episode I'].replace({'Star Wars: Episode I  The Phantom Menace': np.nan})
rank_episode1_mode = original_df['Rank: Star Wars: Episode I'].mode()[0]
original_df.fillna({'Rank: Star Wars: Episode I': rank_episode1_mode}, inplace=True)


original_df['Star Wars: Episode II  Attack of the Clones'] = original_df['Star Wars: Episode II  Attack of the Clones'].replace({'Star Wars: Episode II  Attack of the Clones': np.nan})
rank_episode1_mode = original_df['Rank: Star Wars: Episode II'].mode()[0]
original_df.fillna({'Rank: Star Wars: Episode II': rank_episode1_mode}, inplace=True)


original_df['Star Wars: Episode III  Revenge of the Sith'] = original_df['Star Wars: Episode III  Revenge of the Sith'].replace({'Star Wars: Episode III  Revenge of the Sith': np.nan})
rank_episode1_mode = original_df['Rank: Star Wars: Episode III'].mode()[0]
original_df.fillna({'Rank: Star Wars: Episode III': rank_episode1_mode}, inplace=True)


original_df['Star Wars: Episode V The Empire Strikes Back'] = original_df['Star Wars: Episode V The Empire Strikes Back'].replace({'Star Wars: Episode V The Empire Strikes Back': np.nan})
rank_episode1_mode = original_df['Rank: Star Wars: Episode IV'].mode()[0]
original_df.fillna({'Rank: Star Wars: Episode IV': rank_episode1_mode}, inplace=True)

original_df['Star Wars: Episode V The Empire Strikes Back'] = original_df['Star Wars: Episode V The Empire Strikes Back'].replace({'Star Wars: Episode V The Empire Strikes Back': np.nan})
rank_episode1_mode = original_df['Rank: Star Wars: Episode V'].mode()[0]
original_df.fillna({'Rank: Star Wars: Episode V': rank_episode1_mode}, inplace=True)

original_df['Star Wars: Episode VI Return of the Jedi'] = original_df['Star Wars: Episode VI Return of the Jedi'].replace({'Star Wars: Episode VI Return of the Jedi': np.nan})
rank_episode1_mode = original_df['Rank: Star Wars: Episode VI'].mode()[0]
original_df.fillna({'Rank: Star Wars: Episode VI': rank_episode1_mode}, inplace=True)


han_solo_map = {'Very favorably':6, 'Somewhat favorably':5, 'Neither favorably nor unfavorably (neutral)':4, 'Somewhat unfavorably':3,
                 'Unfamiliar (N/A)':2, 'Very unfavorably':1}

original_df['Han Solo'] = original_df['Han Solo'].replace({'Han Solo': np.nan})
han_solo_mode = original_df['Han Solo'].mode()[0]
original_df.fillna({'Han Solo': han_solo_mode}, inplace=True)
original_df.replace({'Han Solo': han_solo_map}, inplace=True)

Luke_Skywalker_map = {'Very favorably':6, 'Somewhat favorably':5, 'Neither favorably nor unfavorably (neutral)':4, 'Somewhat unfavorably':3,
                 'Unfamiliar (N/A)':2, 'Very unfavorably':1}
original_df['Luke Skywalker'] = original_df['Luke Skywalker'].replace({'Luke Skywalker': np.nan})
Luke_Skywalker_mode = original_df['Luke Skywalker'].mode()[0]
original_df.fillna({'Luke Skywalker': Luke_Skywalker_mode}, inplace=True)
original_df.replace({'Luke Skywalker': Luke_Skywalker_map}, inplace=True)


original_df['Princess Leia Organa'] = original_df['Princess Leia Organa'].replace({'Princess Leia Organa': np.nan})
Princess_Leia_Organa_mode = original_df['Princess Leia Organa'].mode()[0]
original_df.fillna({'Princess Leia Organa': Princess_Leia_Organa_mode}, inplace=True)
original_df.replace({'Princess Leia Organa': han_solo_map}, inplace=True)

original_df['Anakin Skywalker'] = original_df['Anakin Skywalker'].replace({'Anakin Skywalker': np.nan})
Anakin_Skywalker_mode = original_df['Anakin Skywalker'].mode()[0]
original_df.fillna({'Anakin Skywalker': Anakin_Skywalker_mode}, inplace=True)
original_df.replace({'Anakin Skywalker': han_solo_map}, inplace=True)

original_df['Obi Wan Kenobi'] = original_df['Obi Wan Kenobi'].replace({'Obi Wan Kenobi': np.nan})
Obi_Wan_kenobi_mode = original_df['Obi Wan Kenobi'].mode()[0]
original_df.fillna({'Obi Wan Kenobi': Obi_Wan_kenobi_mode}, inplace=True)
original_df.replace({'Obi Wan Kenobi': han_solo_map}, inplace=True)


original_df['Emperor Palpatine'] = original_df['Emperor Palpatine'].replace({'Emperor Palpatine': np.nan})
Emperor_Palpatine_mode = original_df['Emperor Palpatine'].mode()[0]
original_df.fillna({'Emperor Palpatine': Emperor_Palpatine_mode}, inplace=True)
original_df.replace({'Emperor Palpatine': han_solo_map}, inplace=True)


original_df['Darth Vader'] = original_df['Darth Vader'].replace({'Darth Vader': np.nan})
Darth_Vader_mode = original_df['Darth Vader'].mode()[0]
original_df.fillna({'Darth Vader': Darth_Vader_mode}, inplace=True)
original_df.replace({'Darth Vader': han_solo_map}, inplace=True)

original_df['Lando Calrissian'] = original_df['Lando Calrissian'].replace({'Lando Calrissian': np.nan})
Lando_Calrissian_mode = original_df['Lando Calrissian'].mode()[0]
original_df.fillna({'Lando Calrissian': Lando_Calrissian_mode}, inplace=True)
original_df.replace({'Lando Calrissian': han_solo_map}, inplace=True)


original_df['Boba Fett'] = original_df['Boba Fett'].replace({'Boba Fett': np.nan})
Boba_Fett_mode = original_df['Boba Fett'].mode()[0]
original_df.fillna({'Boba Fett': Boba_Fett_mode}, inplace=True)
original_df.replace({'Boba Fett': han_solo_map}, inplace=True)

original_df['C-3P0'] = original_df['C-3P0'].replace({'C-3P0': np.nan})
C_3P0_mode = original_df['C-3P0'].mode()[0]
original_df.fillna({'C-3P0': C_3P0_mode}, inplace=True)
original_df.replace({'C-3P0': han_solo_map}, inplace=True)

original_df['R2 D2'] = original_df['R2 D2'].replace({'R2 D2': np.nan})
R2_D2_mode = original_df['R2 D2'].mode()[0]
original_df.fillna({'R2 D2': R2_D2_mode}, inplace=True)
original_df.replace({'R2 D2': han_solo_map}, inplace=True)

original_df['Jar Jar Binks'] = original_df['Jar Jar Binks'].replace({'Jar Jar Binks': np.nan})
Jar_Jar_Binks_mode = original_df['Jar Jar Binks'].mode()[0]
original_df.fillna({'Jar Jar Binks': Jar_Jar_Binks_mode}, inplace=True)
original_df.replace({'Jar Jar Binks': han_solo_map}, inplace=True)

original_df['Padme Amidala'] = original_df['Padme Amidala'].replace({'Padme Amidala': np.nan})
Padme_Amidala_mode = original_df['Padme Amidala'].mode()[0]
original_df.fillna({'Padme Amidala': Padme_Amidala_mode}, inplace=True)
original_df.replace({'Padme Amidala': han_solo_map}, inplace=True)


original_df['Yoda'] = original_df['Yoda'].replace({'Yoda': np.nan})
Yoda_mode = original_df['Yoda'].mode()[0]
original_df.fillna({'Yoda': Yoda_mode}, inplace=True)
original_df.replace({'Yoda': han_solo_map}, inplace=True)


original_df['First Shot Character'] = original_df['First Shot Character'].replace({'Response': np.nan})
original_df['First Shot Character'] = original_df['First Shot Character'].replace({'I don\'t understand this question': np.nan})
first_shot_mode = original_df['First Shot Character'].mode()[0]
original_df.fillna({'First Shot Character': first_shot_mode}, inplace=True)


original_df['Expanded Universe Familiarity'] = original_df['Expanded Universe Familiarity'].replace({'Response': np.nan})
xpanded_universe_mode = original_df['Expanded Universe Familiarity'].mode()[0]
original_df.fillna({'Expanded Universe Familiarity': xpanded_universe_mode}, inplace=True)

original_df['Expanded Character Fan'] = original_df['Expanded Character Fan'].replace({'Response': np.nan})
expanded_char_mode = original_df['Expanded Character Fan'].mode()[0]
original_df.fillna({'Expanded Character Fan': expanded_char_mode}, inplace=True)


original_df['Star Trek Franchise Fan'] = original_df['Star Trek Franchise Fan'].replace({'Response': np.nan})
star_trek_franchise_mode = original_df['Star Trek Franchise Fan'].mode()[0]
original_df.fillna({'Star Trek Franchise Fan': star_trek_franchise_mode}, inplace=True)


original_df['Gender'] = original_df['Gender'].replace({'Response': np.nan})
gender_mode = original_df['Gender'].mode()[0]
original_df.fillna({'Gender': gender_mode}, inplace=True)


original_df['Age'] = original_df['Age'].replace({'Response': np.nan})
age_mode = original_df['Age'].mode()[0]
original_df.fillna({'Age': age_mode}, inplace=True)


original_df['Household Income'] = original_df['Household Income'].replace({'Response': np.nan})
household_mode = original_df['Household Income'].mode()[0]
original_df.fillna({'Household Income': household_mode}, inplace=True)


original_df['Education'] = original_df['Education'].replace({'Response': np.nan})
education_mode = original_df['Education'].mode()[0]
original_df.fillna({'Education': education_mode}, inplace=True)

original_df['Location (Census Region)'] = original_df['Location (Census Region)'].replace({'Response': np.nan})
location_mode = original_df['Location (Census Region)'].mode()[0]
original_df.fillna({'Location (Census Region)': location_mode}, inplace=True)


movies_watched_mode = original_df['Movies Watched?'].mode()[0]
original_df['Movies Watched?'].replace({'Response': movies_watched_mode}, inplace=True)

# One hot encoding
movies_watched_dummy = pd.get_dummies(original_df['Movies Watched?'], dtype=int, prefix='movies_watched', drop_first=True)
original_df = pd.concat([original_df, movies_watched_dummy], axis=1)
original_df.drop(columns=['Movies Watched?'], axis=1, inplace=True)


star_wars_fan_dummy = pd.get_dummies(original_df['Star Wars Fan'], dtype=int, prefix='star_wars_fan', drop_first=True)
original_df = pd.concat([original_df, star_wars_fan_dummy], axis=1)
original_df.drop(columns=['Star Wars Fan'], axis=1, inplace=True)

original_df['Star Wars: Episode I  The Phantom Menace'] = original_df['Star Wars: Episode I  The Phantom Menace'].astype(int)
original_df['Star Wars: Episode II  Attack of the Clones'] = original_df['Star Wars: Episode II  Attack of the Clones'].astype(int)
original_df['Star Wars: Episode III  Revenge of the Sith'] = original_df['Star Wars: Episode III  Revenge of the Sith'].astype(int)
original_df['Star Wars: Episode IV  A New Hope'] = original_df['Star Wars: Episode IV  A New Hope'].astype(int)
original_df['Star Wars: Episode V The Empire Strikes Back'] = original_df['Star Wars: Episode V The Empire Strikes Back'].astype(int)
original_df['Star Wars: Episode VI Return of the Jedi'] = original_df['Star Wars: Episode VI Return of the Jedi'].astype(int)


# Make Rank columns categorical (ordinal)
original_df['Rank: Star Wars: Episode I'] = pd.Categorical(original_df['Rank: Star Wars: Episode I'], ordered=True)
reversed_order = original_df["Rank: Star Wars: Episode I"].cat.categories[::-1]
original_df["Rank: Star Wars: Episode I"] = original_df["Rank: Star Wars: Episode I"].cat.reorder_categories(reversed_order, ordered=True)


rank_episode2_mode = original_df['Rank: Star Wars: Episode II'].mode()[0]
original_df['Rank: Star Wars: Episode II'].replace({'Star Wars: Episode II  Attack of the Clones': rank_episode2_mode}, inplace=True)
original_df['Rank: Star Wars: Episode II'] = pd.Categorical(original_df['Rank: Star Wars: Episode II'], ordered=True)
reversed_order = original_df["Rank: Star Wars: Episode II"].cat.categories[::-1]
original_df["Rank: Star Wars: Episode II"] = original_df["Rank: Star Wars: Episode II"].cat.reorder_categories(reversed_order, ordered=True)


rank_episode3_mode = original_df['Rank: Star Wars: Episode III'].mode()[0]
original_df['Rank: Star Wars: Episode III'].replace({'Star Wars: Episode III  Revenge of the Sith': rank_episode3_mode}, inplace=True)
original_df['Rank: Star Wars: Episode III'] = pd.Categorical(original_df['Rank: Star Wars: Episode III'], ordered=True)
reversed_order = original_df["Rank: Star Wars: Episode III"].cat.categories[::-1]
original_df["Rank: Star Wars: Episode III"] = original_df["Rank: Star Wars: Episode III"].cat.reorder_categories(reversed_order, ordered=True)


rank_episode4_mode = original_df['Rank: Star Wars: Episode IV'].mode()[0]
original_df['Rank: Star Wars: Episode IV'].replace({'Star Wars: Episode IV  A New Hope': rank_episode4_mode}, inplace=True)
original_df['Rank: Star Wars: Episode IV'] = pd.Categorical(original_df['Rank: Star Wars: Episode IV'], ordered=True)
reversed_order = original_df["Rank: Star Wars: Episode IV"].cat.categories[::-1]
original_df["Rank: Star Wars: Episode IV"] = original_df["Rank: Star Wars: Episode IV"].cat.reorder_categories(reversed_order, ordered=True)


rank_episode5_mode = original_df['Rank: Star Wars: Episode V'].mode()[0]
original_df['Rank: Star Wars: Episode V'].replace({'Star Wars: Episode V The Empire Strikes Back': rank_episode5_mode}, inplace=True)
original_df['Rank: Star Wars: Episode V'] = pd.Categorical(original_df['Rank: Star Wars: Episode V'], ordered=True)
reversed_order = original_df["Rank: Star Wars: Episode V"].cat.categories[::-1]
original_df["Rank: Star Wars: Episode V"] = original_df["Rank: Star Wars: Episode V"].cat.reorder_categories(reversed_order, ordered=True)


rank_episode6_mode = original_df['Rank: Star Wars: Episode VI'].mode()[0]
original_df['Rank: Star Wars: Episode VI'].replace({'Star Wars: Episode VI Return of the Jedi': rank_episode5_mode}, inplace=True)
original_df['Rank: Star Wars: Episode VI'] = pd.Categorical(original_df['Rank: Star Wars: Episode VI'], ordered=True)
reversed_order = original_df["Rank: Star Wars: Episode VI"].cat.categories[::-1]
original_df["Rank: Star Wars: Episode VI"] = original_df["Rank: Star Wars: Episode VI"].cat.reorder_categories(reversed_order, ordered=True)


original_df['Han Solo'] = pd.Categorical(original_df['Han Solo'], ordered=True)
original_df['Luke Skywalker'] = pd.Categorical(original_df['Luke Skywalker'], ordered=True)
original_df['Princess Leia Organa'] = pd.Categorical(original_df['Princess Leia Organa'], ordered=True)
original_df['Anakin Skywalker'] = pd.Categorical(original_df['Anakin Skywalker'], ordered=True)
original_df['Obi Wan Kenobi'] = pd.Categorical(original_df['Obi Wan Kenobi'], ordered=True)
original_df['Emperor Palpatine'] = pd.Categorical(original_df['Emperor Palpatine'], ordered=True)
original_df['Darth Vader'] = pd.Categorical(original_df['Darth Vader'], ordered=True)
original_df['Lando Calrissian'] = pd.Categorical(original_df['Lando Calrissian'], ordered=True)
original_df['Boba Fett'] = pd.Categorical(original_df['Boba Fett'], ordered=True)
original_df['C-3P0'] = pd.Categorical(original_df['C-3P0'], ordered=True)
original_df['R2 D2'] = pd.Categorical(original_df['R2 D2'], ordered=True)
original_df['Jar Jar Binks'] = pd.Categorical(original_df['Jar Jar Binks'], ordered=True)
original_df['Padme Amidala'] = pd.Categorical(original_df['Padme Amidala'], ordered=True)
original_df['Yoda'] = pd.Categorical(original_df['Yoda'], ordered=True)


first_shot_dummy = pd.get_dummies(original_df['First Shot Character'], drop_first=True, dtype=int, prefix='first_shot')
original_df = pd.concat([original_df, first_shot_dummy],axis=1)
original_df.drop(columns=['First Shot Character'], axis=1, inplace=True)

expanded_universe_dummy = pd.get_dummies(original_df['Expanded Universe Familiarity'], drop_first=True, dtype=int, prefix='expanded_universe')
original_df = pd.concat([original_df, expanded_universe_dummy],axis=1)
original_df.drop(columns=['Expanded Universe Familiarity'], axis=1, inplace=True)

expanded_char_dummy = pd.get_dummies(original_df['Expanded Character Fan'], drop_first=True, dtype=int, prefix='expanded_character')
original_df = pd.concat([original_df, expanded_char_dummy],axis=1)
original_df.drop(columns=['Expanded Character Fan'], axis=1, inplace=True)


star_trek_fan_dummy = pd.get_dummies(original_df['Star Trek Franchise Fan'], drop_first=True, dtype=int, prefix='star_trek_fan')
original_df = pd.concat([original_df, star_trek_fan_dummy],axis=1)
original_df.drop(columns=['Star Trek Franchise Fan'], axis=1, inplace=True)


gender_dummy = pd.get_dummies(original_df['Gender'], drop_first=True, dtype=int, prefix='gender')
original_df = pd.concat([original_df, gender_dummy],axis=1)
original_df.drop(columns=['Gender'], axis=1, inplace=True)

original_df['Age'] = pd.Categorical(original_df['Age'], ordered=True)

income_map= {'0 - 24999': 1, '25000 - 49999': 2, '50000 - 99999': 3, '100000 - 149999': 4, '150000': 5}
original_df['Household Income'] = original_df['Household Income'].str.replace('$', '').str.replace('+', '').str.replace(',', '')
original_df['Household Income'] = original_df['Household Income'].map(income_map)
original_df['Household Income'] = pd.Categorical(original_df['Household Income'], ordered=True)


education_map = {'Less than high school degree':1, 'High school degree':2, 'Some college or Associate degree':3, 'Bachelor degree':3,
                 'Graduate degree':4}
original_df['Education'] = original_df['Education'].map(education_map)
original_df['Education'] = pd.Categorical(original_df['Education'], ordered=True)

original_df = original_df.rename(columns={'Location (Census Region)': 'Location'})

location_dummy = pd.get_dummies(original_df['Location'], drop_first=True, dtype=int, prefix='location')
original_df = pd.concat([original_df, location_dummy],axis=1)
original_df.drop(columns=['Location'], axis=1, inplace=True)

from matplotlib import pyplot as plt
import seaborn as sns

# Most  liked film
print(original_df.info())
rank_cols = ['Rank: Star Wars: Episode I', 'Rank: Star Wars: Episode II', 'Rank: Star Wars: Episode III', 'Rank: Star Wars: Episode IV'
             ,'Rank: Star Wars: Episode V', 'Rank: Star Wars: Episode VI']
fig, axes = plt.subplots(2, 3, figsize=(15, 8)) # Increased height to 10 for better readability

# Fix: Use axes.flatten() to loop through all 6 subplots sequentially
for i, col in enumerate(rank_cols):
    ax = axes.flatten()[i]
    sns.countplot(data=original_df, x=col, ax=ax, hue='gender_Male')
    ax.set_title(f'Count of {col}')
    ax.bar_label(ax.containers[0])

plt.tight_layout() # Fixes overlapping text between rows
plt.show()


# most seen film
movie_seen_df = original_df[['Star Wars: Episode I  The Phantom Menace', 'Star Wars: Episode II  Attack of the Clones',
                             'Star Wars: Episode III  Revenge of the Sith', 'Star Wars: Episode IV  A New Hope', 'Star Wars: Episode V The Empire Strikes Back',
                             'Star Wars: Episode VI Return of the Jedi', 'movies_watched_Yes']]

print(len(movie_seen_df[(movie_seen_df['Star Wars: Episode I  The Phantom Menace'] == 1) & (movie_seen_df['movies_watched_Yes'] == 1)].index))
print(len(movie_seen_df[(movie_seen_df['Star Wars: Episode II  Attack of the Clones'] == 1) & (movie_seen_df['movies_watched_Yes'] == 1)].index))
print(len(movie_seen_df[(movie_seen_df['Star Wars: Episode III  Revenge of the Sith'] == 1) & (movie_seen_df['movies_watched_Yes'] == 1)].index))
print(len(movie_seen_df[(movie_seen_df['Star Wars: Episode IV  A New Hope'] == 1) & (movie_seen_df['movies_watched_Yes'] == 1)].index))
print(len(movie_seen_df[(movie_seen_df['Star Wars: Episode V The Empire Strikes Back'] == 1) & (movie_seen_df['movies_watched_Yes'] == 1)].index))
print(len(movie_seen_df[(movie_seen_df['Star Wars: Episode VI Return of the Jedi'] == 1) & (movie_seen_df['movies_watched_Yes'] == 1)].index))







