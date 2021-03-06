import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline

pd.__version__

# Taking a look at the Movies dataset
# This data shows the movies based on their title and the year of release

movies = pd.read_csv('data-wrangling-exercises/data/titles.csv')
movies.info()

movies.head()

# Taking a look at the Cast dataset
# This data shows the cast (actors, actresses, supporting roles) for each movie
#
# The attribute n basically tells the importance of the cast role, lower the number, more important the role.
# Supporting cast usually don't have any value for n
cast = pd.read_csv('data-wrangling-exercises/data/cast.csv')
cast.info()
cast.head(10)

# Taking a look at the Release dataset
# This data shows details of when each movie was release in each country with the release date
release_dates = pd.read_csv('data-wrangling-exercises/data/release_dates.csv', parse_dates=['date'], infer_datetime_format=True)
release_dates.info()
release_dates.head()

# Section I - Basic Querying, Filtering and Transformations
# What is the total number of movies?
len(movies)

# List all Batman movies ever made
batman_df = movies[movies.title == 'Batman']
print('Total Batman Movies:', len(batman_df))
batman_df

# List all Batman movies ever made - the right approach
batman_df = movies[movies.title.str.contains('Batman', case=False)]
print('Total Batman Movies:', len(batman_df))
batman_df.head(10)

# Display the top 15 Batman movies in the order they were released
batman_df.sort_values(by=['year'], ascending=True).iloc[:15]

# Section I - Q1 : List all the 'Harry Potter' movies from the most recent to the earliest
hpdf = movies[movies.title.str.contains('Harry Potter', case=False)]
print('All Harry Potter Movies:', len(hpdf))
print(hpdf.head())

# How many movies were made in the year 2017?
print(len(movies[movies.year == 2017]))

# Section I - Q2 : How many movies were made in the year 2015?
print(len(movies[movies.year == 2015]))

# Section I - Q3 : How many movies were made from 2000 till 2018?
print(len(movies[(movies.year >= 2000) & (movies.year <= 2018)]))

# Section I - Q4: How many movies are titled "Hamlet"?
hamletMovies = movies[movies.title == 'Hamlet']
print(len(hamletMovies))

# Section I - Q5: List all movies titled "Hamlet"
# The movies should only have been released on or after the year 2000
# Display the movies based on the year they were released (earliest to most recent)
print(hamletMovies[hamletMovies.year >= 2000].sort_values(by=['year'], ascending=True))

# Section I - Q6: How many roles in the movie "Inception" are of the supporting cast (extra credits)
# supporting cast are NOT ranked by an "n" value (NaN)
# check for how to filter based on nulls
# Option 1
supportCast = cast[(cast.title == 'Inception') & (pd.isnull(cast.n))]
print(len(supportCast))

# Option 2 - better
inceptionDf = cast[cast.title == 'Inception']
print(len(inceptionDf[inceptionDf.n.isnull()]))

# Section I - Q7: How many roles in the movie "Inception" are of the main cast
# main cast always have an 'n' value
# Option
#   mainCast = cast[(cast.title == 'Inception') & ~(pd.isnull(cast.n))]
#   len(mainCast)

print(len(inceptionDf[~inceptionDf.n.isnull()]))

# Section I - Q8: Show the top ten cast (actors\actresses) in the movie "Inception"
# support cast always have an 'n' value
# remember to sort!
print(inceptionDf[inceptionDf.n < 11].sort_values(by='n', ascending=True))
# topCast = (cast[(cast.title == 'Inception') & ~(pd.isnull(cast.n))].sort_values(by='n', ascending=True)).iloc[:10]
# print(topCast)


# Section I - Q9:
# (A) List all movies where there was a character 'Albus Dumbledore'
dumbledoreDf = cast[cast.character.str.match('Albus Dumbledore')]
print(dumbledoreDf.title)
# (B) Now modify the above to show only the actors who played the character 'Albus Dumbledore'
# For Part (B) remember the same actor might play the same role in multiple movies
print(dumbledoreDf.name.unique)

# Section I - Q10:
# (A) How many roles has 'Keanu Reeves' played throughout his career?
keanuReevesDf = cast[cast.name.str.match('Keanu Reeves')]
print(len(keanuReevesDf))
# (B) List the leading roles that 'Keanu Reeves' played on or after 1999 in order by year.
print(keanuReevesDf[keanuReevesDf.year >= 1999].sort_values(by='year', ascending=True))

# Section I - Q11:
# (A) List the total number of actor and actress roles available from 1950 - 1960
print(len(cast[(cast.year >= 1950) & (cast.year <= 1960)]))
print(len(cast[cast.year.between(1950, 1960)]))
# Group by
print(
    cast[cast.year.between(1950, 1960)][['type', 'name']].groupby('type').count().reset_index().rename({'name': 'freq'}, axis=1))

# (B) List the total number of actor and actress roles available from 2007 - 2017
print(len(cast[(cast.year >= 2007) & (cast.year <= 2017)]))
print(len(cast[cast.year.between(2007, 2017)]))
# Group by
print(
    cast[cast.year.between(2007, 2017)][['type', 'name']].groupby('type').count().reset_index().rename({'name': 'freq'}, axis=1))

# Section I - Q12:
# (A) List the total number of leading roles available from 2000 to present
print(len(cast[(cast.year >= 2000) & (cast.n == 1)]))
# (B) List the total number of non-leading roles available from 2000 - present (exclude support cast)
print(len(cast[(cast.n.notnull()) & (cast.year >= 2000) & (cast.n > 2)]))
# (C) List the total number of support\extra-credit roles available from 2000 - present
print(len(cast[(cast.year > 2000) & (cast.n != 1)]))


##################################################################################

# Section II - Aggregations, Transformations and Visualizations
# What are the top ten most common movie names of all time?
topTenCommonMovieNamesDf = movies.title.value_counts()[:10]

# Plot the top ten common movie names of all time
topTenCommonMovieNamesDf.plot(kind='barh')

# Section II - Q1: Which years in the 2000s saw the most movies released? (Show top 3)
top3=movies[movies.year == 2000].value_counts()[:3]
print(top3)

# Section II - Q2: # Plot the total number of films released per-decade (1890, 1900, 1910,....)
# Hint: Dividing the year and multiplying with a number might give you the decade the year falls into!
# You might need to sort before plotting

moviesByDecadeDf = movies.groupby(movies.year//10 * 10).sum()
moviesByDecadeDf.plot(kind = "bar");


# Section II - Q3:
# (A) What are the top 10 most common character names in movie history?
cast.character.value_counts()[:10]

# (B) Who are the top 10 people most often credited as "Herself" in movie history?
cast[cast.character == 'Herself']['name'].value_counts()[:10]

# (C) Who are the top 10 people most often credited as "Himself" in movie history?
cast[cast.character == 'Himself']['name'].value_counts()[:10]

# Section II - Q4:
# Hint: The startswith() function might be useful
# (A) What are the top 10 most frequent roles that start with the word "Zombie"?
cast[cast.character.str.startswith('Zombie')].character.value_counts().head(10)
# (B) What are the top 10 most frequent roles that start with the word "Police"?
cast[cast.character.str.startswith('Police')].character.value_counts().head(10)

# Section II - Q5: Plot how many roles 'Keanu Reeves' has played in each year of his career.
keanu_movies = cast[cast.name == "Keanu Reeves"]
keanu_movies.groupby("year").plot(kind='barh')

# Section II - Q6: Plot the cast positions (n-values) of Keanu Reeve's roles through his career over the years.
# keanu_movies.groupby("year")["n"].count().plot(kind="bar", figsize = (10,5))
keanu = cast[(cast.name == 'Keanu Reeves') & (pd.notnull(cast.n))][['year', 'n']].sort_values('year')
keanu.plot(x='year', y='n', kind='scatter')

# Section II - Q7: Plot the number of "Hamlet" films made by each decade
hamlet = (movies[movies.title == 'Hamlet'].groupby(movies.year // 10 * 10).count().rename({'title': 'count'}, axis=1))['count']
hamlet.plot(kind='bar')

# Section II - Q8:
# Hint: A specific value of n might indicate a leading role
# (A) How many leading roles were available to both actors and actresses, in the 1960s (1960-1969)?
print (cast[(cast.year.between(1960, 1969)) & (cast.n == 1)].groupby(['year', 'type']).count()[['title']].rename({'title': 'count'}, axis=1))
# (B) How many leading roles were available to both actors and actresses, in the 2000s (2000-2009)?
print (cast[(cast.year.between(2000, 2009)) & (cast.n == 1)].groupby(['year', 'type']).count()[['title']].rename({'title': 'count'}, axis=1))

# Section II - Q9: List, in order by year, each of the films in which Frank Oz has played more than 1 role
frankOz = (cast[cast.name == 'Frank Oz'].groupby(['year', 'title']).count()[['name']].rename({'name': 'freq'}, axis=1)
         .sort_values(by=['year'], ascending=True))
print(frankOz[frankOz.freq > 1])


# Section II - Q10: List each of the characters that Frank Oz has portrayed at least twice
frankOz = cast[cast.name == 'Frank Oz'].groupby(['character']).count()[['name']].rename({'name': 'freq'}, axis=1)
print(frankOz[frankOz.freq > 1])


# Section III - Advanced Merging, Querying and Visualizations
# Make a bar plot with the following conditions
# Frequency of the number of movies with "Christmas" in their title
# Movies should be such that they are released in the USA.
# Show the frequency plot by month
christmas = release_dates[(release_dates.title.str.contains('Christmas')) & (release_dates.country == 'USA')]
print(christmas.date.dt.month.value_counts().sort_index().plot(kind='bar'))


# Section III - Q1: Make a bar plot with the following conditions
# Frequency of the number of movies with "Summer" in their title
# Movies should be such that they are released in the USA.
# Show the frequency plot by month
summer = release_dates[(release_dates.title.str.contains('Summer')) & (release_dates.country == 'USA')]
print(summer.date.dt.month.value_counts().sort_index().plot(kind='bar'))

# Section III - Q2: Make a bar plot with the following conditions
# Frequency of the number of movies with "Action" in their title
# Movies should be such that they are released in the USA.
# Show the frequency plot by week
action = release_dates[(release_dates.title.str.contains('Summer')) & (release_dates.country == 'USA')]
print(action.date.dt.dayofweek.value_counts().sort_index().plot(kind='bar'))

# Section III - Q3: Show all the movies in which Keanu Reeves has played the lead role along with their release date in the USA sorted by the date of release
# Hint: You might need to join or merge two datasets!
keanu_movies = cast[cast.name == "Keanu Reeves"]
usa_release = release_dates[(release_dates.country == 'USA')]
merged_usa_keanu = keanu_movies.merge(usa_release, how = "left", on = ["title", "year"])
print(merged_usa_keanu[merged_usa_keanu.n == 1].sort_values(by = "date"))

# Section III - Q4: Make a bar plot showing the months in which movies with Keanu Reeves tend to be released in the USA?
merged_usa_keanu.date.dt.month.value_counts().sort_index().plot(kind='bar')

# Section III - Q5: Make a bar plot showing the years in which movies with Ian McKellen tend to be released in the USA?
ian_mckellen_movies = cast[cast.name == "Ian McKellen"]
merged_usa_ian_mckellen = ian_mckellen_movies.merge(usa_release, how = "left", on = ["title", "year"])
merged_usa_ian_mckellen.date.dt.year.value_counts().sort_index().plot(kind='bar')

plt.show()