# Vectorized String Operations

import numpy as np
x = np.array([2, 3, 5, 7, 11, 13])
x * 2

data = ['peter', 'Paul', 'MARY', 'gUIDO']
print([s.capitalize() for s in data])

data = ['peter', 'Paul', 'None', 'MARY', 'gUIDO']
print([s.capitalize() for s in data])

# Pandas includes features to address both this need for vectorized string operations and for correctly handling missing data via the str attribute of Pandas Series and Index objects containing strings. So, for example, suppose we create a Pandas Series with this data:

import pandas as pd
names = pd.Series(data)
print(names)

# We can now call a single method that will capitalize all the entries, while skipping over any missing values:
print(names.str.capitalize())

# Using tab completion on this str attribute will list all the vectorized string methods available to Pandas.

# Tables of Pandas String Methods
# If you have a good understanding of string manipulation in Python, most of Pandas string syntax is intuitive enough that it's probably sufficient to just list a table of available methods; we will start with that here, before diving deeper into a few of the subtleties. The examples in this section use the following series of names:

monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])
# Methods similar to Python string methods
# Nearly all Python's built-in string methods are mirrored by a Pandas vectorized string method. Here is a list of Pandas str methods that mirror Python string methods:
#
# len()	lower()	translate()	islower()
# ljust()	upper()	startswith()	isupper()
# rjust()	find()	endswith()	isnumeric()
# center()	rfind()	isalnum()	isdecimal()
# zfill()	index()	isalpha()	split()
# strip()	rindex()	isdigit()	rsplit()
# rstrip()	capitalize()	isspace()	partition()
# lstrip()	swapcase()	istitle()	rpartition()
# Notice that these have various return values. Some, like lower(), return a series of strings:

print(monte.str.lower())
print(monte.str.len())
print(monte.str.startswith('T'))
print(monte.str.split())

# Methods using regular expressions
# In addition, there are several methods that accept regular expressions to examine the content of each string element,
# and follow some of the API conventions of Python's built-in re module:

# Method	Description
# match()	Call re.match() on each element, returning a boolean.
# extract()	Call re.match() on each element, returning matched groups as strings.
# findall()	Call re.findall() on each element
# replace()	Replace occurrences of pattern with some other string
# contains()	Call re.search() on each element, returning a boolean
# count()	Count occurrences of pattern
# split()	Equivalent to str.split(), but accepts regexps
# rsplit()	Equivalent to str.rsplit(), but accepts regexps

# Extract the first name from each by asking for a contiguous group of characters at the beginning of each element:
print(monte.str.extract('([A-Za-z]+)', expand=False))

# finding all names that start and end with a consonant, making use of the start-of-string (^) and end-of-string ($) regular expression characters:
print(monte.str.findall(r'^[^AEIOU].*[^aeiou]$'))


# Miscellaneous methods
# get()	Index each element
# slice()	Slice each element
# slice_replace()	Replace slice in each element with passed value
# cat()	Concatenate strings
# repeat()	Repeat values
# normalize()	Return Unicode form of string
# pad()	Add whitespace to left, right, or both sides of strings
# wrap()	Split long strings into lines with length less than a given width
# join()	Join strings in each element of the Series with passed separator
# get_dummies()	extract dummy variables as a dataframe
# Vectorized item access and slicing
# The get() and slice() operations, in particular, enable vectorized element access from each array.

# slice of the first three characters of each array using str.slice(0, 3).
# Python's normal indexing syntax–for example, df.str.slice(0, 3) is equivalent to df.str[0:3]:
print(monte.str[0:3])


# extract the last name of each entry, we can combine split() and get():
monte.str.split().str.get(-1)

# get_dummies() method - This is useful when your data has a column containing some sort of coded indicator.
# For example, we might have a dataset that contains information in the form of codes,
# such as A="born in America," B="born in the United Kingdom," C="likes cheese," D="likes spam":

full_monte = pd.DataFrame({'name': monte,
                           'info': ['B|C|D', 'B|D', 'A|C',
                                    'B|D', 'B|C', 'B|C|D']})
print(full_monte)

# The get_dummies() routine lets you quickly split-out these indicator variables into a DataFrame:
full_monte['info'].str.get_dummies('|')

# With these operations as building blocks, you can construct an endless range of string processing procedures when cleaning your data.
#
# Recipe Database
# These vectorized string operations become most useful in the process of cleaning up messy, real-world data.
# Here I'll walk through an example of that, using an open recipe database compiled from various sources on the Web.
# Our goal will be to parse the recipe data into ingredient lists, so we can quickly find a recipe based on some ingredients we have on hand.
#
# The scripts used to compile this can be found at https://github.com/fictivekin/openrecipes, and the link to the
# current version of the database is found there as well.
#
# As of Spring 2016, this database is about 30 MB, and can be downloaded and unzipped with these commands:

# !curl -O https://s3.amazonaws.com/openrecipes/20170107-061401-recipeitems.json.gz
# !gunzip recipeitems-latest.json.gz

# The database is in JSON format, so we will try pd.read_json to read it:

try:
    recipes = pd.read_json('data-wrangling-exercises/data/recipeitems-latest.json')
except ValueError as e:
    print("ValueError:", e)

with open('data-wrangling-exercises/data/recipeitems-latest.json') as f:
    line = f.readline()
pd.read_json(line).shape

# read the entire file into a Python array
with open('data-wrangling-exercises/data/recipeitems-latest.json', 'r') as f:
    # Extract each line
    data = (line.strip() for line in f)
    # Reformat so each line is the element of a list
    data_json = "[{0}]".format(','.join(data))
# read the result as a JSON
recipes = pd.read_json(data_json)
print(recipes.shape)
print(recipes.iloc[0])

recipes.ingredients.str.len().describe()


# Which recipe has the longest ingredient list:
print(recipes.name[np.argmax(recipes.ingredients.str.len())])

# how many of the recipes are for breakfast food:
print(recipes.description.str.contains('[Bb]reakfast').sum())

# how many of the recipes list cinnamon as an ingredient:
print(recipes.ingredients.str.contains('[Cc]innamon').sum())

# any recipes misspell the ingredient as "cinamon":
print(recipes.ingredients.str.contains('[Cc]inamon').sum())

# A simple recipe recommender
# given a list of ingredients, find a recipe that uses all those ingredients. While conceptually straightforward,
# the task is complicated by the heterogeneity of the data: there is no easy operation,
# for example, to extract a clean list of ingredients from each row. So we will cheat a bit: we'll start with a list of
# common ingredients, and simply search to see whether they are in each recipe's ingredient list.
# For simplicity, let's just stick with herbs and spices for the time being:

spice_list = ['salt', 'pepper', 'oregano', 'sage', 'parsley',
              'rosemary', 'tarragon', 'thyme', 'paprika', 'cumin']

# Boolean DataFrame consisting of True and False values, indicating whether this ingredient appears in the list:
import re
spice_df = pd.DataFrame(dict((spice, recipes.ingredients.str.contains(spice, re.IGNORECASE))
                             for spice in spice_list))
spice_df.head()

#compute this very quickly using the query() method of DataFrames, discussed in High-Performance Pandas: eval() and query():
selection = spice_df.query('parsley & paprika & tarragon')
len(selection)

# Use the index returned by this selection to discover the names of the recipes that have this combination:
print(recipes.name[selection.index])