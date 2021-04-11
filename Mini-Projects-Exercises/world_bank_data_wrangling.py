# JSON examples and exercise
# get familiar with packages for dealing with JSON
# study examples with JSON strings and files
# work on exercise to be completed and submitted
# reference: http://pandas.pydata.org/pandas-docs/stable/io.html#io-json-reader
# data source: http://jsonstudio.com/resources/

import pandas as pd
# imports for Python, Pandas
import json
from pandas import json_normalize
# JSON example, with string
# demonstrates creation of normalized dataframes (tables) from nested json string
# source: http://pandas.pydata.org/pandas-docs/stable/io.html#normalization
# define json string
data = [{'state': 'Florida',
         'shortname': 'FL',
         'info': {'governor': 'Rick Scott'},
         'counties': [{'name': 'Dade', 'population': 12345},
                      {'name': 'Broward', 'population': 40000},
                      {'name': 'Palm Beach', 'population': 60000}]},
        {'state': 'Ohio',
         'shortname': 'OH',
         'info': {'governor': 'John Kasich'},
         'counties': [{'name': 'Summit', 'population': 1234},
                      {'name': 'Cuyahoga', 'population': 1337}]}]

# use normalization to create tables from nested element
json_normalize(data, 'counties')

# further populate tables created from nested element
json_normalize(data, 'counties', ['state', 'shortname', ['info', 'governor']])


# JSON example, with file
# demonstrates reading in a json file as a string and as a table
# uses small sample file containing data about projects funded by the World Bank
# data source: http://jsonstudio.com/resources/
# load json as string
json.load((open('data/world_bank_projects_less.json')))


# load as Pandas dataframe
wb_json_df = pd.read_json('data/world_bank_projects.json')
print(wb_json_df)


# JSON exercise
# Using data in file 'data/world_bank_projects.json' and the techniques demonstrated above,
#
# Find the 10 countries with most projects
# Find the top 10 major project themes (using column 'mjtheme_namecode')
# In 2. above you will notice that some entries have only the code and the name is missing. Create a dataframe with the missing names filled in.


#1. Find the 10 countries with most projects
print(wb_json_df.groupby('countryshortname').size().sort_values(ascending=False)[:10])
# print(wb_json_df.groupby('countryshortname')['project_name'].count().nlargest(10))

#2. Find the top 10 major project themes (using column 'mjtheme_namecode')
json_normalize(wb_json_df.mjtheme_namecode[0])

df = pd.DataFrame(columns=['name', 'code'])
for index, row in wb_json_df.iterrows():
    append_me = json_normalize(row.mjtheme_namecode)
    df = pd.concat([df, append_me])

print(df.isnull().sum())

# OR
# # load json file w/ context manager
# with open('data/world_bank_projects.json') as file:
#     json_obj = json.load(file)
#
# # normalize the mjtheme_namecode columns using json_normalize
# df_mjtheme_namecode = json_normalize(json_obj, 'mjtheme_namecode')
# df_mjtheme_namecode.head()
# group by name
# df_mjtheme_namecode['name'].value_counts().nlargest(10)

# Since there're missing names, try group by code instead
# df_mjtheme_namecode['code'].value_counts().nlargest(10)

# In 2. above you will notice that some entries have only the code and the name is missing. Create a dataframe with the missing names filled in.
import numpy as np
filled_wb_df = df.replace(r'^\s*$', np.nan, regex=True)
filled_wb_df.head()

temp = filled_wb_df.dropna().drop_duplicates()
final = filled_wb_df.merge(temp, on="code", how="left", suffixes=("_old", "_new")).drop(columns=["name_old"])
print(final.shape)
print(temp.shape)


# OR

# # Collect name for each code
# code_name_dict = {}
#
# for code, name in zip(df_mjtheme_namecode['code'], df_mjtheme_namecode['name']):
#     if name != None and name != '' and code not in code_name_dict:
#         code_name_dict[code] = name
#
# print(code_name_dict)
#
# # Fill missing name using the name collected in code_name_dict
# df_mjtheme_namecode['name'] = df_mjtheme_namecode['code'].map(code_name_dict)
# df_mjtheme_namecode.head()
#
# # Double check by group by name again & confirm that there're no missing names
# # and the results are same as group by code as in question #2
# df_mjtheme_namecode['name'].value_counts().nlargest(10)
#
# # Plot barchart after filling missing names
# import matplotlib.pyplot as plt
#
# plot = df_mjtheme_namecode['name'].value_counts().plot(kind='bar', title='Top 10 Major Project Themes', rot=90)
# plot.set_xlabel("Project Themes")
# plot.set_ylabel("Project Count")
# plt.show()