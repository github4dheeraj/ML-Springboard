## MISSING DATA

import pandas as pd
df = pd.read_csv("/Users/dheeraj.chaudhari2/Documents/workspace/ML/Python/data/Fake.csv", parse_dates=['date'])
# Inplace replace the original df
df.set_index('date', inplace=True)
df.head()

# Replace all NAN
new_df = df.fillna("NO NEWS")
new_df.head()

# Replace column specific NAN using dict
new_df = df.fillna({
    "title": "NO TITLE",
    "text":"NO_TEXT",
    "subject":"NO_SUBJECT"
})
new_df.head()

# Forward and backward fill
new_df = df.fillna(method="ffill")
new_df.head()
new_df = df.fillna(method="bfill")
new_df.head()

# Only applicable for numeric columns
# new_df = df.interpolate()
# new_df.head()
