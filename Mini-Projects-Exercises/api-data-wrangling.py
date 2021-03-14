import requests
import pandas as pd

API_KEY = 'NjijCAaXMzUxtSFvgLnh'

# 1. Collect data from the Franfurt Stock Exchange, for the ticker AFX_X, for the whole year 2017 (keep in mind that the date format is YYYY-MM-DD).
# 2. Convert the returned JSON object into a Python dictionary.

payload = {'api_key': API_KEY, 'start_date': '2017-01-01', 'end_date': '2017-12-31'}
response = requests.get('https://www.quandl.com/api/v3/datasets/FSE/AFX_X/data.json', params=payload).json()
print(response)

#3 Calculate what the highest and lowest opening prices were for the stock in this period.
data = response['dataset_data']['data']
cols = response['dataset_data']['column_names']
df = pd.DataFrame(data, columns=cols)
print("Maximum",df["Open"].max())
print("Minimum", df["Open"].min())

#4. What was the largest change in any one day (based on High and Low price)?
maxChange = df["High"] - df["Low"]
print("Largest change in any one day", maxChange.max())

#5. What was the largest change between any two days (based on Closing Price)?
prior = df["Close"][1:]
print(prior)
after = df["Close"][:-1]
print(after)
changed = prior - after
print("Largest change between any two days", changed.max())

#6. What was the average daily trading volume during this year?
print("average daily trading volume", df["Traded Volume"].mean())

#7. (Optional) What was the median trading volume during this year. (Note: you may need to implement your own function for calculating the median.)
print("average daily trading volume", df["Traded Volume"].median())


