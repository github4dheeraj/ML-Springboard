import inline as inline
import numpy as np  #for mathematical operations
import pandas as pd #for dataframes
import matplotlib.pyplot as plt
import seaborn as sns #visualisation library
import os
# %matplotlib inline

# LOADING BOSTON DATA SET
from sklearn.datasets import load_boston
boston=load_boston()
dataset=pd.DataFrame(boston['data'],columns=boston['feature_names'])
dataset.head(n=5)
dataset.tail(n=6)
dataset.info()

# BOX PLOT
sns.boxplot(x=dataset['DIS'])


