#!/usr/bin/env python
# coding: utf-8

# # GETTING DATA

# In[1]:


import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import csv
import matplotlib.pyplot as plt
import nltk
import nltk.tokenize as tokenize
import seaborn as sns


# In[2]:


def getKaggleNewsDataSet():
    fake = pd.read_csv("data/Fake.csv")
    true = pd.read_csv("data/True.csv")
    # add a field to determine fake and real
    fake['isFake'] = 1
    true['isFake'] = 0
    # combine
    data = pd.concat([fake, true]).reset_index(drop = True)
    # suffle to prevent bias
    data = shuffle(data)
    data = data.reset_index(drop=True)
    return data


# In[3]:


def getReserachArticleNewsDataSet():
    data_dir = "data/research-data/"
    rd = pd.read_csv(data_dir+"researcharticles.csv", sep=',', names=["id", "url", "source", "desc"])
    fake = readFile(rd.loc[rd['desc'] == 'Not-Real-Other'], data_dir, 1)
    real = readFile(rd.loc[rd['desc'] == 'Real'], data_dir, 0)
    data = pd.concat([fake, real]).reset_index(drop = True)
    # suffle to prevent bias
    data = shuffle(data)
    data = data.reset_index(drop=True)
    data['isFake'] = data['isFake'].astype(int) 
    return data
    
def readFile(df, data_dir, isFake):
    column_names = ['text', 'isFake']
    news_data = pd.DataFrame(columns = column_names)
    
    for index, row in df.iterrows():
        txt = pd.read_csv(data_dir+row['id'], sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')
        news_data = news_data.append({'text':txt, 'isFake':isFake}, ignore_index=True)
    return news_data


# # DATA  SPLITTING

# In[4]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split the data into a training and test set.
def split_data(data, labels):
#     X_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)
    return train_test_split(data.text, data.isFake, test_size=0.2, random_state=42, shuffle="true")

# X_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)


def train_NB(train_data, train_labels):
    return MultinomialNB().fit(train_data, train_labels)


def train_random_forest(train_data, train_labels, est):
    return RandomForestClassifier(n_estimators=est).fit(train_data, 
        train_labels)


def test_classifier(clf, validate_data, validate_labels, str):
    predicted = clf.predict(validate_data)
    print(str)
    print(np.mean(predicted == validate_labels))    


# # DATA CLEANING

# ### Case Insensitive
# ### Remove Stopwords
# ### Remove Punctuations
# ### Lemmatization OR Stemming - Lemmarization 
# ### POS - parts-of-speech

# In[ ]:


import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
import spacy
import re
# Using Porter Stemmer implementation in nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer() 

lm = WordNetLemmatizer()
punct = set(string.punctuation)
sw = set(stopwords.words('english'))

def clean_data(df, columns):
    for i, col in enumerate(columns):
        df[col] = df[col].apply(lambda text: clean_text(text))

### Lemmatization OR Stemming
# Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological 
# analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary 
# form of a word, which is known as the lemma.
        
def clean_text(text):
    text = ''.join(char.lower() for char in text if char not in punct)
    tokens = re.split('\W+', text)
    text = ' '.join([lm.lemmatize(word) for word in tokens if word not in sw])
#     text = ' '.join([(lm.lemmatize(word) and stemmer.stem(word)) for word in tokens if word not in sw])
    return text

########### BELOW METHOD IS TIME CONSUMING and NOT WORKING  ##########

def clean_text_data(df, columns):
    for i, col in enumerate(columns):
        print (i, ",",col)
        # convert text to lower case
        df[col] = df[col].str.lower()
        # remove punctuations
        df[col] = df[col].apply(lambda text: remove_punctuation(text))
        # tokenize and remove stopwords
        df[col] = df[col].apply(lambda text: remove_stopwords(text))

        

def remove_punctuation(text):
    return str(text).translate(str.maketrans('', '', string.punctuation))
    
def remove_stopwords(text):
    remove_nltk_stopwords(text)
#     remove_spacy_stopwords(text)
        
def remove_nltk_stopwords(text):
#     text = all_data["text"]
#     stop = stopwords.words('english')
#     return text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    sw = set(stopwords.words('english'))
    deto = Detok()
    
    all_cleaned = list()
    
    for article in text:
        word_tokens = word_tokenize(article) 
        all_cleaned.append(deto.detokenize(
            [w for w in word_tokens if not w in sw]))
    return all_cleaned

def remove_spacy_stopwords(text):
    spacy_nlp = spacy.load("en_core_web_sm")
    sw = spacy.lang.en.stop_words.STOP_WORDS
    deto = Detok()

    all_cleaned = list()

    for article in text:
        word_tokens = word_tokenize(article) 
        all_cleaned.append(deto.detokenize(
            [w for w in word_tokens if not w in sw]))

    return all_cleaned


# # FEATURE EXTRACTION
# ### Word counts and Cloud
# ### Frequency Distributions
# ### Relevancy
# ### Sentiment Analysis

# ## Word Cloud

# In[ ]:


from wordcloud import WordCloud
def word_cloud(text, column):
    all_words = ' '.join(str(text) for text in text[column])
    wordcloud = WordCloud(width= 800, height= 500,
                              max_font_size = 110,
                              collocations = False).generate(all_words)
    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# ## Count (words and ngrams)

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

def count_words(all_data, train_data, test_data):
    cv = CountVectorizer()
    cv = count_vect.fit(all_data)
    x_train_data =  cv.transform(train_data)
    x_test_data =  cv.transform(test_data)
    return x_train_data, x_test_data


# ## N-grams
# N-grams are simply all combinations of adjacent words or letters of length n that you can find in your source text. For example, given the word fox, all 2-grams (or “bigrams”) are fo and ox. You may also count the word boundary – that would expand the list of 2-grams to #f, fo, ox, and x#, where # denotes a word boundary.
# 
# You can do the same on the word level. As an example, the hello, world! text contains the following word-level bigrams: # hello, hello world, world #.
# 
# The basic point of n-grams is that they capture the language structure from the statistical point of view, like what letter or word is likely to follow the given one. The longer the n-gram (the higher the n), the more context you have to work with. Optimum length really depends on the application – if your n-grams are too short, you may fail to capture important differences. On the other hand, if they are too long, you may fail to capture the “general knowledge” and only stick to particular cases. 
# 

# In[ ]:


def count_ngrams(all_data, train_data, test_data):
    cv = CountVectorizer(ngram_range=(2,3))
    cv = count_vect.fit(all_data)
    x_train_data =  cv.transform(train_data)
    x_test_data =  cv.transform(test_data)
    return x_train_data, x_test_data


# ## Frequency Distributions

# In[ ]:


token_space = tokenize.WhitespaceTokenizer()
def frequency(text, column, quantity):
    all_words = ' '.join(str(text) for text in text[column])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()


# ## Relevancy (TF-IDF)
# TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection of documents

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_words(all_data, train_data, test_data):
    tfidfVector = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidfVector.fit(all_data)
    x_train_data =  tfidfVector.transform(train_data)
    x_test_data =  tfidfVector.transform(test_data)
    return x_train_data, x_test_data


# In[ ]:


def tfidf_ngrams(all_data, train_data, test_data):
    tfidfVector = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', 
                                 ngram_range=(2,3), max_features=5000)
    tfidfVector.fit(all_data)
    x_train_data =  tfidf_vect.transform(train_data)
    x_test_data =  tfidf_vect.transform(test_data)
    return x_train_data, x_test_data


# ## Stemming

# In[ ]:


from TagLemmatize import *
def tag_lemmatize(data_list):
    ret_list = []
    for d in data_list:
        ret_list.append(tag_and_lem(d))
    return ret_list


# ## Sentiment Analysis

# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
def vader_score(data_list):
    analyser = SentimentIntensityAnalyzer()
    ret_list = list()
    for data in data_list:
        ret_list.append(list(analyser.polarity_scores(data).values()))
    return ret_list

def vader_score_non_neg(article_list):
    ret_list = list()
    for article_vals in article_list:
        ret_list.append([x+1 for x in article_vals])
    return ret_list


# ## POS - parts-of-speech

# In[ ]:


def parts_of_speech(all_data):
    # Turn all_data into PoS
    all_pos = list()
    for article in all_data:
        all_pos.append(pos_tag(word_tokenize(article)))

    # Create a counter for all_pos
    all_pos_counter = list()
    for article in all_pos:
        all_pos_counter.append(Counter( tag for word,  tag in article))

    all_pos_count = list()

    tagdict = load('help/tagsets/upenn_tagset.pickle')
    # Count up each PoS and giving a value of 0 to those that do not occur
    for counter in all_pos_counter:
        temp = list()
        for key in tagdict:
            temp.append(counter[key])
        all_pos_count.append(temp)

    return all_pos_count

