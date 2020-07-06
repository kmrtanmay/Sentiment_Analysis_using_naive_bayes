from utils import process_tweet, lookup , count_tweets
from filter import get_words_by_threshold
from test import test_naive_bayes
from prediction import naive_bayes_predict
from train import train_naive_bayes

import pdb
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd

# Downloading the dataset
nltk.download('stopwords')
nltk.download('twitter_samples')

# add folder, tmp2, from our local workspace containing pre-downloaded corpora files to nltk's data path
filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

# get the sets of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# avoid assumptions about the length of all_positive_tweets
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

# Build the freqs dictionary for storing the frequency of each word in each class

freqs = count_tweets({}, train_x, train_y)

input("Press enter to see the logprior and length of loglikelihood dictionary")

# Train the naive bayes classifier
logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
print("logprior = ",logprior)
print("length of loglikelihood dictionary = ",len(loglikelihood))

input("Press enter to see the train and test accuracy")

# Calculates training accuracy
print("Naive Bayes Training accuracy = %0.4f" %
      (test_naive_bayes(train_x, train_y, logprior, loglikelihood)))

# Calculates Testing Acuracy
print("Naive Bayes Test accuracy = %0.4f" %
      (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))

input("Press enter to see filtering the words on the basis of threshold")

# find negative words at or below a threshold
print("find negative words at or below a threshold")
print(get_words_by_threshold(freqs, label=0, threshold=0.05))

# find positive words at or above a threshold
print("find positive words at or below a threshold")
print(get_words_by_threshold(freqs, label=1, threshold=10))

input("Press enter to see the misclassified tweets")

# Error analysis: Tweets misclassifed
print('Truth Predicted Tweet')
for x, y in zip(test_x, test_y):
    y_hat = naive_bayes_predict(x, logprior, loglikelihood)
    if y != (np.sign(y_hat) > 0):
        print('%d\t%0.2f\t%s' % (y, np.sign(y_hat) > 0, ' '.join(
            process_tweet(x)).encode('ascii', 'ignore')))

# Test your tweet
##my_tweet = 'I am happy because I am learning :)'
##p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
##print(p)