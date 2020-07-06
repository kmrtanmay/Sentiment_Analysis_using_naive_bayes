## Sentiment Analysis using Naive Bayes model
This is used to classify the tweets whether they are positive or negative. It is basically solved using [Naive bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) technique.
### Dataset: twitter_samples imported from Natural Language Toolkit(**nltk**)
- It consists of 5000 positive sentiments tweets and 5000 negative sentiments tweets. Out of which 4000 each positive and negative tweets are used for training while the remaining tweets used for testing purpose.
### Data Preprocessing
- `process_tweet` function: It is used to remove all the hashings, urls, stopwords, punctuations. It further replaces the words with their stem words. It finally tokenizes the processed tweet.for example
> This is an example of a positive tweet: 
> #FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my >community this week :)

>This is an example of the processed version of the tweet: 
> ['followfriday', 'top', 'engag', 'member', 'commun', 'week', ':)']
- `lookup function` : It calculates the frequency of a word present in the corpus on the basis of its label.
> The key is the tuple (word, label), such as ("happy",1) or ("happy",0). The value stored for each key is the count of how many times the word "happy" was associated with a positive label, or how many times "happy" was associated with a negative label.

- `count_tweets()` : It takes a list of tweets as input, cleans all of them, and returns a dictionary.
> The key in the dictionary is a tuple containing the stemmed word and its class label, e.g. ("happi",1).

>The value the number of times this word appears in the given collection of tweets (an integer).

### **Training The Naive Bayes Classifier**
##### So how do you train a Naive Bayes classifier?
- The first part of training a naive bayes classifier is to identify the number of classes that you have.
- You will create a probability for each class.
$P(D_{pos})$ is the probability that the document is positive.
$P(D_{neg})$ is the probability that the document is negative.
Use the formulas as follows and store the values in a dictionary:

$$P(D_{pos}) = \frac{D_{pos}}{D}\tag{1}$$

$$P(D_{neg}) = \frac{D_{neg}}{D}\tag{2}$$

Where $D$ is the total number of documents, or tweets in this case, $D_{pos}$ is the total number of positive tweets and $D_{neg}$ is the total number of negative tweets.

#### Prior and Logprior

The prior probability represents the underlying probability in the target population that a tweet is positive versus negative.  In other words, if we had no specific information and blindly picked a tweet out of the population set, what is the probability that it will be positive versus that it will be negative? That is the "prior".

The prior is the ratio of the probabilities $\frac{P(D_{pos})}{P(D_{neg})}$.
 Take the log of the prior to rescale it, and we'll call this the logprior

$$\text{logprior} = log \left( \frac{P(D_{pos})}{P(D_{neg})} \right) = log \left( \frac{D_{pos}}{D_{neg}} \right)$$.

Note that $log(\frac{A}{B})$ is the same as $log(A) - log(B)$.  So the logprior can also be calculated as the difference between two logs:

$$\text{logprior} = \log (P(D_{pos})) - \log (P(D_{neg})) = \log (D_{pos}) - \log (D_{neg})\tag{3}$$

#### Positive and Negative Probability of a Word
To compute the positive probability and the negative probability for a specific word in the vocabulary, use the following inputs:

- $freq_{pos}$ and $freq_{neg}$ are the frequencies of that specific word in the positive or negative class. In other words, the positive frequency of a word is the number of times the word is counted with the label of 1.
- $N_{pos}$ and $N_{neg}$ are the total number of positive and negative words for all documents (for all tweets), respectively.
- $V$ is the number of unique words in the entire set of documents, for all classes, whether positive or negative.

Use these to compute the positive and negative probability for a specific word using this formula:

$$ P(W_{pos}) = \frac{freq_{pos} + 1}{N_{pos} + V}\tag{4} $$
$$ P(W_{neg}) = \frac{freq_{neg} + 1}{N_{neg} + V}\tag{5} $$

Notice that "+1" is added in the numerator for additive smoothing.  This [wiki article](https://en.wikipedia.org/wiki/Additive_smoothing) explains more about additive smoothing.

#### Log likelihood
To compute the loglikelihood of that very same word, implement the following equations:
$$\text{loglikelihood} = \log \left(\frac{P(W_{pos})}{P(W_{neg})} \right)\tag{6}$$

#####  `freqs` dictionary
- Given  `count_tweets()` function,  compute a dictionary called `freqs` that contains all the frequencies.
- In this `freqs` dictionary, the key is the tuple (word, label)
- The value is the number of times it has appeared.

###  **Test naive bayes**

Now that we have the `logprior` and `loglikelihood`, we can test the naive bayes function by making predicting on some tweets!

####  `naive_bayes_predict` : This function to make predictions on tweets.
* The function takes in the `tweet`, `logprior`, `loglikelihood`.
* It returns the probability that the tweet belongs to the positive or negative class.
* For each tweet, sum up loglikelihoods of each word in the tweet.
* Also add the logprior to this sum to get the predicted sentiment of that tweet.

$$ p = logprior + \sum_i^N (loglikelihood_i)$$

####  `test_naive_bayes` : This  function is used to check the accuracy of predictions.
* The function takes in your `test_x`, `test_y`, log_prior, and loglikelihood
* It returns the test accuracy of your model.
* First, use `naive_bayes_predict` function to make predictions for each tweet in text_x.

###  Filter words by Ratio of positive to negative counts

- Some words have more positive counts than others, and can be considered "more positive".  Likewise, some words can be considered more negative than others.
- One way for us to define the level of positiveness or negativeness, without calculating the log likelihood, is to compare the positive to negative frequency of the word.
    - Note that we can also use the log likelihood calculations to compare relative positivity or negativity of words.
- We can calculate the ratio of positive to negative frequencies of a word.
- Once we're able to calculate these ratios, we can also filter a subset of words that have a minimum ratio of positivity / negativity or higher.
- Similarly, we can also filter a subset of words that have a maximum ratio of positivity / negativity or lower (words that are at least as negative, or even more negative than a given threshold).

####  `get_ratio()`:
- Given the `freqs` dictionary of words and a particular word, use `lookup(freqs,word,1)` to get the positive count of the word.
- Similarly, use the `lookup()` function to get the negative count of that word.
- Calculate the ratio of positive divided by negative counts
####  `get_words_by_threshold(freqs,label,threshold)`:
-   If we set the label to 1, then we'll look for all words whose threshold of positive/negative is at least as high as that threshold, or higher.
-   If we set the label to 0, then we'll look for all words whose threshold of positive/negative is at most as low as the given threshold, or lower.
-   Use the  `get_ratio()`  function to get a dictionary containing the positive count, negative count, and the ratio of positive to negative counts.
-   Append a dictionary to a list, where the key is the word, and the dictionary is the dictionary  `pos_neg_ratio`  that is returned by the  `get_ratio()`  function. An example key-value pair would have this structure:
    
    ```
    {'happi':
      {'positive': 10, 'negative': 20, 'ratio': 0.5}
    }
    ```
#### Test Accuracy : **99.40%**

##### Error Analysis 
- There are some tweets which are wrongly predicted. On running the code, one can see all those tweets which are wrongly classified