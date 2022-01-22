Twitter Sentiment Analysis : Practice Problem
==============================================

Sentiment Analysis:
====================
* It is one of the many application of Natural Language Processing (NLP).
* It is a set of methods and techniques used for extracting 
   subjective information from text or speech, such as opinions or attitudes.

Objective of the course:
========================
It is to solve sentiment analysis problem using python.



Expectations from the Course
=============================
The course is divided into below modules:
*   Text Preprocessing
*   Data Exploration    
*   Feature Extraction 
*   Model Building

NOTE : python > 3


Data
======

Our overall collection of tweets was split in the ratio of 65:35 into 
training and testing data. Out of the testing data, 30% is public and 
the rest is private.

 
Data Files
===========

    train.csv - For training the models, we provide a labelled dataset 
    of 31,962 tweets. The dataset is provided in the form of a csv file with 
    each line storing a tweet id, its label and the tweet.

    There is 1 test file (public)
    test_tweets.csv - The test data file contains only 
    tweet ids and the tweet text with each tweet in a new line.

 
Submission Details
===================

The following 3 files are to be uploaded.

  test_predictions.csv - This should contain the 0/1 label for the 
    tweets in test_tweets.csv, in the same order corresponding to the 
    tweets in test_tweets.csv. Each 0/1 label should be in a new line.

    A .zip file of source code - The code should produce the output file 
        submitted and must be properly commented.

 
Evaluation Metric:
====================
The metric used for evaluating the performance of classification model 
would be F1-Score.

The metric can be understood as -

True Positives (TP) - These are the correctly predicted positive 
values which means that the value of actual class is yes and 
the value of predicted class is also yes.

True Negatives (TN) - These are the correctly predicted 
negative values which means that the value of actual class
 is no and value of predicted class is also no.

False Positives (FP) – When actual class is no and
 predicted class is yes.

False Negatives (FN) – When actual class is yes but 
predicted class in no.

Precision = TP/TP+FP

Recall = TP/TP+FN
 
F1 Score = 2*(Recall * Precision) / (Recall + Precision)

F1 is usually more useful than accuracy, especially if for 
   an uneven class distribution.


Learning Path
===============

1. Understand the Problem Statement    
2. Tweets Preprocessing and Cleaning
    * Data Inspection
    * Data Cleaning    
3. Story Generation and Visualization from Tweets
4. Extracting Features from Cleaned Tweets
    *  Bag-of-Words
    *  TF-IDF
    *  Word Embeddings   
5. Model Building: Sentiment Analysis        
    * Logistic Regression        
    * Support Vector Machine        
    * RandomForest        
    * XGBoost    
6. Model Fine-tuning 


Problem Statement:
==================

The objective of this task is to detect hate speech in tweets.
For the sake of simplicity, we say a tweet contains hate speech if 
it has a racist or sexist sentiment associated with it.
So, the task is to classify racist or sexist tweets from other tweets.

Example:
---------

Formally, given a training sample of tweets and labels, 
where label ‘1’ denotes the tweet is racist/sexist 
and label ‘0’ denotes the tweet is not racist/sexist, 
your objective is to predict the labels on the given test dataset.


Text Preprocessing
==================
    two ways of pre-processing:

* Data Inspection
=================

Text is a hightly unstructured form of data, various types of 
noises are present in it and the data is not readily analyzable
without any pre-processing. The entire process of cleaning and
standardization of text, making  it noise-free and ready for analysis
is known as text Preprocessing.

* Data Cleaning
================

In any natural language processing task, cleaning raw text data is an 
important step. It helps in getting rid of the unwanted words and 
characters which helps in obtaining better features. If we skip 
this step then there is a higher chance that you are working with 
noisy and inconsistent data. The objective of this step is to clean
noise those are less relevant to find the sentiment of tweets such
as punctuation, special characters, numbers, and terms which don’t 
carry much weightage in context to the text.

The Porter stemming algorithm / STEMMING
=========================================

Stemming is the process of reducing a word to its word stem that affixes
to suffixes and prefixes or to the roots of words known as a lemma. For 
example: words such as “Likes”, ”liked”, ”likely” and ”liking” will be 
reduced to “like” after stemming.

alogorithm link :
https://www.tartarus.org/~martin/PorterStemmer/

natural language Processing link/ documentation : 
=================================================

https://pypi.org/project/nltk/


