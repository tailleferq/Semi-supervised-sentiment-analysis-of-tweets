# Semi-supervised-sentiment-analysis-of-tweets


Here we want to build a classifier of tweet. He will classify tweets in positive or negative with a sentiment analysis.

The objective is to use a semi-supervised methodology : Co-training. This methodology use two classifiers. You will find here one classifier.

It is a classifier using Naive Bayes or Support Vector Machine. It is a standard classifier, all the work is about the representation of tweets. So you can choose your methodology (NB or SVM) and the features of the representation of the tweets.
uni : use only unigram as feature
uni_neg : use unigram and words related by a neg relation from a dependency parser as feature
uni_mw : use unigram and word related by a relation with modification from a dependency parser as feature
all_feat : use all feat use above

DATABASE 

You need two files : train.csv and test.csv. The first one contains tweets for training and the second one contains tweets for testing. 
Each line of the file represent one tweet. You need 3 columns. The first one is the id of the tweet, the second one is the polarity of the tweet and the last one is the tweet. 

PACKAGES 

You need to install the following packages to run this programm : 

nltk 
re
pandas
autocorrect
unidecode
os
csv
matplotlib
spacy
optunity

You also need to download the stopword corpus for nltk too, using this command : 

python -c "import nltk; nltk.download('stopwords')"

HOW TO RUN THE PROGRAM

You have to install all packages above and write 
    python create_dataset.py
in a terminal in the good folder and answer all the question, paying attention write the good words.

RESULTS

You will see on the terminal the acuracy and the f-measure. Don't worry if the program take time to run

Don't worry about the files who appear in your folder. It is the results of several step of the program. If you run again a program with the same parameters, he will read these files instead of compute again.

BE CARREFUL, if you change the dataset, move the files that my program create !!! 
