from __future__ import division
from nltk.tokenize import TweetTokenizer
import re
import pandas as pd
from nltk.corpus import stopwords
import nltk
from autocorrect import spell
from nltk.stem.porter import *
from unidecode import unidecode
import math
from sklearn.metrics import *
import os
import csv
from matplotlib import pyplot as plt
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from sklearn.svm import SVC
import optunity
import optunity.metrics
from sklearn.linear_model import SGDClassifier


#####################################################################
def read_preprocess (string) :
    #extract all datas from csv_files and creation of a dataframe
    dfdataset = pd.read_csv(string, sep=',', engine='python')
    #extract colums
    df1stID=dfdataset[dfdataset.columns[0]]
    dfpolar=dfdataset[dfdataset.columns[1]]
    dforigi=dfdataset[dfdataset.columns[2]]
    dfclean=dfdataset[dfdataset.columns[3]]
    #conversion of the dataframe of tweet in a list of tweets
    acc=dfclean.tolist()
    clean_l = []
    for e in acc :
        e = e[1:]
        e = e[1:]
        e = e[:-1]
        e = e[:-1]
        clean_l.append(e.split ("', '"))
    fstID_l=df1stID.tolist()
    polar_l=dfpolar.tolist()
    origi_l=dforigi.tolist()
    #creation of the list with all details
    res =[]
    for i in range(len (fstID_l)) :
        res.append([fstID_l[i], polar_l[i], origi_l[i],clean_l[i]])
    return (res)

def write_preprocess (l,string) :
    c = csv.writer(open(string, "wb"))
    c.writerow(["id"]+["polarity"]+["original tweet"]+["clean tweet "])
    for e in l :
        c.writerow([e[0]]+[e[1]]+[e[2]]+[e[3]])
    return (True)
#####################################################################



#####################################################################
def extract_tweet (csv_files) :
    #extract all datas from csv_files and creation of a dataframe
    dfdataset = pd.read_csv(csv_files, sep=',', engine='python')
    #extract colums
    df1stID=dfdataset[dfdataset.columns[0]]
    dfpolar=dfdataset[dfdataset.columns[1]]
    dftweet=dfdataset[dfdataset.columns[2]]
    #conversion of the dataframe of tweet in a list of tweets
    tweet_l=dftweet.tolist()
    fstID_l=df1stID.tolist()
    polar_l=dfpolar.tolist()
    #creation of the list with all details
    res =[]
    for i in range(len (fstID_l)) :
        res.append([(fstID_l[i]), (polar_l[i]), (tweet_l[i])])
    return (res)
#####################################################################

#####################################################################
def stemming_tweet(l) :
    stemmer = PorterStemmer()
    return (map(lambda x:stemmer.stem(x), l))

def tokenizeTweet (s) :
    tknzr = TweetTokenizer()
    s0=s
    s1=tknzr.tokenize(s0)
    return(s1)

def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def reduce_lengthening(text):
    return (reduce(lambda x,y: x+y if x[-3:]!=y*3 else x, text, ""))

def spellCorr (l) :
    l1 =[]
    for e in l :
        reduce_word = reduce_lengthening(e)
        if e != reduce_word :
            l1.append(reduce_word)
        else :
            dec_word = decontracted(e)
            if e == (dec_word) :
                l1.append(spell(e))
            else :
                l1 = l1 + tokenizeTweet(dec_word)
    return l1

def removeNER(l):
    chunked = nltk.ne_chunk(nltk.pos_tag(l))
    tokens = [leaf[0] for leaf in chunked if type(leaf) != nltk.Tree]
    return(tokens)

def removeNumber (l) :
    l = map(lambda x:(re.sub("[0-9]\S+", "", x)), l) # remove word which begin by a number
    l = map(lambda x:(re.sub("[0-9]", "", x)), l) # reove number
    return (removeEmpty(l))

def removeSWs (l) :
    return ([word for word in l if word not in stopwords.words('english')])

def expandAccronym(l, dic):
    l1 = []
    for e in l :
        e = e.lower()
        acc = 0
        for key in dic.iterkeys() :
            if e == key :
                res = tokenizeTweet(dic[key])
                l1= l1+res
                acc = 1 
        if acc == 0 :
            l1.append(e)
    return (l1)

def removeUppercase(l):
    return map(lambda x:(x.lower()), l)

def removePunct(l):
    l = map(lambda x:(re.sub("[^A-Za-z0-9'#\s]+", '', x)), l) 
    return(removeEmpty(l))

def removeHTs (l) :
    l = map(lambda x:(re.sub("#\S+", "", x)), l)
    return (removeEmpty(l))

def removeEmpty (l) :
    res = []
    for e in l :
        if e != "" :
            res. append(e)
    return res

def removeRefs (s) :
    return re.sub("@\S+", "", s)

def removeReferences (l) :
    l = map(lambda x:removeRefs(x), l)
    return removeEmpty(l)

def removeURLs (s) :
    return re.sub("http\S+", "", s)

def removeLink (l) :
    l = map(lambda x:removeURLs(x), l)
    return (removeEmpty(l))

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7f]',r' ',text)
    #return unidecode(unicode(text, encoding = "utf-8"))

def pre_pro (s1,s2) :
    return (stemming_tweet((spellCorr(removeUppercase([str(s1)])))))[0] + "_" + (stemming_tweet((spellCorr(removeUppercase([str(s2)])))))[0]

def extract_negation(l):
    res = []
    for e in l :
        if e[1] == "neg" :
            res.append(pre_pro(e[2],e[0]))
    return res

def remove_elem(uni,bi) :
    for f in bi:
        f0 = f.split ('_')[0]
        f1 = f.split ('_')[1]
        if f0 in uni :
            uni.remove(f0)
        if f1 in uni :
            uni.remove(f1)
    return uni 

def extract_pattern(l):
    res = []
    for e in l :
        e0 = e[0]
        e1 = e[1]
        e2 = e[2]
        if e1 == "amod" : 
            res.append(pre_pro(e2,e0))
        else :
            if e1 == "compound" :
                res.append(pre_pro(e2,e0))
            else :
                if e1 == "dobj" :
                    res.append(pre_pro(e2,e0))
                else :
                    if e1 == "appos" :
                        res.append(pre_pro(e2,e0))
                    else :
                        if e1 == "nounmod" :
                           res. append(pre_pro(e2,e0))
                        else :
                           if e1 == "npmod" :
                               res.append(pre_pro(e2,e0))
                           else :
                               if e1 == "advmod" :
                                   res.append(pre_pro(e2,e0))
    return res

def cleaner (l, dic,string):
    original_tweet = unicode(remove_non_ascii(l[2]), "utf-8")
    res1 = remove_non_ascii(original_tweet)
    res2 = tokenizeTweet(res1)
    res3 = removeLink(res2)
    res4 = removeReferences(res3)
    res5 = removeHTs(res4)
    res6 = removePunct(res5)
    res7 = removeNumber(res6)
    res8 = removeNER(res7)
    res9 = expandAccronym(res8,dic)
    res10 = removeUppercase(res9)
    res11 = spellCorr(res10)
    res12 = removeSWs(res11)
    uni = stemming_tweet(res12)
    if string == "uni" :
        res = [l[0], l[1], l[2], uni]
    else :
        doc = nlp(decontracted(original_tweet))
        acc=[]
        for token in doc:
            acc.append([token.text, token.dep_, token.head.text])    
        if string == "uni_neg" :
            neg = extract_negation(acc)
            uni_without_neg = remove_elem(uni,neg)
            res = [l[0], l[1], l[2], neg + uni_without_neg]
        else :
            if string == "uni_mw" :
                multiwords = extract_pattern(acc)
                res = [l[0], l[1], l[2], multiwords + uni]
            else :
                neg = extract_negation(acc)
                uni_without_neg = remove_elem(uni,neg)
                multiwords = extract_pattern(acc)
                res = [l[0], l[1], l[2], multiwords + neg + uni_without_neg]
    f = []
    for e in res[3] :
         f.append(e.encode('ascii','ignore'))
    return [res[0], res[1], res[2], f]
#####################################################################




#####################################################################
def createDic(name):
    dfdic = pd.read_csv(name, sep=',', header=None, engine='python')
    df1=dfdic[dfdic.columns[0]]
    df2=dfdic[dfdic.columns[1]]
    l1=df1.tolist()
    l2=df2.tolist()
    l1 = map(lambda x:str(x), l1)
    l2 = map(lambda x:str(x), l2)
    return (dict(zip( l1, l2)))

def cleaner_l (l,s1,s3) :
    s2 = ""
    dicFiles = "acronym_dic.csv"
    dic = createDic(dicFiles)
    if os.path.isfile("prepros_"+s2+"_"+s1+"_"+s3+".csv"):
        res = read_preprocess ("prepros_"+s2+"_"+s1+"_"+s3+".csv")
    else :
        res = map(lambda x:(cleaner(x, dic, s1)), l)
        write_preprocess(res, "prepros_"+s2+"_"+s1+"_"+s3+".csv")
    return (res)
#####################################################################



##################################################################### 
def get_words_in_tweets(l):
    all_feats = []
    for e in l :
        for feat in e[3] :
            all_feats.append(feat)
    return all_feats

def get_word_features(wordlist,var):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    lenght = len(word_features)
    print "nb feature = " + str(lenght)
    return word_features
#####################################################################



#####################################################################
def count(f, all_tweets):
     return sum (1 for j in all_tweets if f in j[3])

def create_dic_idf(l, all_tweets):
    res = {}
    for e in l :
        res[e] = count(e, all_tweets)
    return res
    
def tf(feat, tweet_l):
    acc = 0
    for f in tweet_l :
        if f == feat : acc = acc + 1
    if len(tweet_l) == 0 :
        res = 0
    else :
        res = acc/len(tweet_l)
    return res

def idf(feat, tweets_l, dic_idf):
    return math.log(len(tweets_l) / (1+dic_idf[feat]))

def tfidf(feat, tweet_l, tweets_l, dic_idf):
    res =  tf(feat, tweet_l) * idf(feat, tweets_l, dic_idf)
    return res

def extract_features(tweet_l, feat_l,tweets_l, dic_idf):
    #document_words = set(tweet_l)
    features = {}
    for feat in feat_l :
        features[feat] = (tfidf (feat, tweet_l, tweets_l, dic_idf))
        #features[feat] = (word in document_words)
    return features

def write_training_set(tweets_l, string) :
    c = csv.writer(open(string, "wb"))
    c.writerow(["id"]+["polarity"]+["original tweet"]+tweets_l[0][3].keys())
    for e in tweets_l :
        c.writerow([e[0]]+[e[1]]+[e[2]]+e[3].values())
    return ("training_with_features.csv")

def read_training_set (name):
    res = []
    dftr = pd.read_csv(name, sep=',', engine='python')
    dfid=dftr[dftr.columns[0]]
    dfpo=dftr[dftr.columns[1]]
    dfot=dftr[dftr.columns[2]]
    id_l=dfid.tolist()
    po_l=dfpo.tolist()
    ot_l=dfot.tolist()
    headers = list(dftr.head(0))
    headers.pop(0)
    headers.pop(0)
    headers.pop(0)
    res = []
    for i in range(len(id_l)) :
        t_l = (dftr.iloc[i]).tolist()
        t_l.pop(0)
        t_l.pop(0)
        t_l.pop(0)
        acc = (dict(zip( headers, t_l)))
        res.append ([id_l[i], po_l[i], ot_l[i], acc])
    return (res)

def createTrainingSet (tweets_l,feat_l,tweets_all, dic_idf, s1, s3, s4, s5) :
    s2 = ""
    if os.path.isfile("data_all_feat_"+s2+"_"+s4+"_"+s1+"_"+s3+"_"+s5+".csv"):
        res = read_training_set ("data_all_feat_"+s2+"_"+s4+"_"+s1+"_"+s3+"_"+s5+".csv")
        print len (res[0][3])
    else :
        res=[]
        acc = 0
        for e in tweets_l :
            res.append([e[0],e[1],e[2],extract_features(e[3],feat_l,tweets_all, dic_idf)])
            if acc % 1000 == 0 : print "Representation of " + str(acc + 1) + " st tweet"
            acc = acc + 1
        write_training_set(res, "data_all_feat_"+s2+"_"+s4+"_"+s1+"_"+s3+"_"+s5+".csv")
    return(res)
#####################################################################


##################################################################### 
def write_pred(pred,string):
    c = csv.writer(open(string, "wb"))
    for e in pred :
        c.writerow([e])

def read_pred(string):
    dfpol = pd.read_csv(string, header=None, sep=',', engine='python')
    dfpola=dfpol[dfpol.columns[0]]
    pol_l=dfpola.tolist()
    return (pol_l)

def createListTrue(l):
    true = []
    for e in l :
        if e[1] == "positive" :
            true.append(1)
        else :
            true.append(0)
    return(true)

def createListPred_nb(l,classifier, s1, s3, s4):
    s2 = ""
    pred = []
    if not os.path.isfile("prediction_"+s2+"_"+s4+"_"+s1+"_"+s3+"_"+"nb"+".csv"):
        for e in l :
            if classifier.classify(e[3]) == "positive" :
                pred.append(1)
            else :
                pred.append(0)
        write_pred(pred,"prediction_"+s2+"_"+s4+"_"+s1+"_"+s3+"_"+"nb"+".csv")
    else :
        pred = read_pred("prediction_"+s2+"_"+s4+"_"+s1+"_"+s3+"_"+"nb"+".csv")
    return (pred)

def createListPred_svm(l,classifier, s1, s3, s4):
    s2 = ""
    pred = []
    if not os.path.isfile("prediction_"+s2+"_"+s4+"_"+s1+"_"+s3+"_"+"svm"+".csv"):
        acc = []
        for e in l :
            acc.append(e[3].values())
        pred = classifier.predict(acc)
        write_pred(pred,"prediction_"+s2+"_"+s4+"_"+s1+"_"+s3+"_"+"svm"+".csv")
    else :
        pred = read_pred("prediction_"+s2+"_"+s4+"_"+s1+"_"+s3+"_"+"svm"+".csv")
    return (pred)
#####################################################################

def detecteDoublons(liste):
    i = 0
    j = 0
    for elem in liste:
	i += 1
	if elem in liste[i:]:
	    j=j+1
    return j


def extract_polarity(s) :
    #extract all datas from csv_files and creation of a dataframe
    dfdataset = pd.read_csv(s, sep=',', engine='python')
    #extract colums
    dfpolar=dfdataset[dfdataset.columns[1]]
    #conversion of the dataframe of tweet in a list of tweets
    polar_l=dfpolar.tolist()
    res = []
    for e in polar_l :
        if e == "positive" :
            res.append(1)
        else :
            if e == "negative" :
                res.append(0)
            else :
                print "ALERTE ROUGE"
    return res

def write_errors_false_pos(s,string,pred) :
    dfdataset = pd.read_csv(string, sep=',', engine='python')
    dfpolar=dfdataset[dfdataset.columns[1]]
    dftweet=dfdataset[dfdataset.columns[2]]
    tweet_l=dftweet.tolist()
    polar_l=dfpolar.tolist()
    acc =[]
    for i in range(len (polar_l)) :
        acc.append([(polar_l[i]), (tweet_l[i])])
    res = []
    for i in range(len (pred)) :
        if pred[i] == 1 and acc[i][0] == "negative" :
            res.append(acc[i][1])
    c = csv.writer(open("false_pos__"+s, "wb"))
    for e in res :
        c.writerow([e])
    return res 

def write_errors_false_neg(s,string,pred) :
    dfdataset = pd.read_csv(string, sep=',', engine='python')
    dfpolar=dfdataset[dfdataset.columns[1]]
    dftweet=dfdataset[dfdataset.columns[2]]
    tweet_l=dftweet.tolist()
    polar_l=dfpolar.tolist()
    acc =[]
    for i in range(len (polar_l)) :
        acc.append([(polar_l[i]), (tweet_l[i])])
    res = []
    for i in range(len (pred)) :
        if pred[i] == 0 and acc[i][0] == "positive" :
            res.append(acc[i][1])
    c = csv.writer(open("false_neg__"+s, "wb"))
    for e in res :
        c.writerow([e])
    return res 

#####################################################################
#input :    
#output :   write a csv files with all the tf idf of all features
#date :     7/19/2018
#author :   Quentin Taillefer
def main () :
    trainset = raw_input("Dataset ? possible answer : train.csv or labeled.csv or unlabeled.csv  ")
    type_of_feature = raw_input("Feature ? possible answer : uni or uni_neg or uni_mw or all_feat  ")
    name_classifier = raw_input("classifier ? possible answer : nb or svm  ")
    erreur = raw_input("wrtitting of errors ? possible answer : y or n  ")
    
    if trainset == "train.csv" :
        name_set = "100_percent"
    else :
        if trainset == "labeled.csv" :
            name_set = "10_percent"
        else :
            name_set = "90_percent"
    feature_selection = "no_feat_sel"
    name = feature_selection + "_" + type_of_feature + "_" + name_set
    print ""
    print ""
    print name + "_" + name_classifier
    print ""
    print ""

    testset = "test.csv"
    test_set = "test"

    if not os.path.isfile("prediction_"+"_"+feature_selection+"_"+type_of_feature+"_"+name_set+"_"+name_classifier+".csv") :
        if trainset == "labeled.csv" : 
            print "Preprocessing..."
            tweets_l_train_big = extract_tweet("train.csv")
            tweets_l_c_train_big = cleaner_l(tweets_l_train_big,type_of_feature,"100_percent")
            tweets_l_train = extract_tweet(trainset)
            if detecteDoublons(tweets_l_train) != 0 : print "ALERTE ROUGE"
            tweets_l_test = extract_tweet(testset)
            if detecteDoublons(tweets_l_test) != 0 : print "ALERTE ROUGE"
            tweets_l_c_train = cleaner_l (tweets_l_train,type_of_feature, name_set)
            tweets_l_c_test = cleaner_l (tweets_l_test,type_of_feature, test_set)
            tweets_l_c_all = tweets_l_c_train + tweets_l_c_test

            print "Creation feature set"
            features_set = get_word_features(get_words_in_tweets(tweets_l_c_train_big),"0")
            dic_idf = create_dic_idf(features_set, tweets_l_c_all)
            print "Creation training set"
            training_set = createTrainingSet(tweets_l_c_train, features_set, tweets_l_c_all, dic_idf, type_of_feature, name_set, feature_selection, "")
        else :
            print "Preprocessing..."
            tweets_l_train = extract_tweet(trainset)
            if detecteDoublons(tweets_l_train) != 0 : print "ALERTE ROUGE"
            tweets_l_test = extract_tweet(testset)
            if detecteDoublons(tweets_l_test) != 0 : print "ALERTE ROUGE"
            tweets_l_c_train = cleaner_l (tweets_l_train,type_of_feature, name_set)
            tweets_l_c_test = cleaner_l (tweets_l_test,type_of_feature, test_set)
            tweets_l_c_all = tweets_l_c_train + tweets_l_c_test

            print "Creation feature set"
            features_set = get_word_features(get_words_in_tweets(tweets_l_c_train),feat_selec)
            dic_idf = create_dic_idf(features_set, tweets_l_c_all)
            print "Creation training set"
            training_set = createTrainingSet(tweets_l_c_train, features_set, tweets_l_c_all, dic_idf, type_of_feature, name_set, feature_selection, "")
            
        print "Creation of classifier"
        train_set_nb = []
        for e in training_set :
            train_set_nb.append((e[3],e[1]))
        data = []
        labels = []
        for e in training_set :
            data.append(e[3].values())
            if e[1] == "positive" :
                labels.append(1)
            else :
                labels.append(0)
        

        if name_classifier == "nb" :
            classifier_nb = nltk.NaiveBayesClassifier.train(train_set_nb)
        else :
            classifier_svm = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, random_state=42, max_iter=5, tol=None)
            classifier_svm.fit(data, labels)


        


        print "Creation test set"
        newSet = createTrainingSet(tweets_l_c_test, features_set, tweets_l_c_all, dic_idf, type_of_feature, name_set, feature_selection, "test")
        print "Calculation of accuracy and f-measure"
        if name_classifier == "nb" :
            pred = createListPred_nb(newSet, classifier_nb, type_of_feature, name_set, feature_selection)
        else :
            pred = createListPred_svm(newSet, classifier_svm, type_of_feature, name_set, feature_selection)
    else :
        print "Already done..."
        pred = read_pred("prediction_"+"_"+feature_selection+"_"+type_of_feature+"_"+name_set+"_"+name_classifier+".csv")

    true = extract_polarity("test.csv")
    print ""
    print "Results with test set : "
    print "Accuracy = " + str(accuracy_score(true,pred))
    print "Fscore = " + str(f1_score(true, pred))
    print "recall = " + str(recall_score(true, pred))
    print "precision = " + str (precision_score(true, pred))

    print ""
    if erreur == "y" :
        write_errors_false_pos(feature_selection+"_"+type_of_feature+"_"+name_set+"_"+name_classifier+".csv","test.csv",pred)
        write_errors_false_neg(feature_selection+"_"+type_of_feature+"_"+name_set+"_"+name_classifier+".csv","test.csv",pred)

    
    return (True)
#####################################################################


main()






