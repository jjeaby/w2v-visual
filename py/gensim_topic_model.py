
import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from gensim import corpora
import pickle
import gensim


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)




def prepare_text_for_lda(text):

    read_stop_words = open('ko_stop_words.txt', mode='r', encoding='utf-8')


    ko_stop = read_stop_words.readlines()


    for idx, ko in enumerate(ko_stop):
        ko_stop[idx] = ko.replace('\n','')


    tokens = text.split(' ')
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in ko_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


documents = []
text_data = []
for index in range(0, 100):
    # intput_fname = '../node/WIKI_DATA/' + str(index) + '.txt'
    intput_fname = './NEWS_DATA/' + str(index) + '.txt_norm.txt'
    input_file = open(intput_fname, mode='r', encoding='utf-8')
    text = input_file.read().replace('\n', ' ')
    documents.append(text)
    print(text)
    tokens = prepare_text_for_lda(text)
    print(tokens)
    text_data.append(tokens)


dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

new_doc = 'Practical Bayesian Optimization of Machine Learning Algorithms'
new_doc = prepare_text_for_lda(new_doc)
new_doc_bow = dictionary.doc2bow(new_doc)
print(new_doc_bow)
print(ldamodel.get_document_topics(new_doc_bow))


ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 3, id2word=dictionary, passes=15)
ldamodel.save('model3.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)