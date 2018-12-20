import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import os, sys, email
import gensim
from gensim.models import Doc2Vec
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from string import punctuation
import timeit
from sklearn.cluster import KMeans
from sklearn import metrics
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import python.w2vft_util as ut


start = timeit.default_timer()


doc2vector_data = []
counter = 0

for index in range(0, 750):
    # intput_fname = '../node/WIKI_DATA/' + str(index) + '.txt'
    intput_fname = './NEWS_DATA/' + str(index) + '.txt'
    output_fname = intput_fname + '_norm.txt'

    texts = ut.normalizeCF(intput_fname, output_fname)
    print("*" * 100)
    print(texts)
    print("*" * 100)

    texts = ut.get_texts(output_fname)

    counter = counter + 1

    print("-" * 100)
    print(counter, texts)
    print("-" * 100)

    summay_text = ut.sumy(" ".join(texts), COUNT=2)

    print("=" * 10)
    print(summay_text)
    print("=" * 10)

    nn_word_list = ut.kakao_postagger_nn_finder(summay_text)

    print(nn_word_list)

    doc2vector_data.append(nn_word_list)






LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
all_content = []
texts = []
j = 0
for docu in doc2vector_data:
    all_content.append(LabeledSentence1(docu, [j]))
    j+= 1

print("Number of docu vectors: ", j)
print(all_content[278])

d2v_model = Doc2Vec(all_content, size = 2000, window = 10, workers=7, dm = 1,  alpha=0.025, min_alpha=0.001)
d2v_model.train(all_content, total_examples=d2v_model.corpus_count, epochs=10, start_alpha=0.002, end_alpha=-0.016)

d2v_model.save('doc2vec.model')
d2v_model = Doc2Vec.load('doc2vec.model')


print (d2v_model.docvecs.most_similar(1))



# kmean cluster

kmeans_model = KMeans(n_clusters=8, init='k-means++', max_iter=100)
X = kmeans_model.fit(d2v_model.docvecs.doctag_syn0)
labels=kmeans_model.labels_.tolist()

l = kmeans_model.fit_predict(d2v_model.docvecs.doctag_syn0)
pca = PCA(n_components=2).fit(d2v_model.docvecs.doctag_syn0)
datapoint = pca.transform(d2v_model.docvecs.doctag_syn0)



# kmeas show plt
label1 = ["#660000", "#AA0000", "#CC0000", "#00AA00", "#006600", "#00CC00", "#0000AA", "#000066"]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.savefig('kmeandoc2vec.png', quality=100, dpi=500)

plt.show()

stop = timeit.default_timer()
execution_time = stop - start

print(execution_time) #It returns time in sec