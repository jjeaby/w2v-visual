from collections import Counter

from SmiToText.find.extract_noun import expect_noun_text
from SmiToText.wordcheck.kktExtractNoun import extractNoun
from krwordrank.hangle import normalize
from krwordrank.word import KRWordRank

import sys, os
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector


def visualize(model, output_path):
    meta_file = "w2x_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), model.vector_size))  # 'model.vector_size' used to be '100'
    # I needed to change '100' to 'model.vector_size' to accommodate generalized sizes of word vectors.
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    # Nothing changed below this point.
    with open(os.path.join(output_path,meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable = False, name = 'w2x_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path,'w2x_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))






def get_texts(fname):
    with open(fname, encoding='utf-8') as f:
        docs = [doc.lower() for doc in f]

        if not docs:
            return []

        return docs


def normalizeCF(input_fname, output_fname):
    texts = get_texts(input_fname)
    with open(output_fname, 'w', encoding='utf-8') as f:
        for text in texts:
            text = normalize(text, english=True, number=True)
            text0 = text

            # # # 명사 추출 1 번
            noun_text = expect_noun_text(text)
            text1 = ' '.join(noun_text)
            text = text0 + text1
            #
            # # # 명사 추출 2 번
            # noun_text = extract_noun.findKoNoun(text)
            # noun_text_list = noun_text[0] + noun_text[1]
            # text2 = ' '.join(noun_text_list)
            # text = text0 + ' ' +  text1 + ' ' + text2

            if text.strip() == '':
                continue

            print('*' * 10, text)

            f.write('%s\n' % (text))
    return (texts)





extract_noun = extractNoun()


if __name__ == '__main__':

    ##### 파일 읽어오기

    counter = 0
    raw_data = []
    for index in range(0, 750):

        # intput_fname = '../node/WIKI_DATA/' + str(index) + '.txt'
        intput_fname = './NEWS_DATA/' + str(index) + '.txt'
        output_fname = intput_fname + '_norm.txt'

        texts = normalizeCF(intput_fname, output_fname)
        print(texts)

        texts = get_texts(output_fname)

        # if len(texts) == 0 or texts[0].startswith('redirect') or texts[0].startswith('#redirect'):
        #     # or  len( texts[0])<50:
        #     print('continue')
        #     continue

        counter = counter + 1

        print("-" * 100)
        print(counter)
        print("-" * 100)
        print(counter, texts)

        wordrank_extractor = KRWordRank(
            min_count=2,  # 단어의 최소 출현 빈도수 (그래프 생성 시)
            max_length=10,  # 단어의 최대 길이
            verbose=True
        )

        beta = 0.85  # PageRank의 decaying factor beta
        max_iter = 10

        keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)

        temp_data = []
        for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[: 10]:
            print('%8s:\t%.4f' % (word, r))
            temp_data.append(word)
        # raw_data.append(' '.join(temp_data))
        raw_data.append(temp_data)
        print("=" * 100)
        print('raw_data', raw_data)
        print("=" * 100)

    ## 현재 문서를 제외한 다른 문서의 keyword 를 구한다
    # print('counter', counter)
    remove_data = []
    for index_i in range(0, counter):
        remove_data.append('')
        for index_j in range(0, counter):
            if index_i != index_j:
                remove_data[index_i] = str(remove_data[index_i]) + ' ' + ' '.join(raw_data[index_j])

        # print(index_i )

    ## 현재 문서를 제외한 다른 문서의 keyword 의 반복 횟수를 구한다

    confirm_remove_data = []
    for index_i in range(0, counter):
        confirm_remove_data.append([])
        counter_word = Counter(remove_data[index_i].strip().split(' ')).most_common()

        confirm_remove_data.append([])
        for index, word in enumerate(counter_word):
            if index == 10:
                break;

            confirm_remove_data[index_i].append(word)

        print(index_i, confirm_remove_data[index_i])

    print("-" * 100)

    word2vec_data = []

    for index_i in range(0, counter):
        for idx, remove_data in enumerate(confirm_remove_data[index_i]):
            if idx in range(0, 1, 2) or idx in range(len(confirm_remove_data) - 2,
                                                     len(confirm_remove_data) - 1,
                                                     len(confirm_remove_data)):
                continue

            s = set(remove_data[0])
            result = [x for x in raw_data[index_i] if x not in s]  # 순서 보존됨
            # print(raw_data[index_i])
        word2vec_data.append(result)
        print(index_i, result)

    print(word2vec_data)

# Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec

path = get_tmpfile("word2vec.model")

model = Word2Vec(word2vec_data, size=200, window=5, min_count=1, workers=4, iter=100, sg=1)
model.save("word2vec.model")

print('words list')
words = list(model.wv.vocab)
print(words)

## Visualize Word Embedding

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

print('버전: ', mpl.__version__)
print('설치 위치: ', mpl.__file__)
print('설정 위치: ', mpl.get_configdir())
print('캐시 위치: ', mpl.get_cachedir())

font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
[print((f.name, f.fname)) for f in fm.fontManager.ttflist if 'Nanum' in f.name]

path = '/Users/actmember/Library/Fonts/NanumBarunGothic.otf'
fontprop = fm.FontProperties(fname=path, size=8)

mpl.rcParams['axes.unicode_minus'] = False
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]), fontproperties=fontprop)
plt.savefig("test.png", quality=100, dpi=500)
plt.show()

print('similar list')
# similar = model.most_similar(positive=['윤호', '하이킥'], topn=10)
# [('순재', 0.9943705797195435), ('거침없이', 0.9900286197662354), ('그에게', 0.9879124164581299), ('중매역활', 0.9861310720443726), ('하거나', 0.9786599278450012), ('자이젠', 0.9707398414611816), ('민정은', 0.9691370725631714), ('프리실라', 0.9552605152130127), ('시온', 0.954103946685791), ('타바사', 0.9522706866264343)]

# similar = model.most_similar(positive=['카파도키아', '아르메니아', '기원전'], topn=10)
# [('에우메네스', 0.9945849180221558), ('페르디카스로부터', 0.9932612180709839), ('공격하', 0.9814687967300415), ('받아', 0.9809004068374634), ('알케타스', 0.9726078510284424), ('321년', 0.97102952003479), ('마족', 0.9679989814758301), ('영웅전', 0.9672538638114929), ('것이다', 0.9660188555717468), ('에린이', 0.9653569459915161)]


similar = model.most_similar(positive=['유료방송', 'kt가'], topn=10)
print(similar)



"""
Just run `py w2v_visualizer.py word2vec.model visualize_result`
"""

model = Word2Vec.load("./word2vec.model")
visualize(model, "./log")



