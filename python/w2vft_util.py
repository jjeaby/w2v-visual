import os

import khaiii
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from krwordrank.hangle import normalize
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words
from tensorflow.contrib.tensorboard.plugins import projector
from gensim.models import FastText



def visualize(model, output_path):
    meta_file = "w2x_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), model.vector_size))  # 'model.vector_size' used to be '100'
    # I needed to change '100' to 'model.vector_size' to accommodate generalized sizes of word vectors.
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    # Nothing changed below this point.
    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
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

    embedding = tf.Variable(placeholder, trainable=False, name='w2x_metadata')
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
    saver.save(sess, os.path.join(output_path, 'w2x_metadata.ckpt'))
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
            text = normalize(text, english=True, number=True, punctuation=True)
            text0 = text

            # # # 명사 추출 1 번
            # noun_text = expect_noun_text(text)
            # text1 = ' '.join(noun_text)
            # text = text0 + text1
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




def sumy(text, LANGUAGE='english', COUNT=2):
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    # summarizer = TextRankSummarizer()
    summarizer = TextRankSummarizer(Stemmer(LANGUAGE))
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summay_text = ""
    for sentence in summarizer(parser.document, COUNT):
        summay_text = summay_text + " " + str(sentence)
    summay_text = summay_text.strip()
    # summay_text = re.sub(' +', ' ', summay_text)

    return summay_text


def kakao_postagger_nn_finder(summay_text):
    api = khaiii.KhaiiiApi()
    api.open()
    nn_word_list = []
    for word in api.analyze(summay_text):
        morphs_str = ' + '.join([(m.lex + '/' + m.tag) for m in word.morphs])
        # print(f'{word.lex}\t{morphs_str}')

        morphs_str_list = morphs_str.split(" + ")

        complex_morphs = ""
        for mophs_item in morphs_str_list:
            if mophs_item.split("/")[1].startswith("N") or mophs_item.split("/")[1].startswith("MM") or \
                    mophs_item.split("/")[1].startswith("SN") or mophs_item.split("/")[1].startswith("SL"):
                complex_morphs = complex_morphs + mophs_item.split("/")[0]

        if len(complex_morphs) > 1:
            # print("->", complex_morphs)
            nn_word_list.append(complex_morphs)

    return nn_word_list




def word2vec(word2vec_data):
    # Word2Vec
    # model_ted = Word2Vec(sentences=sentences_ted, size=100, window=5, min_count=5, workers=4, sg=0)


    model = Word2Vec(word2vec_data, size=200, window=5, min_count=5, workers=4, iter=100, sg=0)
    model.save("word2vec.model")
    words = list(model.wv.vocab)

    print('words list')
    print(words)

    return model





def fastText(fastText_data):
    # Word2Vec
    # model_ted = Word2Vec(sentences=sentences_ted, size=100, window=5, min_count=5, workers=4, sg=0)

    model = FastText(fastText_data, size=200, window=5, min_count=5, workers=4, iter=100, sg=0)
    model.save("fastText.model")
    words = list(model.wv.vocab)

    print('words list')
    print(words)

    return model


def plt_show(model,img_name = "plt.png"):
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
    plt.savefig(img_name, quality=100, dpi=500)
    plt.show()



