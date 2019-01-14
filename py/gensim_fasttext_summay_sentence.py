

from SmiToText.wordcheck.kktExtractNoun import extractNoun
from gensim.models import Word2Vec

from gensim.models import FastText

import py.w2vft_util as ut


if __name__ == '__main__':

    extract_noun = extractNoun()

    ##### 파일 읽어오기

    counter = 0
    fasttext_data = []

    for index in range(0, 750):
        # intput_fname = '../node/WIKI_DATA/' + str(index) + '.txt'
        intput_fname = './NEWS_DATA/' + str(index) + '.txt'
        output_fname = intput_fname + '_norm.txt'

        texts = ut.normalizeFile(intput_fname, output_fname)
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

        fasttext_data.append(nn_word_list)

    fastText_model = ut.fastText(fasttext_data)



    print('fastText_model similar list')
    # similar = model.most_similar(positive=['윤호', '하이킥'], topn=10)
    # [('순재', 0.9943705797195435), ('거침없이', 0.9900286197662354), ('그에게', 0.9879124164581299), ('중매역활', 0.9861310720443726), ('하거나', 0.9786599278450012), ('자이젠', 0.9707398414611816), ('민정은', 0.9691370725631714), ('프리실라', 0.9552605152130127), ('시온', 0.954103946685791), ('타바사', 0.9522706866264343)]
    # similar = model.most_similar(positive=['카파도키아', '아르메니아', '기원전'], topn=10)
    # [('에우메네스', 0.9945849180221558), ('페르디카스로부터', 0.9932612180709839), ('공격하', 0.9814687967300415), ('받아', 0.9809004068374634), ('알케타스', 0.9726078510284424), ('321년', 0.97102952003479), ('마족', 0.9679989814758301), ('영웅전', 0.9672538638114929), ('것이다', 0.9660188555717468), ('에린이', 0.9653569459915161)]
    similar = fastText_model.most_similar(positive=['삼성'], negative=['제로페이'], topn=10)

    print(similar)

    ut.plt_show(fastText_model, img_name='fasttext.png')

    """
    Just run `py w2v_visualizer.py word2vec.model visualize_result`
    """


    word2vec_model = FastText.load("./fastText.model")
    ut.visualize(word2vec_model, "./fastText_log")
