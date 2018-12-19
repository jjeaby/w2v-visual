from collections import Counter


from krwordrank.hangle import normalize
from krwordrank.word import KRWordRank

import torch
from torch.optim import SGD
from torch.autograd import Variable, profiler
import numpy as np
import torch.functional as F
import torch.nn.functional as F

from SmiToText.find.extract_noun import expect_noun_text
from SmiToText.wordcheck.kktExtractNoun import extractNoun
extract_noun = extractNoun()


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
        word2vec_data.append(' '.join(result))
        print(index_i, result)

    print('word2vec_data')
    print(word2vec_data)

# Word2Vec
