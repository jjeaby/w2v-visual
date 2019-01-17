import math
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pylab
from collections import Counter
import csv

# Seq2Seq Items
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.layers.core import Dense

from nltk.tokenize import wordpunct_tokenize
from SmiToText.tokenizer.mecab import mecabTokenizer

vocab_size = 50000
num_units = 128
input_size = 128
batch_size = 16
source_sequence_length = 40
target_sequence_length = 60
decoder_type = 'basic'  # could be basic or attention
sentences_to_read = 2000

tgt_word_maxlength = 300

max_tgt_sent_lengths = 0

src_max_sent_length = tgt_word_maxlength
tgt_max_sent_length = tgt_word_maxlength



# vocab 만들기
def make_vocab():
    src_vocab_set = set()
    tgt_vocab_set = set()

    src_dictionary = dict()
    tgt_dictionary = dict()


    with open('../data/naver/naver_news_economy_norm_sort.txt', mode='r', encoding='utf-8') as input_fname:
        for line in input_fname:
            line_array = line.split("∥")

            if not line_array[1].strip() == "" and len(line_array[1]) <= tgt_word_maxlength:

                src_word = wordpunct_tokenize(line_array[1].strip())
                tgt_word = wordpunct_tokenize(line_array[0].strip())
                # src_word = mecabTokenizer(line_array[1].strip())
                # tgt_word = mecabTokenizer(line_array[0].strip())

                for word in src_word:
                    src_vocab_set.add(word)

                for word in tgt_word:
                    tgt_vocab_set.add(word)
    src_vocab = open("../data/naver/naver_news_economy_norm_sort_src_vocab.txt", mode="w", encoding="utf-8")
    for word_1 in src_vocab_set:
        word_2 = word_1.split("'")
        for word in word_2:
            if len(str(word).strip()) > 0:
                src_vocab.write(str(word) + "\n")
    # for word in src_vocab_set:
    #     if len(str(word).strip()) > 0:
    #         src_vocab.write(str(word) + "\n")
    src_vocab.write("<unk>\n")
    src_vocab.write("<s>\n")
    src_vocab.write("</s>\n")

    tgt_vocab = open("../data/naver/naver_news_economy_norm_sort_tgt_vocab.txt", mode="w", encoding="utf-8")
    for word_1 in tgt_vocab_set:
        word_2 = word_1.split("'")
        for word in word_2:
            if len(str(word).strip()) > 0:
                tgt_vocab.write(str(word) + "\n")
    # for word in tgt_vocab_set:
    #     if len(str(word).strip()) > 0:
    #         src_vocab.write(str(word) + "\n")
    tgt_vocab.write("<unk>\n")
    tgt_vocab.write("<s>\n")
    tgt_vocab.write("</s>\n")

    src_vocab.close()
    tgt_vocab.close()

    with open('../data/naver/naver_news_economy_norm_sort_src_vocab.txt', mode='r', encoding='utf-8') as fname:
        for src_line in fname:
            src_dictionary[src_line[:-1]] = len(src_line)
    src_reverse_dictionary = dict(zip(src_dictionary.values(), src_dictionary.keys()))

    with open('../data/naver/naver_news_economy_norm_sort_tgt_vocab.txt', mode='r', encoding='utf-8') as fname:
        for tgt_line in fname:
            tgt_dictionary[tgt_line[:-1]] = len(tgt_line)
    tgt_reverse_dictionary = dict(zip(tgt_dictionary.values(), tgt_dictionary.keys()))

    print('Source')
    print('\t', list(src_dictionary.items())[:10])
    print('\t', list(src_reverse_dictionary.items())[:10])
    print('\t', 'Vocabulary size: ', len(src_dictionary))

    print('Target')
    print('\t', list(tgt_dictionary.items())[:10])
    print('\t', list(tgt_reverse_dictionary.items())[:10])
    print('\t', 'Vocabulary size: ', len(tgt_dictionary))

    return src_dictionary, tgt_dictionary, src_reverse_dictionary, tgt_reverse_dictionary


# vocab 만들기
src_dictionary, tgt_dictionary, src_reverse_dictionary, tgt_reverse_dictionary = make_vocab()

# Loading Sentences
source_sent = []
target_sent = []

test_source_sent = []
test_target_sent = []

with open('../data/naver/naver_news_economy_norm_sort.txt', mode='r', encoding='utf-8') as input_fname:
    for line in input_fname:
        line_array = line.split("∥")

        src_line = line_array[1].strip()
        tgt_line = line_array[0].strip()

        if len(source_sent) >= sentences_to_read or len(target_sent) >= sentences_to_read:
            break

        source_sent.append(src_line)
        target_sent.append(tgt_line)

assert len(source_sent) == len(target_sent), 'Source: %d, Target: %d' % (len(source_sent), len(target_sent))

print('Sample translations (%d)' % len(source_sent))
for i in range(0, sentences_to_read, 10000):
    print('(', i, ') SRC: ', source_sent[i])
    print('(', i, ') TGT: ', target_sent[i])


def split_to_tokens(sent, is_source):
    # sent = sent.replace('-',' ')


    sent_toks = wordpunct_tokenize(sent)
    for t_i, tok in enumerate(sent_toks):
        if is_source:
            if tok not in src_dictionary.keys():
                sent_toks[t_i] = '<unk>'
        else:
            if tok not in tgt_dictionary.keys():
                sent_toks[t_i] = '<unk>'
    return sent_toks


# Let us first look at some statistics of the sentences
source_len = []
source_mean, source_std = 0, 0
for sent in source_sent:
    source_len.append(len(split_to_tokens(sent, True)))

print('(Source) Sentence mean length: ', np.mean(source_len))
print('(Source) Sentence stddev length: ', np.std(source_len))

target_len = []
target_mean, target_std = 0, 0
for sent in target_sent:
    target_len.append(len(split_to_tokens(sent, False)))

print('(Target) Sentence mean length: ', np.mean(target_len))
print('(Target) Sentence stddev length: ', np.std(target_len))

####

train_inputs = []
train_outputs = []
train_inp_lengths = []
train_out_lengths = []

for s_i, (src_sent, tgt_sent) in enumerate(zip(source_sent, target_sent)):

    src_sent_tokens = split_to_tokens(src_sent, True)
    tgt_sent_tokens = split_to_tokens(tgt_sent, False)

    num_src_sent = []
    for tok in src_sent_tokens:
        num_src_sent.append(src_dictionary[tok])

    num_src_set = num_src_sent[::-1]  # we reverse the source sentence. This improves performance
    num_src_sent.insert(0, src_dictionary['<s>'])
    train_inp_lengths.append(min(len(num_src_sent) + 1, src_max_sent_length))

    # append until the sentence reaches max length
    if len(num_src_sent) < src_max_sent_length:
        num_src_sent.extend([src_dictionary['</s>'] for _ in range(src_max_sent_length - len(num_src_sent))])
    # if more than max length, truncate the sentence
    elif len(num_src_sent) > src_max_sent_length:
        num_src_sent = num_src_sent[:src_max_sent_length]
    assert len(num_src_sent) == src_max_sent_length, len(num_src_sent)

    train_inputs.append(num_src_sent)

    num_tgt_sent = [tgt_dictionary['</s>']]
    for tok in tgt_sent_tokens:
        num_tgt_sent.append(tgt_dictionary[tok])

    train_out_lengths.append(min(len(num_tgt_sent) + 1, tgt_max_sent_length))

    if len(num_tgt_sent) < tgt_max_sent_length:
        num_tgt_sent.extend([tgt_dictionary['</s>'] for _ in range(tgt_max_sent_length - len(num_tgt_sent))])
    elif len(num_tgt_sent) > tgt_max_sent_length:
        num_tgt_sent = num_tgt_sent[:tgt_max_sent_length]

    train_outputs.append(num_tgt_sent)

assert len(train_inputs) == len(source_sent), \
    'Size of total bin elements: %d, Total sentences: %d' \
    % (len(train_inputs), len(source_sent))

print('Max sent lengths: ', max_tgt_sent_lengths)

train_inputs = np.array(train_inputs, dtype=np.int32)
train_outputs = np.array(train_outputs, dtype=np.int32)
train_inp_lengths = np.array(train_inp_lengths, dtype=np.int32)
train_out_lengths = np.array(train_out_lengths, dtype=np.int32)
print('Samples from bin')
print('\t', [src_reverse_dictionary[w] for w in train_inputs[0, :].tolist()])
print('\t', [tgt_reverse_dictionary[w] for w in train_outputs[0, :].tolist()])
print('\t', [src_reverse_dictionary[w] for w in train_inputs[10, :].tolist()])
print('\t', [tgt_reverse_dictionary[w] for w in train_outputs[10, :].tolist()])
print()
print('\tSentences ', train_inputs.shape[0])
