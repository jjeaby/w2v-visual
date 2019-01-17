# -*- coding: utf-8 -*-
from pickle import dump, load

import numpy as np
from keras import Input, Model
from keras.layers import LSTM, Dense
from py.w2vft_util import sumy


input_fname = '../data/naver/naver_news_economy_norm_sort.txt'
output_fname = '../data/naver/naver_news_economy_norm_sort_sumy.txt'

source = open(input_fname, mode='r', encoding='utf-8')
target = open(output_fname, mode='w', encoding='utf-8')

line_number = 1
src_maxlength = 0
tgt_maxlength = 0
while (True):
    line = source.readline()
    if not line:
        break
    line_array = line.split("∥")

    src_news = sumy(line_array[1].strip())
    tgt_news = line_array[0].strip()

    if(len(src_news) == 0 ) :
        continue

    if len(src_news) > src_maxlength:
        src_maxlength = len(src_news)
    if len(tgt_news) > tgt_maxlength:
        tgt_maxlength = len(tgt_news)

    target.write(tgt_news + "|" + src_news + "\n")

    print("line number:", line_number)
    line_number = line_number +1

print("*", line_number)
print("*", src_maxlength)
print("*", tgt_maxlength)

