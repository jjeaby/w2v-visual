from pickle import dump, load

import numpy as np
from keras import Input, Model
from keras.layers import LSTM, Dense


def save_train_data():
    stories = list()

    input_fname = './naver/naver_news_economy.txt_norm.txt_sort.txt'

    source = open(input_fname, mode='r', encoding='utf-8')

    while (True):
        line = source.readline()
        if not line:
            break
        line_array = line.split("∥")
        stories.append({'story': line_array[1], 'highlights': line_array[0]})

    dump(stories, open('./naver/naver_news_economy.pkl', 'wb'))


batch_size = 64
epochs = 110
latent_dim = 256
num_samples = 10000

stories = load(open('./naver/naver_news_economy.pkl', 'rb'))
print('Loaded Stories %d' % len(stories))
print(type(stories))

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
for story in stories:
    input_text = story['story']
    target_text = story['highlights']
 
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

print("len(input_texts):", len(input_texts))
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')




for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    print("i:", i, "input_text:", input_text, "target_text:", target_texts)
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
