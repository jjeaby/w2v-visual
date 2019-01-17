import re

from SmiToText.tokenizer.nltk import nltkSentTokenizer




import py.w2vft_util as ut
''


def remove_naver_news( text):
    # def sub(pattern, repl, string, count=0, flags=0):
    text = re.sub(r'function _flash_removeCallback\(\) \{\}', ' ', text)
    text = re.sub(r'\/\/ flash 오류를 우회하기 위한 함수 추가', ' ', text)
    text = re.sub(
        r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
        ' ', text)
    text = re.sub(r'다\.', '다. ', text)
    return text

def remove_keyboard_out_chractor( text):
    text = re.sub(r'\'', '\'', text)
    text = re.sub(r'!', '!', text)
    text = re.sub(r'"', '"', text)
    text = re.sub(r'`', '`', text)
    text = re.sub(r'＇', '\'', text)
    text = re.sub(r'＇', '\'', text)
    text = re.sub(r'｀', '`', text)
    text = re.sub(r'´', '\'', text)
    text = re.sub(r'˙', '\'', text)
    text = re.sub(r'˝', '\"', text)
    text = re.sub(r'＂', '\"', text)
    text = re.sub(r'”', '\"', text)
    text = re.sub(r'“', '\"', text)
    text = re.sub(r'”', '\"', text)
    text = re.sub(r'‘', '\'', text)
    text = re.sub(r'’', '\'', text)
    text = re.sub(r'′', '\'', text)
    text = re.sub(r'″', '\"', text)

    return text




if __name__ == '__main__':


    ##### 파일 읽어오기

    counter = 0
    fasttext_data = []

     # intput_fname = '../node/WIKI_DATA/' + str(index) + '.txt'
    intput_fname = './naver/naver_news_economy.txt'
    output_fname = intput_fname + '_norm.txt'

    input_file = open(intput_fname, mode='r', encoding='utf-8'    )

    line_number = 1
    while(True):
        text = input_file.readline()
        if not text :
            break



        text = remove_keyboard_out_chractor(text)
        text = remove_naver_news(text)
        
        if  len(re.findall('function', text)) > 1 or len(re.findall('var currentDateParam',text)) > 0 :
            continue

        line_sentence = []
        for text_item in text.split("\n"):
            sentences = nltkSentTokenizer(text_item)

            for sent in sentences:

                for line in sent.split(r'\n'):
                    if  line.strip() :
                        line = ut.normalizeText(line)
                        line_sentence.append(line)



        if  len(line_sentence) < 3 :
            continue

        print(line_number, line_sentence[:-2])
        line_number = line_number +1

        contents = (" ".join(line_sentence[1:-2]).strip())
        if len(contents) > 0 :
            output_file = open(output_fname, mode='a+', encoding='utf-8')
            output_file.write(line_sentence[0] + " ∥ " + " ".join(line_sentence[1:-2]) + "\n")
            output_file.close()


