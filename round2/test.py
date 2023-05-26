import sys
from vectorize import ForwardMaxMatch


if __name__ == '__main__':

    tokenizer = ForwardMaxMatch('./sgns.weibo.word')

    with open('./train.csv', 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split(',')
            label = items[0]
            sentence = ','.join(items[1:])
            words = tokenizer.cut(sentence)
            print(words)
