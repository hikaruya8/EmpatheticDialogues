# coding: UTF-8
import pdb
import json
from pprint import pprint


def build_dict_from_reddit_raw_data():
    f = open('../../reddit-folder/raw_data', 'r')
    line = f.readline()
    word_dictionary = [json.loads(f.readline()) for i,l in enumerate(line) if i < 4]
    f.close
    pprint(word_dictionary)

    '''
    words: dict with words as keys and word idxs as tokens
    iwords: list of words in order (where the index of each word is given by words, above)
    wordcounts: 1D Tensor indexed the same way as iwords, where each value is the frequency of that word in the corpus
    '''
    words = [w['body'] if w['body'] != '[deleted]' else None for w in word_dictionary]
    # pdb.set_trace()
if __name__ == '__main__':
    build_dict_from_reddit_raw_data()
    # json_load()
