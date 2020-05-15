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

if __name__ == '__main__':
    build_dict_from_reddit_raw_data()
    # json_load()
