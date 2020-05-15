# coding: UTF-8
import pdb
from pprint import pprint

def build_dict_from_reddit_raw_data():
    f = open('../../reddit-folder/raw_data', 'r')
    line = f.readline()
    word_dictionary = [f.readline() for i,l in enumerate(line) if i < 4]

    # n = 0
    # while line:
    #     print(line)
    #     line = f.readline()
    #     n += 1
    #     if n > 3:
    #         break

    f.close
    pprint(word_dictionary)

if __name__ == '__main__':
    build_dict_from_reddit_raw_data()
