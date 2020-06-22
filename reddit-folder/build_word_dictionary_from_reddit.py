#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
from torchtext import data
from torchtext.vocab import Vectors
import json
import string
import os

np.random.seed(42)
random.seed(42)

# def make_chunk_from_reddit():
#     with open('./raw_data', 'r') as infile:
#         o = json.load(infile)
#         chunk_size = 1000
#         for i in range(0, len(o), chunk_size):
#             ipdb.set_trace()
#             with open('chunk' + i + '.pth', 'w') as outfile:
#                 json.dump(o[i:i+chunk_size], outfile)


# # make my dataset
# class RedditDataset(data.Dataset):
#     def __init__(self, filename):
#         self._filename = filename
#         self._total_data = 0
#         self._total_data = int(subprocess.check_output("wc -l " + filename, shell=True).split()[0])

#     def __getitem__(self, idx):
#         with open(self._filename) as f:
#             return json.loads(f.readline())['body']
#             # 毎回lineの最初がloadされるので修正必要

#     def __len__(self):
#         return self._total_data


def preprocessing_text(text):
    # カンマ、ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

        # ピリオドなどの前後にはスペースを入れておく
        text = text.replace(".", " . ")
        text = text.replace(",", " , ")

        return text

# 分かち書き
def tokenizer_punctuation(text):
    return text.strip().split()

# 前処理と分かち書きをまとめる
def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_punctuation(text)
    return ret


def get_reddit_dic(file_path, max_length=256, batch_size=64):
    W = data.Field(
        sequential=True, tokenize=tokenizer_with_preprocessing, lower=True, use_vocab=False, include_lengths=True, batch_first=True, fix_length=max_length, init_token='cstart', eos_token='cend')
    fields = {'body': ('w', W)}
    ds = data.TabularDataset(
        path=file_path, format='json',
        fields=fields)

    # test dataloader
    # print(f'データ数{len(ds)}')
    # print(f'1つ目のデータ{vars(ds[0])}')

    vectors = Vectors(name='../crawl-300d-2M.vec')
    W.build_vocab(ds, vectors=vectors)
    # print(W.vocab.vectors.shape)
    # print(W.vocab.vectors)
    # print(W.vocab.stoi)

    '''
    words: dict with words as keys and word idxs as tokens
    iwords: list of words in order (where the index of each word is given by words, above)
    wordcounts: 1D Tensor indexed the same way as iwords, where each value is the frequency of that word in the corpus
    '''
    words = W.vocab.stoi
    iwords = W.vocab.itos
    wordcounts = [v for v in dict.values(W.vocab.freqs)]
    word_dictionary = {}
    word_dictionary['words'] = words
    word_dictionary['iwords'] = iwords
    word_dictionary['wordcounts'] = wordcounts

    with open('./word_dictionary', mode='wb') as f:
        torch.save(word_dictionary, f)


if __name__ == '__main__':
    # test_data = RedditDataset('./chunk000.pth')
    ```
    # test with chunked reddit data
    # file_path='./chunk000.pth'
    file_path = './raw_data'
    get_reddit_dic(file_path=file_path)