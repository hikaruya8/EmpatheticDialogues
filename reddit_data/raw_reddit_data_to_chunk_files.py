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

import time
from tqdm import tqdm
import ipdb

np.random.seed(42)
random.seed(42)

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


def convert_raw_reddit_to_chunk_files(max_length=100):
    W = data.Field(
        sequential=True, tokenize=tokenizer_with_preprocessing, lower=True, use_vocab=False, include_lengths=True, batch_first=True, fix_length=max_length, init_token='cstart', eos_token='cend')
    UID =data.Field(
        sequential=False)
    P2C =data.Field(
        sequential=False)
    fields = {'body': ('w', W),
                'id': ('uid', UID)}

    chunk_num = 1000
    chunk_id = 998
    chunked_ds = [''] * chunk_num

    with tqdm() as pbar:
        while chunk_id < chunk_num:
            chunk_zero_fill = str(chunk_id).zfill(3)
            chunked_ds[chunk_id] = data.TabularDataset(
                path=os.path.join(f'../reddit_raw_data_folder/chunk{chunk_zero_fill}.pth'),
                format='json',
                fields=fields)

            vectors = Vectors(name='../crawl-300d-2M.vec')
            W.build_vocab(chunked_ds[chunk_id], vectors=vectors)
            print(W.vocab.vectors.shape)
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

            chunked_word_dictionary = {}
            chunked_word_dictionary["w"] = [x for x in chunked_ds[chunk_id].w]
            chunked_word_dictionary["uid"] = [x for x in chunked_ds[chunk_id].uid]
            chunked_word_dictionary["p2c"] = chunked_word_dictionary["uid"][:-1]
            chunked_word_dictionary["cstart"] = [x for x in chunked_ds[chunk_id].cstart]
            chunked_word_dictionary["cend"] = [x for x in chunked_ds[chunk_id].cend]
            chunked_word_dictionary["words"] = words
            chunked_word_dictionary["iwords"] = iwords
            chunked_word_dictionary["wordcounts"] = wordcounts

            # ipdb.set_trace()

            with open(os.path.join(f'./chunk{chunk_zero_fill}.pth'), mode='wb') as f:
                torch.save(chunked_word_dictionary, f)


            time.sleep(0.05)
            pbar.update(1)
            chunk_id += 1



def check_word_dictionary():
    word_dictionary = torch.load('./word_dictionary')
    ipdb.set_trace()

if __name__ == '__main__':
    convert_raw_reddit_to_chunk_files()
