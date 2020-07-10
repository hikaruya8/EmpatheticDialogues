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

from transformers import BertTokenizer
import time
from tqdm import tqdm
import ipdb

np.random.seed(42)
random.seed(42)
pre_trained_weights = 'bert-base-uncased'

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


import collections
def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def convert_raw_reddit_to_chunk_files(max_length=100):
    # torchtext用にBertTokenizerの準備
    tokenizer = BertTokenizer.from_pretrained(pre_trained_weights)
    pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    # eos_index = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    unk_index = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    mask_index = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    cls_index = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    # sep_index = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

    W = data.Field(sequential=True, tokenize=tokenizer.tokenize, use_vocab=False, include_lengths=True, batch_first=True, fix_length=max_length, init_token=cls_index, pad_token=pad_index, unk_token=unk_index)
    UID =data.Field(
        sequential=False)
    LID =data.Field(
        sequential=False)
    PID =data.Field(
        sequential=False)
    fields = {'body': ('w', W),
                'id': ('uid', UID),
                'link_id': ('lid', LID),
                'parent_id': ('pid', PID)}

    chunk_num = 1000
    chunk_id = 999
    chunked_ds = [''] * chunk_num

    # with open ('../reddit_raw_data_folder/chunk999.pth') as f:
    #     for line in f:
    #         print(line)
    #         ipdb.set_trace()


    with tqdm() as pbar:
        while chunk_id < chunk_num:
            chunk_zero_fill = str(chunk_id).zfill(3)
            chunked_ds[chunk_id] = data.TabularDataset(
                path=os.path.join(f'../reddit_raw_data_folder/chunk{chunk_zero_fill}.pth'),
                format='json',
                fields=fields)

            # vectors = Vectors(name='../crawl-300d-2M.vec')
            # W.build_vocab(chunked_ds[chunk_id], vectors=vectors)
            # print(W.vocab.vectors.shape)
            # print(W.vocab.vectors)
            # print(W.vocab.stoi)

            '''
            words: dict with words as keys and word idxs as tokens
            iwords: list of words in order (where the index of each word is given by words, above)
            wordcounts: 1D Tensor indexed the same way as iwords, where each value is the frequency of that word in the corpus
            '''
            W.build_vocab(chunked_ds[chunk_id], min_freq=0)
            W.vocab.stoi = tokenizer.vocab
            words = W.vocab.stoi
            iwords = list(words.keys())
            wordcounts = torch.IntTensor([v for v in dict.values(W.vocab.freqs)])

            ipdb.set_trace()

            chunked_word_dictionary = {}
            # word_list = [x for x in chunked_ds[chunk_id].w]
            # chunked_word_dictionary["w"] = list(flatten(word_list))
            chunked_word_dictionary["w"] = [x for x in chunked_ds[chunk_id].w]
            chunked_word_dictionary["word_list"] = [tokenizer.encode(x) for x in chunked_word_dictionary["w"]]
            chunked_word_dictionary["uid"] = [x for x in chunked_ds[chunk_id].uid]
            chunked_word_dictionary["lid"] = [x for x in chunked_ds[chunk_id].lid]
            chunked_word_dictionary["pid"] = [x for x in chunked_ds[chunk_id].pid]
            chunked_word_dictionary["p2c"] = []
            chunked_word_dictionary["words"] = words
            chunked_word_dictionary["iwords"] = iwords
            chunked_word_dictionary["wordcounts"] = wordcounts
            start_char = [x[0] for x in chunked_word_dictionary["word_list"] if x]
            end_char = [x[-1] for x in chunked_word_dictionary["word_list"] if x]
            chunked_word_dictionary["cstart"] = [words[x] for x in start_char]
            chunked_word_dictionary["cend"] = [words[x] for x in end_char]

            ipdb.set_trace()

            for cp in tqdm(chunked_word_dictionary["pid"]):
                if cp not in chunked_word_dictionary["lid"]:
                    chunked_word_dictionary["p2c"].append(-1)

                else:
                    parent_id = chunked_word_dictionary['lid'].index(cp)
                    chunked_word_dictionary["p2c"].append(parent_id)

            chunked_word_dictionary["p2c"] = torch.ByteTensor(chunked_word_dictionary["p2c"])


            with open(os.path.join(f'./chunk{chunk_zero_fill}.pth'), mode='wb') as f:
                torch.save(chunked_word_dictionary, f)

            time.sleep(0.05)
            pbar.update(1)
            chunk_id += 1



def check_word_dictionary():
    word_dictionary = torch.load('./chunk999.pth')
    ipdb.set_trace()

if __name__ == '__main__':
    convert_raw_reddit_to_chunk_files()
    # check_word_dictionary()
