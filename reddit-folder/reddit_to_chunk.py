import random
import numpy as np
import torch
import torchtext.utils
# from torch.utils.data import Dataset, DataLoader
from torchtext.data import Dataset, TabularDataset
from torchtext import data
import json
import os
import subprocess
import linecache
import csv
import ipdb

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

reddit_file = './raw_data'

# def make_chunk_from_reddit():
#     with open('./raw_data', 'r') as infile:
#         o = json.load(infile)
#         chunk_size = 1000
#         for i in range(0, len(o), chunk_size):
#             ipdb.set_trace()
#             with open('chunk' + i + '.pth', 'w') as outfile:
#                 json.dump(o[i:i+chunk_size], outfile)



class RedditDataset(Dataset):
    def __init__(self, filename):
        self._filename = filename
        self._total_data = 0
        self._total_data = int(subprocess.check_output("wc -l " + filename, shell=True).split()[0])

    def __getitem__(self, idx):
        with open(self._filename) as f:
            return json.loads(f.readline())['body']
            # 毎回lineの最初がloadされるので修正必要
        # line = linecache.getline(self._filename, idx + 1)
        # return line
        # csv_line = csv.reader(line)
        # return next(csv_line)

    def __len__(self):
        return self._total_data


def get_reddit_dic():
    BODY = data.Field(sequential=True, batch_first=True)
    fields = {'body': ('body', BODY)}
    ds = data.TabularDataset(
        path='./chunk000.pth', format='json',
        fields=fields)

    # test dataloader
    print(f'データ数{len(ds)}')
    print(f'1つ目のデータ{vars(ds[0])}')
    ipdb.set_trace()






if __name__ == '__main__':
    # test_data = RedditDataset('./chunk000.pth')
    get_reddit_dic()