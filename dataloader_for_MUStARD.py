import json
import string
import re
import os
import urllib
import zipfile
import random
import pdb
import torchtext
from torchtext.vocab import Vectors

def download_word_vectors():
    # フォルダ「data」が存在しない場合は作成する
    data_dir = "./data/"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # fastTextの公式の英語学習済みモデル（650MB）をダウンロード。解凍
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
    save_path = "./data/wiki-news-300d-1M.vec.zip"
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)
    zip = zipfile.ZipFile("./data/wiki-news-300d-1M.vec.zip")
    zip.extractall("./data/")  # ZIPを解凍
    zip.close()  # ZIPファイルをクローズ

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


def get_utterance_and_context_loader(max_length=256, batch_size=64):
    max_length = max_length
    batch_size = batch_size

    ID = torchtext.data.Field(sequential=False, use_vocab=False)
    UTTERANCE = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
    SPEAKER = torchtext.data.Field(sequential=False, use_vocab=True)
    CONTEXT_ALL = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False, preprocessing=lambda l: 0 if l=='TRUE' else 1, is_target=True)
   #  CONTEXT1 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT2 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT3 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT4 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT5 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT6 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT7 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT8 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT9 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT10 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   # CONTEXT11 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")


    ds = torchtext.data.TabularDataset(
        path='./MUStARD/sarcasm_data.csv', format='csv',
        fields=[("id", None),
                ("utterance", UTTERANCE),
                ("speaker", None),
                ("context_all", CONTEXT_ALL),
                ("label", LABEL)],
                skip_header=True)

    # test dataloader
    print(f'データ数{len(ds)}')
    print(f'1つ目のデータ{vars(ds[1])}')

    # dsをtrain, val, testに分ける. ランダムに8:1:1 で分ける
    train_ds, val_ds, test_ds = ds.split(split_ratio=[0.8, 0.1, 0.1], random_state=random.seed(1234))

    # test split
    print(f'train dataの数:{len(train_ds)}, validataion dataの数:{len(val_ds)}, test dataの数: {len(test_ds)}')
    print(f'1つ目のデータ{vars(train_ds[1])}')


    english_fasttext_vectors = Vectors(name='data/wiki-news-300d-1M.vec')

    # ベクトル化したバージョンのボキャブラリーを作成. (UTTERANCEとCONTEXTの2つのフィールドで同一のvocabを作成したため少し変則的)
    UTTERANCE.build_vocab(ds.utterance, ds.context_all, vectors=english_fasttext_vectors, min_freq=1)
    CONTEXT_ALL.vocab = UTTERANCE.vocab
    # 普通のbuild_vocab
    # UTTERANCE.build_vocab(ds, vectors=english_fasttext_vectors, min_freq=1)
    # CONTEXT_ALL.build_vocab(ds, vectors=english_fasttext_vectors, min_freq=1)
    # LABEL.build_vocab()

    # ボキャブラリーのベクトルを確認
    print(UTTERANCE.vocab.vectors.shape)
    print(UTTERANCE.vocab.vectors)

    # ボキャブラリーの単語の順番を確認
    # print(CONTEXT_ALL.vocab.stoi)

    # make dataloader
    train_dl = torchtext.data.Iterator(train_ds, batch_size=24, train=True)
    val_dl = torchtext.data.Iterator(val_ds, batch_size=24, train=False, sort=False)
    test_dl = torchtext.data.Iterator(test_ds, batch_size=24, train=False, sort=False)

    # pdb.set_trace()
    # test   train_dataで確認
    batch = next(iter(train_dl))
    print(batch.utterance)
    print(batch.label)


if __name__ == '__main__':
    # download_word_vectors()
    get_utterance_and_context_loader()




