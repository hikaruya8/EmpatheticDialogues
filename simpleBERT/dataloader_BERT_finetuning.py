import random
import torchtext
from transformers import BertTokenizer

def get_utterance_and_context_loader(max_length=256, batch_size=32):
    max_length = max_length
    batch_size = batch_size

    # torchtext用にBertTokenizerの準備
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    eos_index = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    unk_index = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    mask_index = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    cls_index = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)

    ID = torchtext.data.Field(sequential=False, use_vocab=False)
    UTTERANCE = torchtext.data.Field(sequential=True, tokenize=tokenizer.encode, use_vocab=False, include_lengths=True, batch_first=True, fix_length=max_length, init_token=cls_index, eos_token=eos_index, pad_token=pad_index, unk_token=unk_index)
    SPEAKER =  torchtext.data.Field(sequential=True, tokenize=tokenizer.encode, use_vocab=False, include_lengths=True, batch_first=True, fix_length=max_length,
        init_token=cls_index, eos_token=eos_index, pad_token=pad_index, unk_token=unk_index)
    CONTEXT_ALL = torchtext.data.Field(sequential=True, tokenize=tokenizer.encode, use_vocab=False, include_lengths=True, batch_first=True, fix_length=max_length, init_token=cls_index, eos_token=eos_index, pad_token=pad_index, unk_token=unk_index)
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False, preprocessing=lambda l: 0 if l=='TRUE' else 1, is_target=True)

    '''if use CONTEXT respectively'''
   #  CONTEXT1 =  torchtext.data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token=cls_index, eos_token=eos_index, pad_token=pad_index, unk_token=unk_index)
   #  CONTEXT2 =  torchtext.data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token=cls_index, eos_token=eos_index, pad_token=pad_index, unk_token=unk_index)
   #  CONTEXT3 =  torchtext.data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token=cls_index, eos_token=eos_index, pad_token=pad_index, unk_token=unk_index)
   #  CONTEXT4 =  torchtext.data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token=cls_index, eos_token=eos_index, pad_token=pad_index, unk_token=unk_index)
   #  CONTEXT5 =  torchtext.data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token=cls_index, eos_token=eos_index, pad_token=pad_index, unk_token=unk_index)
   #  CONTEXT6 =  torchtext.data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token=cls_index, eos_token=eos_index, pad_token=pad_index, unk_token=unk_index)
   #  CONTEXT7 =  torchtext.data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token=cls_index, eos_token=eos_index, pad_token=pad_index, unk_token=unk_index)
   #  CONTEXT8 =  torchtext.data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token=cls_index, eos_token=eos_index, pad_token=pad_index, unk_token=unk_index)
   #  CONTEXT9 =  torchtext.data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token=cls_index, eos_token=eos_index, pad_token=pad_index, unk_token=unk_index)
   #  CONTEXT10 =  torchtext.data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token=cls_index, eos_token=eos_index, pad_token=pad_index, unk_token=unk_index)
   # CONTEXT11 =  torchtext.data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token=cls_index, eos_token=eos_index, pad_token=pad_index, unk_token=unk_index)


    ds = torchtext.data.TabularDataset(
        path='../MUStARD/sarcasm_data.csv', format='csv',
        fields=[("id", ID),
                ("utterance", UTTERANCE),
                ("speaker", SPEAKER),
                ("context_all", CONTEXT_ALL),
                ("label", LABEL)],
                skip_header=True)

    # test dataloader
    # print(f'データ数{len(ds)}')
    # print(f'1つ目のデータ{vars(ds[1])}')

    # dsをtrain, val, testに分ける. ランダムに8:1:1 で分ける
    train_ds, val_ds, test_ds = ds.split(split_ratio=[0.8, 0.1, 0.1], random_state=random.seed(1234))

    # test split
    # print(f'train dataの数:{len(train_ds)}, validataion dataの数:{len(val_ds)}, test dataの数: {len(test_ds)}')
    # print(f'1つ目のデータ{vars(train_ds[1])}')

    # make dataloader
    train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)
    val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)
    test_dl = torchtext.data.Iterator(test_ds, batch_size=batch_size, train=False, sort=False)

    # test   train_dataで確認
    # batch = next(iter(train_dl))
    # print(batch.utterance)
    # print(batch.label)


if __name__ == '__main__':
    get_utterance_and_context_loader()




