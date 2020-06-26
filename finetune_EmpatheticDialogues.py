import torch
import torchtext
from empchat.transformer_local import TransformerAdapter
from empchat.util import get_opt
import ipdb


pretrained_state_dict = torch.load('./normal_transformer_finetuned.mdl')
opt = pretrained_state_dict['opt']
word_dict = pretrained_state_dict['word_dict']

ipdb.set_trace()

# opt = get_opt()
# word_dict = torch.load('./reddit_data/word_dictionary')
# model = TransformerAdapter(opt, word_dict)
# model = TransformerAdapter(pretrained_state_dict['opt'], pretrained_state_dict['word_dict']['words'])
# model = torch.load('./raw_model/bert.pth')

model = torch.load('raw_model/finetune_model.pth')

ipdb.set_trace()

state_dict = model.load_state_dict(torch.load('./normal_transformer_finetuned.mdl'))
my_state_dict = model.load_state_dict(torch.load('./transformer_finetuning_train_save_folder/model.mdl'))
# model_txt = torch.load('../../EmpatheticDialogues/transformer_finetuning_train_save_folder/model.txt')

ipdb.set_trace()