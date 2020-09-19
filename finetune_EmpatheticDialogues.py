import torch
import torchtext
from empchat.transformer_local import TransformerAdapter
from empchat.util import get_opt
import ipdb


pretrained_params = torch.load('./normal_transformer_pretrained.mdl')
pretrained_state_dict = pretrained_params['state_dict']
pretrained_opt = pretrained_params['opt']
pretrained_word_dict = pretrained_params['word_dict']['words']


finetuned_params = torch.load('./normal_transformer_finetuned.mdl')
finetuned_state_dict = finetuned_params['state_dict']
finetuned_opt = finetuned_params['opt']
finetuned_word_dict = finetuned_params['word_dict']['words']
finetuned_optim_dict = finetuned_params['optim_dict']


pretrained_model = TransformerAdapter(pretrained_opt, pretrained_word_dict)
finetuned_model = TransformerAdapter(finetuned_opt, finetuned_word_dict)



def create_finetune_model():
    # First, stop the
    return None



if __name__ == '__main__':
    create_finetune_model()
    ipdb.set_trace()