import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn


from model import generate_model
from opts import parse_opts
from mean import get_mean

if __name__=="__main__":
    opt = parse_opts()
    opt.resnext_cardinality = 32
    opt.model_name = 'resnext'
    opt.model_depth = 101
    opt.model = './resnext-101-kinetics.pth'
    opt.resnet_shortcut = 'B'
    opt.verbose = True
    
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 400    
    print(opt)
    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    print("here")
    if opt.verbose:
        print(model)



#python main_stage4.py --model ./resnext-101-kinetics.pth --model_name resnext --model_depth 101 --resnet_shortcut B