"""
Generate a resnext-101 model using parsed inputs from opts.py

"""

import torch
from models import resnext


def generate_model(opt):
    
    assert opt.mode in ['score', 'feature']
    if opt.mode == 'score':
        last_fc = True
    elif opt.mode == 'feature':
        last_fc = False


    assert opt.model_name in ['resnext']
    assert opt.model_depth in [101]
    
    model = resnext.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                      sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                      last_fc=last_fc)


    if not opt.no_cuda:
        print("CUDA is required for this project")

    return model
