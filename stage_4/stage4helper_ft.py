import torch
from torch import nn

from models import resnext


def generate_model(opt):
    assert opt.model in ['resnext']

    assert opt.model_depth in [101]
    
    from models.resnext import get_fine_tuning_parameters

    if opt.model_depth == 101:
        model = resnext.resnet101(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)

    if opt.no_cuda:
        print ("CUDA required")
        
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)    
    
    """ modify model for fine-tuning """
    # set the pretrain path in main_stage4_ft.py
    print('loading pretrained model {}'.format(opt.pretrain_path))
    pretrain = torch.load(opt.pretrain_path)
    
    assert opt.arch == pretrain['arch']

    model.load_state_dict(pretrain['state_dict'])

    #replace the last fc layer of the model
    model.module.fc = nn.Linear(model.module.fc.in_features,
                                opt.n_finetune_classes)
    model.module.fc = model.module.fc.cuda()

    parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
    return model, parameters            
        

    