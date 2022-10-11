myfcn32_cfg = {
    'model_type': 'fcn32',
    'training_data_path' : './hw1_data/hw1_data/p2_data/train/',
    'val_data_path' : './hw1_data/hw1_data/p2_data/validation/',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 1,
    'do_semi': False,
    # training hyperparameters
    'batch_size': 8,
    # 'lr': 0.0005,
    # 'milestones': [20, 40, 60, 80, 100],
    'milestones': [30, 60, 90, 120],
    'num_out': 7,
    'num_epoch': 300,
    'early_stop': 30,
    
    # 'optimizer': 'SGD',             # optimization algorithm (optimizer in torch.optim)
    # 'optim_hparas': {               # hyper-parameters for the optimizer (depends on which optimizer you are using)
    #     'lr': 0.001,              # learning rate of SGD
    #     'momentum': 0.8,              # momentum for SGD
    #     'weight_decay':1e-4,
    # },
    'optimizer': 'Adam',             # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {               # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0005,              # learning rate of Adam
        # 'weight_decay':1e-3,
        # 'betas':(0.4, 0.999)      # Adam才有此參數
    },
}


myfcn8_cfg = {
    'model_type': 'fcn8',
    'training_data_path' : './hw1_data/hw1_data/p2_data/train/',
    'val_data_path' : './hw1_data/hw1_data/p2_data/validation/',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 1,
    'do_semi': False,
    # training hyperparameters
    'batch_size': 8,
    # 'lr': 0.0005,
    # 'milestones': [20, 40, 60, 80, 100],
    'milestones': [30, 60, 90, 120],
    'num_out': 7,
    'num_epoch': 300,
    'early_stop': 30,
    
    # 'optimizer': 'SGD',             # optimization algorithm (optimizer in torch.optim)
    # 'optim_hparas': {               # hyper-parameters for the optimizer (depends on which optimizer you are using)
    #     'lr': 0.001,              # learning rate of SGD
    #     'momentum': 0.8,              # momentum for SGD
    #     'weight_decay':1e-4,
    # },
    'optimizer': 'Adam',             # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {               # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0005,              # learning rate of Adam
        'weight_decay':1e-3,
        'betas':(0.4, 0.999)      # Adam才有此參數
    },
}
myfcn32_cfg = {
    'model_type': 'fcn32',
    'training_data_path' : './hw1_data/hw1_data/p2_data/train/',
    'val_data_path' : './hw1_data/hw1_data/p2_data/validation/',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 1,
    'do_semi': False,
    # training hyperparameters
    'batch_size': 8,
    # 'lr': 0.0005,
    # 'milestones': [20, 40, 60, 80, 100],
    'milestones': [30, 60, 90, 120],
    'num_out': 7,
    'num_epoch': 300,
    'early_stop': 30,
    
    # 'optimizer': 'SGD',             # optimization algorithm (optimizer in torch.optim)
    # 'optim_hparas': {               # hyper-parameters for the optimizer (depends on which optimizer you are using)
    #     'lr': 0.001,              # learning rate of SGD
    #     'momentum': 0.8,              # momentum for SGD
    #     'weight_decay':1e-4,
    # },
    'optimizer': 'Adam',             # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {               # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0005,              # learning rate of Adam
        # 'weight_decay':1e-3,
        # 'betas':(0.4, 0.999)      # Adam才有此參數
    },
}


resnet50_fcn8_cfg = {
    'model_type': 'resnet50_fcn8',
    'training_data_path' : './hw1_data/hw1_data/p2_data/train/',
    'val_data_path' : './hw1_data/hw1_data/p2_data/validation/',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 1,
    'do_semi': False,
    # training hyperparameters
    'batch_size': 8,
    # 'lr': 0.0005,
    # 'milestones': [20, 40, 60, 80, 100],
    'milestones': [30, 60, 90, 120],
    'num_out': 7,
    'num_epoch': 300,
    'early_stop': 30,
    
    # 'optimizer': 'SGD',             # optimization algorithm (optimizer in torch.optim)
    # 'optim_hparas': {               # hyper-parameters for the optimizer (depends on which optimizer you are using)
    #     'lr': 0.0005,              # learning rate of SGD
    #     'momentum': 0.8,              # momentum for SGD
    #     'weight_decay':1e-4,
    # },
    'optimizer': 'Adam',             # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {               # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0005,              # learning rate of Adam
        'weight_decay':1e-5,
        'betas':(0.4, 0.999)      # Adam才有此參數
    },
}
