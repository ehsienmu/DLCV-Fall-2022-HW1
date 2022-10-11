mycnn_cfg = {
    'model_type': 'myCNN',
    'training_data_path' : './hw1_data/hw1_data/p1_data/train_50/',
    'val_data_path' : './hw1_data/hw1_data/p1_data/val_50/',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    'do_semi': False,
    # training hyperparameters
    'batch_size': 512,
    'lr': 0.001,
    # 'milestones': [20, 40, 60, 80, 100],
    'milestones': [30, 60, 90, 120],
    'num_out': 50,
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
        'lr': 0.001,              # learning rate of Adam
        'weight_decay':1e-3,
        # 'betas':(0.4, 0.999)      # Adam才有此參數
    },
}

pretrained_resnet50_cfg = {
    'model_type': 'pretrained_resnet50',
    'training_data_path' : './hw1_data/hw1_data/p1_data/train_50/',
    'val_data_path' : './hw1_data/hw1_data/p1_data/val_50/',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    'do_semi': False,
    # training hyperparameters
    'batch_size': 16,
    # 'lr': 0.001,
    'milestones': [20, 40, 60, 80, 100],
    # 'milestones': [30, 60, 90, 120],
    'num_out': 50,
    'num_epoch': 30,
    'early_stop': 10,
    
    'optimizer': 'SGD',             # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {               # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0005,              # learning rate of SGD
        'momentum': 0.8,              # momentum for SGD
        'weight_decay':1e-5,
    },
    # 'optimizer': 'Adam',             # optimization algorithm (optimizer in torch.optim)
    # 'optim_hparas': {               # hyper-parameters for the optimizer (depends on which optimizer you are using)
    #     'lr': 0.001,              # learning rate of Adam
    #     'weight_decay':1e-3,
    #     'betas':(0.4, 0.999)      # Adam才有此參數
    # },
}
myresnet_cfg = {
    'model_type': 'myresnet',
    'training_data_path' : './hw1_data/hw1_data/p1_data/train_50/',
    'val_data_path' : './hw1_data/hw1_data/p1_data/val_50/',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    'do_semi': False,
    # training hyperparameters
    'batch_size': 16,
    # 'lr': 0.001,
    'milestones': [20, 40, 60, 80, 100],
    # 'milestones': [30, 60, 90, 120],
    'num_out': 50,
    'num_epoch': 30,
    'early_stop': 10,
    
    'optimizer': 'SGD',             # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {               # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0005,              # learning rate of SGD
        'momentum': 0.8,              # momentum for SGD
        'weight_decay':1e-5,
    },
    # 'optimizer': 'Adam',             # optimization algorithm (optimizer in torch.optim)
    # 'optim_hparas': {               # hyper-parameters for the optimizer (depends on which optimizer you are using)
    #     'lr': 0.001,              # learning rate of Adam
    #     'weight_decay':1e-3,
    #     'betas':(0.4, 0.999)      # Adam才有此參數
    # },
}