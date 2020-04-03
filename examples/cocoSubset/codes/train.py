import os
import sys
import time
import torch
import numpy

# setup data and environment
gpu_name = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_name
gpus = list(map(int, gpu_name.split(',')))
device = torch.device('cuda' if gpus[0] >= 0 else 'cpu')
chunk_sizes = [13]
batch_size = sum(chunk_sizes)
num_epochs = 270
num_worker = 10 
learn_rate = 5e-5 / 128 * batch_size
decay_step = [90, 120]
root_path = '/home'
train_image_path = root_path + '/data/coco/train2017_subset'
train_annotate_path = root_path + '/data/coco/annotations/instances_train2017_subset.json'
validate_image_path = root_path + '/data/coco/val2017_subset'
validate_annotate_path = root_path + '/data/coco/annotations/instances_val2017_subset.json'
load_path = 'models/resnet50v1b'
save_path = 'models/exp01_resnet50v1b_reconstruct_cocoSubset_finetune'
if not os.path.exists(save_path):
    os.makedirs(save_path)
val_intervals = 5
metric = 'loss'
torch.cuda.manual_seed_all(317)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# import sparse resent50v1d through Moffett IR 
from utils import load_model, save_model
from resnet_IR import get_pose_net
jsn = '../centernet_resnetv1b_map31_sparsity80/resnet50_centernet.json'
params = '../centernet_resnetv1b_map31_sparsity80/resnet50_centernet.npz'
model = get_pose_net(jsn, params, input_node_ids = [0], output_node_ids = [570], heads={'hm': 3, 'reg': 2, 'wh': 2}, head_conv=64)

# import Moffett pruning optimizer
import pytorch_pruning
optimizer = pytorch_pruning.AdamSparse(model.parameters(), lr=learn_rate, restore_sparsity=True, fix_sparsity=True, param_name=model.named_parameters())

# setup trainer
start_epoch = 0
from trainer import Trainer
trainer = Trainer(model, optimizer, gpus, chunk_sizes, device)

# setup dataloader
import dataset_train_cocoSubset as dataset_train
import dataset_validate_cocoSubset as dataset_validate
val_loader = torch.utils.data.DataLoader(
    dataset=dataset_validate.Dataset(validate_image_path, validate_annotate_path),
    num_workers=num_worker,
    pin_memory=True)
train_loader = torch.utils.data.DataLoader(
    dataset=dataset_train.Dataset(train_image_path, train_annotate_path),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_worker,
    pin_memory=True,
    drop_last=True)

#training
best = 1e10
for epoch in range(start_epoch + 1, num_epochs + 1):
    log_dict_train, _ = trainer.run_epoch('train', epoch, train_loader)
    for k, v in log_dict_train.items():
        SummaryWriter.add_scalar('train_{}'.format(k), v, epoch)
    if val_intervals > 0 and epoch % val_intervals == 0:
        save_model(os.path.join(save_path, 'model_{}.pth'.format('last')), epoch, model, optimizer)
        with torch.no_grad():
            log_dict_val, preds = trainer.run_epoch('val', epoch, val_loader)
        for k, v in log_dict_val.items():
            SummaryWriter.add_scalar('val_{}'.format(k), v, epoch)
        if log_dict_val[metric] < best:
            best = log_dict_val[metric]
            save_model(os.path.join(save_path, 'model_best.pth'), epoch, model)
    else:
        save_model(os.path.join(save_path, 'model_last.pth'), epoch, model, optimizer)
    if epoch in decay_step:
        save_model(os.path.join(save_path, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
        lr = learn_rate * (0.1 ** (decay_step.index(epoch) + 1))
        for param_group in optimizer.param_groups:
            param_group['lr' ] = lr
