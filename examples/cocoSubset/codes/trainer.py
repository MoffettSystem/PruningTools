import time
import torch
import numpy
from losses import FocalLoss, RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from utils import _sigmoid
from progress.bar import Bar
from data_parallel import DataParallel

class CtdetLoss(torch.nn.Module):
    def __init__(self):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = RegL1Loss()

    def forward(self, outputs, batch):
        hm_loss, wh_loss, off_loss = 0, 0, 0

        output = outputs[0]
        output['hm'] = _sigmoid(output['hm'])

        hm_loss += self.crit(output['hm'], batch['hm'])
        wh_loss += self.crit_reg(output['wh'], batch['reg_mask'], batch['ind'], batch['wh'])
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'], batch['ind'], batch['reg'])

        loss = 1 * hm_loss + 0.1 * wh_loss + 1 * off_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
###########################################
        self.model = model
        self.loss = loss
        # self.model = model.half()
        # self.loss = loss.half()
#############################################
    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats

class Trainer(object):
    def __init__(self, model, optimizer, gpus, chunk_sizes, device):
        self.gpus = gpus
        self.device = device
        self.optimizer = optimizer
        self.loss_stats = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        self.loss = CtdetLoss()
        self.model_with_loss = ModleWithLoss(model, self.loss)

        if len(self.gpus) > 1:
            self.model_with_loss = DataParallel(self.model_with_loss, device_ids=self.gpus, output_device=0,
            chunk_sizes=chunk_sizes
            ).to(self.device)
        else:
            self.model_with_loss = self.model_with_loss.to(self.device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=self.device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        bar = Bar('{}'.format('ctdet'), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=self.device, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(epoch, iter_id, num_iters, phase=phase, total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) |Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            bar.next()
            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results