import os
import torch
from moffett_ir import TorchIr


class PoseResNet(torch.nn.Module):
    def __init__(self,jsn, params, input_node_ids, output_node_ids, heads={'hm': 3, 'reg': 2, 'wh': 2}, head_conv=64, **kwargs):
        self.heads = heads
        super(PoseResNet_subset, self).__init__()

        # load Moffett ir
        self.resnetv1d_centernet = TorchIr(jsn, params, input_node_ids, output_node_ids)

        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = torch.nn.Sequential(
                torch.nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0))
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.resnetv1d_centernet(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


def get_pose_net(jsn, params, input_node_ids, output_node_ids, heads={'hm': 3, 'reg': 2, 'wh': 2}, head_conv=64):
    model = PoseResNet(jsn, params, input_node_ids, output_node_ids,heads={'hm': 3, 'reg': 2, 'wh': 2}, head_conv=64)
    return model
