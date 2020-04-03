import torch
import numpy as np
import json, os
import scipy.spatial.distance
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def distance(x, y):
    return scipy.spatial.distance.cosine(x.flatten(), y.flatten())

def find_node(nodes, idx):
    for node in nodes:
        if node['node_id'] == idx:
            return node
    return None

def get_input_node(node):
    return [id_ for id_ in node.get('inputs', []) if type(id_) == int]

def find_unused_node(node_infos, in_ids, out_ids):
    all_ids = [node['node_id'] for node in node_infos]
    flags = dict((id_, False) for id_ in all_ids)
    left_ids = out_ids
    while(left_ids) :
        for id_ in left_ids:
            flags[id_] = True
        needed_ids = []
        for id_ in left_ids:
            node = find_node(node_infos, id_)
            needed_ids += get_input_node(node)
        needed_ids = [id_ for id_ in needed_ids if id_ not in in_ids]
        left_ids = needed_ids
    return [id_ for id_ in all_ids if not flags[id_]]

def update_info(nodes, node_cfg):
    node = find_node(nodes, node_cfg['node_id'])
    node.update(node_cfg)

class TorchIr(torch.nn.Module):

    supported_ops = ['nn.conv2d', 'nn.relu', 'nn.batch_norm',
            'nn.global_avg_pool2d', 'nn.batch_flatten', 'nn.dense',
            'nn.max_pool2d', 'add', 'nn.avg_pool2d', 'nn.bias_add',
            'nn.conv2d_transpose', 'sigmoid']

    def __init__(self, json_file, param_file,
            input_node_ids=[], output_node_ids=[],
            update_node_cfg=[]):
        super(TorchIr, self).__init__()
        self.model = torch.nn.ModuleList()
        self.nodes = []
        self.node_in = []
        with open(json_file, 'r') as f:
            node_infos = json.load(f)
        self.param = np.load(param_file, allow_pickle=True)["arr_0"][()]
        if update_node_cfg:
            for node_cfg in update_node_cfg:
                update_info(node_infos, node_cfg)
        self.index_node = {}
        self.input_node_ids = input_node_ids
        self.output_node_ids = output_node_ids
        self.node_infos = self._find_input_output_nodes(node_infos)
        self.bias_register = self._register_bias()
        self._parse_model(self.node_infos)

    def _register_bias(self):
        tmp = {}
        for node in self.node_infos:
            if node.get('op_type') != 'nn.bias_add':
                continue
            input_node_id = node['inputs'][0]
            tmp[input_node_id] = node['node_id']
        return tmp

    def _find_input_output_nodes(self, node_infos):
        node_infos = [node for node in node_infos if node.get('op_type') in self.supported_ops]
        all_node_ids = [node['node_id'] for node in node_infos]
        all_needed_ids = []
        for node in node_infos:
            all_needed_ids += get_input_node(node)
        if not self.input_node_ids:
            for id_ in all_needed_ids:
                if id_ not in all_node_ids:
                    self.input_node_ids.append(id_)
        assert self.input_node_ids
        if not self.output_node_ids:
            for id_ in all_node_ids:
                if id_ not in all_needed_ids:
                    self.output_node_ids.append(id_)
        unused_node_ids = find_unused_node(node_infos, self.input_node_ids, self.output_node_ids)
        node_infos = [node for node in node_infos if node['node_id'] not in unused_node_ids]
        return node_infos

    def _parse_model(self, node_infos):
        for id_ in reversed(self.input_node_ids):
            node_infos.insert(0, dict(node_id=id_, node_type='Input'))
        for node in node_infos:
            node_id = node['node_id']
            if node.get('op_type', '') in self.supported_ops:
                inputs = [find_node(node_infos, idx) for idx in node['inputs']]
                if node['op_type'] == 'nn.conv2d':
                    in_channel = node['attrs']['W_shape'][1]
                    out_channel = node['attrs']['W_shape'][0]
                    strides = node['attrs']['strides']
                    padding = node['attrs']['padding']
                    if len(padding) == 4:
                        padding = padding[:2]
                    kernel_size = node['attrs']['kernel_size']
                    bias = True if node['node_id'] in self.bias_register else False
                    cur_module = torch.nn.Conv2d(in_channel, out_channel,
                            kernel_size, stride=strides,
                            padding=padding, bias=bias)
                    weight = self.param[node['inputs'][1]]
                    cur_module.weight.data.copy_(torch.from_numpy(weight))
                    if bias:
                        bias_node = find_node(node_infos, self.bias_register[node_id])
                        bias_ = self.param[bias_node['inputs'][1]]
                        cur_module.bias.data.copy_(torch.from_numpy(bias_))
                    node_in = [inputs[0]['node_id']]
                elif node['op_type'] == 'nn.conv2d_transpose':
                    weight = self.param[node['inputs'][1]]
                    in_channel = weight.shape[1]
                    out_channel = weight.shape[0]
                    strides = node['attrs']['strides']
                    padding = node['attrs']['padding']
                    if len(padding) == 4:
                        padding = padding[:2]
                    kernel_size = node['attrs']['kernel_size']
                    cur_module = torch.nn.ConvTranspose2d(in_channel, out_channel,
                            kernel_size, stride=strides,
                            padding=padding, bias=False)
                    cur_module.weight.data.copy_(torch.from_numpy(weight))
                    node_in = [inputs[0]['node_id']]
                elif node['op_type'] == 'nn.batch_norm':
                    out_channel = in_channel = node['attrs']['_shape'][0]
                    eps = node['attrs']['epsilon']
                    cur_module = torch.nn.BatchNorm2d(in_channel,eps=eps, momentum=0.9)
                    gamma = self.param[node['inputs'][1]]
                    beta = self.param[node['inputs'][2]]
                    running_mean = self.param[node['inputs'][3]]
                    running_var = self.param[node['inputs'][4]]
                    cur_module.weight.data.copy_(torch.from_numpy(gamma))
                    cur_module.bias.data.copy_(torch.from_numpy(beta))
                    cur_module.running_mean.data.copy_(torch.from_numpy(running_mean))
                    cur_module.running_var.data.copy_(torch.from_numpy(running_var))
                    node_in = [inputs[0]['node_id']]
                elif node['op_type'] == 'nn.relu':
                    # out_channel = in_channel = self.node_out[inputs[0]['node_id']]
                    cur_module = torch.nn.ReLU(inplace=True)
                    node_in = [inputs[0]['node_id']]
                elif node['op_type'] == 'sigmoid':
                    import pdb; pdb.set_trace()
                    cur_module = torch.nn.Sigmoid()
                    node_in = [inputs[0]['node_id']]
                elif node['op_type'] == 'nn.max_pool2d':
                    # out_channel = in_channel = self.node_out[inputs[0]['node_id']]
                    pool_size = node['attrs'].get('pool_size', (1,1))
                    strides = node['attrs'].get('strides', (1,1))
                    padding = node['attrs'].get('padding', (0,0))
                    ceil_mode = node['attrs'].get('ceil_mode', 0)
                    ceil_mode = True if ceil_mode else False
                    cur_module = torch.nn.MaxPool2d(
                            pool_size, stride=strides,
                            padding=padding,
                            ceil_mode=ceil_mode)
                    node_in = [inputs[0]['node_id']]
                elif node['op_type'] == 'nn.avg_pool2d':
                    pool_size = node['attrs'].get('pool_size', (1,1))
                    strides = node['attrs'].get('strides', (1,1))
                    padding = node['attrs'].get('padding', (0,0))
                    cur_module = torch.nn.AvgPool2d(
                            pool_size, stride=strides,
                            padding=padding,
                            ceil_mode=ceil_mode)
                    node_in = [inputs[0]['node_id']]
                elif node['op_type'] == 'add':
                    cur_module = torch.nn.Sequential()
                    node_in = [in_['node_id'] for in_ in inputs]
                elif node['op_type'] == 'nn.global_avg_pool2d':
                    cur_module = torch.nn.AdaptiveAvgPool2d(1)
                    node_in = [inputs[0]['node_id']]
                elif node['op_type'] == 'nn.batch_flatten':
                    cur_module = torch.nn.Flatten()
                    node_in = [inputs[0]['node_id']]
                elif node['op_type'] == "nn.dense":
                    out_channel = node['attrs']['_shape'][0]
                    in_channel = node['attrs']['_shape'][1]
                    bias = True if node['node_id'] in self.bias_register else False
                    cur_module = torch.nn.Linear(in_channel, out_channel, bias=bias)
                    weight = self.param[node['inputs'][1]]
                    cur_module.weight.data.copy_(torch.from_numpy(weight))
                    if bias:
                        bias_node = find_node(node_infos, self.bias_register[node_id])
                        bias_ = self.param[bias_node['inputs'][1]]
                        cur_module.bias.data.copy_(torch.from_numpy(bias_))
                    node_in = [inputs[0]['node_id']]
                elif node['op_type'] == 'nn.bias_add':
                    cur_module = torch.nn.Sequential()
                    node_in = [inputs[0]['node_id']]

                self.index_node[node['node_id']] = cur_module
                self.model.append(cur_module)
                self.nodes.append(node)
                self.node_in.append(node_in)
            else:
                print("** converting IR to network **")

    def forward(self, *args):
        assert len(args) == len(self.input_node_ids)
        outs = dict((id_, x) for (id_, x) in zip(self.input_node_ids, args))
        for node, node_in, module in zip(self.nodes, self.node_in, self.model):
            in_ = [outs[idx] for idx in node_in]
            if node['op_type'] == 'add':
                x = in_[0] + in_[1]
            else:
                x = module(*in_)
            outs[node['node_id']] = x
        if len(self.output_node_ids) == 1:
            return outs[self.output_node_ids[0]]
        else:
            return [outs[id_] for id_ in self.output_node_ids]

    def save_torch_weight(self, wf):
        torch.save(self.state_dict(), wf)

    def save_json(self, fp, *args):
        from copy import deepcopy
        default_input_node = {
          "node_id": 0,
          "node_type": "Var",
          "node_name": "data",
          "op_type": "Const",
          "shape": [ 1, 3, 224, 224]
        }
        node_infos = [node for node in self.node_infos if node['node_type'] != 'Input']
        for idx, id_ in enumerate(reversed(self.input_node_ids)):
            default_input_node['node_id'] = id_
            default_input_node['shape'] = list(args[idx].shape)
            node_infos.insert(0, deepcopy(default_input_node))

        with open(fp, 'w') as f:
            json.dump(node_infos, f, indent=2)

