# pytorch tool
## reconstruct_torch.json
    model = TorchIr('resnet50v1d.json', 'resnet50v1d.npz')
### 导入稀疏化网络和参数（部分或全部） 
    找到对应node的id (in "resnet50v1d.png")，以input_node_id和output_node_id的方式指定片段的开始和结束的地方
    model = TorchIr('resnet50v1d.json', 'resnet50v1d.npz', input_node_id=[291], output_node_id=[525])
### 修改导入网络的节点属性
    #load Moffett IR
    net = TorchIr("examples/cifar10/resnet50v1d.json",
                  "examples/cifar10/resnet50v1d.npz", output_node_ids=[527])
    
    # change the strides of first conv2d and maxpooling
    net.index_node[287].stride = [1, 1]
    net.index_node[299].stride = [1, 1]
    net.index_node[299].kernel_size = [3, 3]
    net.index_node[299].padding = [1, 1]

### 在外部使用TorchIr
```
    class MyExample(nn.Module):
        def __init__(self):
            self.conv1 = conv2d(...)
            self.somenet = TorchIr(...)
            self.fc = lieanr(...)
            self.init_params()

        def forward(self, x):
            x = self.fc(self.somenet(self.conv1(x)))
            ...

        def load_pretrained_weights(self, wf):
            self.somenet.load_state_dict(torch.load(wf))
```
