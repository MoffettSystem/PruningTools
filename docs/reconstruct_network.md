# pytorch tool
## import network and sparse weights 
    model = TorchIr('resnet50v1d.json', 'resnet50v1d.npz')

### Example for using TorchIr
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
