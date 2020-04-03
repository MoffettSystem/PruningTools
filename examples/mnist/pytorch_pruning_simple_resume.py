import os
import sys
import numpy
import torch
import torchvision
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from optimizers import pytorch_pruning as pruning

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, 2)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, 2)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, 3, 2)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.linear1 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        output = torch.nn.functional.softmax(x, dim=1)
        return output

if __name__ == '__main__':
    # setup gpu device(s) 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #setup training parameters
    batch_size = 1000
    pretrain_epoch = 0 
    epoch = 50
    pruning_epoch = 40 
    assert pretrain_epoch <= epoch, 'reset pruning epoch to be smaller than total epoch!!'
    lr = 0.1 / 256 * batch_size
    print('*** initial training parameters ***')
    print('batch size: {};\ntraining epochs without pruning: {};\ntotal training epochs: {};\ninitial learning rate: {:.3f} with cosine decay;'.format(batch_size, pretrain_epoch, epoch, lr))

    # data loader 
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST('~/.pytorch/datasets', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST('~/.pytorch/datasets', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    step = train_data.data.shape[0] // batch_size

    # define network 
    model = Net().to(device)
    model.load_state_dict(torch.load('pytorch_mnist.pth'))

    # pruning setup 
    target_sparsity = 0.95
    pretrain_step = step * pretrain_epoch
    sparse_step = pruning_epoch * step
    frequency = step / 5
    keywords_no_sparse = ['bias', 'bn', 'conv1.weight'] 
    special_sparsity_dict = {'conv1_weight': 0.7}
    #optimizer = pruning.SGDSparse(model.parameters(), lr, weight_decay=1e-5, restore_sparsity=True, fix_sparsity=True, param_name=model.named_parameters())
    optimizer = pruning.SGDSparse(model.parameters(), lr, weight_decay=1e-5, target_sparsity=target_sparsity, pretrain_step=pretrain_step, sparse_step=sparse_step, frequency=frequency, keywords_no_sparse=keywords_no_sparse, special_sparsity_dict=special_sparsity_dict, param_name=model.named_parameters(), restore_sparsity=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, step * epoch)
    print('*** pruning setup ***')
    print('pruning optimizer is based on {};\ntarget pruning rate: {};\npruning_epoch: {};\npruning frequency: {};\nops without pruning: {};'.format(
        'SGD', 
        target_sparsity, 
        pruning_epoch, 
        frequency, 
        keywords_no_sparse)
        )
    print('*** special sparsity layers ***')
    for k, v in special_sparsity_dict.items():
        print(k, ": ", v)

    # training 
    print('start training')
    for idx in range(epoch):
        model.train()
        for (data, target) in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)

        # print(torch.cuda.memory_allocated(torch.device('cuda')))
        temp_dict = {}
        for n, p in model.named_parameters():
            if ('weight' not in n) or ('bn' in n) or ('conv1.weight' in n):
                continue
            temp = p.data.cpu().numpy()
            rate = numpy.flatnonzero(temp).size / temp.size
            temp_dict[n] = rate
        temp_rate = numpy.mean([one[1] for one in temp_dict.items()])
        print('epoch {}|accuracy: {:.4f}|pruning_rate={:.4f}'.format(idx, test_acc, 1.0-temp_rate))
    torch.save(model.state_dict(), 'pytorch_mnist_resume.pth')

    print('*** summary ***')
    print('pruning rate = (MACs of sparse processing)/(MACs of dense processing)')
    print('MACs = number of multiply-accumulate')
    print('final accuracy: {:.4f}, final pruning rate: {:.4f}'.format(test_acc, 1-temp_rate))
    print("*** model saved as pytorch_mnist_resume.pth ***")
