import os
import torch
import torchvision
from moffett_ir import TorchIr
from optimizers import pytorch_pruning
from cifar_utils import progress_bar

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 0.1
epoch = 200
batch_size = 128
num_workers = 16
resume = ''
save_dir = 'checkpoint/resnet_mx2pt_fix_224.pth'
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.RandomCrop(224, padding=28),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root='~/.pytorch/datasets', train=True, transform=transform_train, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testset = torchvision.datasets.CIFAR10(root='~/.pytorch/datasets', train=False, transform=transform_test, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Model
#load Moffett IR
net = TorchIr("examples/cifar10/resnet50v1d.json",
        "examples/cifar10/resnet50v1d.npz", output_node_ids=[527])
# add fully connection layer with 10 classes for cifar10
net = torch.nn.Sequential(net, torch.nn.Linear(2048, 10))
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    torch.backends.cudnn.benchmark = True

best_acc = 0
if resume:
    checkpoint = torch.load(resume)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']

criterion = torch.nn.CrossEntropyLoss()
optimizer = pytorch_pruning.SGDSparse(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, restore_sparsity=True, fix_sparsity=True, param_name=net.named_parameters())
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (trainset.data.shape[0] // batch_size + 1) * epoch)

for ep in range(0, epoch):
    print('\nEpoch: %d' % ep)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': ep}
    torch.save(state, save_dir)
