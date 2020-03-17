import os
import sys
import numpy
import mxnet
import gluoncv
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from optimizers import mxnet_pruning as pruning

class Net(mxnet.gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.conv1 = mxnet.gluon.nn.Conv2D(64, 3, 2)
        self.bn1 = mxnet.gluon.nn.BatchNorm()
        self.conv2 = mxnet.gluon.nn.Conv2D(128, 3, 2)
        self.bn2 = mxnet.gluon.nn.BatchNorm()
        self.conv3 = mxnet.gluon.nn.Conv2D(256, 3, 2)
        self.bn3 = mxnet.gluon.nn.BatchNorm()
        self.dense1 = mxnet.gluon.nn.Dense(10)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = mxnet.ndarray.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = mxnet.ndarray.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = mxnet.ndarray.relu(x)
        x = mxnet.ndarray.flatten(x)
        x = self.dense1(x)
        return x

if __name__ == '__main__':
    # setup gpu device(s) 
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    num_gpus = 1
    ctx = [mxnet.gpu(i) for i in range(num_gpus)] if num_gpus else [mxnet.cpu()]

    #setup training parameters
    batch_size = 1000
    pretrain_epoch = 0 
    epoch = 50
    lr = 0.1 / 256 * batch_size
    print('*** resume pruning ***')
    print('batch size: {};\ntotal training epochs: {};\ninitial learning rate: {:.3f} with cosine decay;'.format(batch_size, epoch, lr))

    # data loader
    def transform(image, label):
        image = image.astype('float32') / 255
        image = image.transpose([2, 0, 1])
        return image, label

    train_data = mxnet.gluon.data.vision.MNIST('~/.mxnet/datasets', train=True, transform=transform)
    test_data = mxnet.gluon.data.vision.MNIST('~/.mxnet/datasets', train=False, transform=transform)
    train_loader = mxnet.gluon.data.DataLoader(train_data, batch_size, True, num_workers=0)
    test_loader = mxnet.gluon.data.DataLoader(test_data, batch_size, False, num_workers=0)
    step = train_data._data.shape[0] // batch_size

    # define network 
    model = Net()
    model.load_parameters('mxnet_mnist', ctx)
    lr_scheduler = gluoncv.utils.LRSequential([gluoncv.utils.LRScheduler('cosine', base_lr=lr, target_lr=0, nepochs=epoch, iters_per_epoch=step)])

    # pruning setup 
    target_sparsity = 0.95
    pretrain_step = 0 
    pruning_epoch = 40 
    assert pretrain_epoch <= epoch, 'reset pruning epoch to be smaller than total epoch!!'
    sparse_step = pruning_epoch * step
    frequency = step / 5
    keywords_no_sparse = ['bias', 'beta', 'gamma', 'conv0_weight'] 
    special_sparsity_dict = {'conv1_weight': 0.7}
    resume = True
    optimizer = pruning.NAGSparse(wd=1e-5, lr_scheduler=lr_scheduler, target_sparsity=target_sparsity, pretrain_step=pretrain_step, sparse_step=sparse_step, frequency=frequency, keywords_no_sparse=keywords_no_sparse, special_sparsity_dict=special_sparsity_dict, resume=resume)
    print('*** pruning setup ***')
    print('pruning optimizer is based on {};\ntarget pruning rate: {};\npruning_epoch: {};\npruning frequency: {};\nops without pruning: {}'.format(
        'NAG', 
        target_sparsity, 
        pruning_epoch, 
        frequency, 
        keywords_no_sparse)
        )
    print('special sparsity layers')
    for k, v in special_sparsity_dict.items():
        print(k, ": ", v)

    trainer = mxnet.gluon.Trainer(model.collect_params(), optimizer)
    metric = mxnet.metric.Accuracy()
    softmax_cross_entropy_loss = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()

    # training 
    print('*** start training ***')
    for i in range(epoch):
        for data, label in train_loader:
            data = mxnet.gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
            label = mxnet.gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=0)
            outputs = []
            with mxnet.autograd.record():
                for x, y in zip(data, label):
                    z = model(x)
                    loss = softmax_cross_entropy_loss(z, y)
                    loss.backward()
                    outputs.append(z)
            trainer.step(batch_size)

        for data, label in test_loader:
            data = mxnet.gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
            label = mxnet.gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=0)
            outputs = []
            for x in data:
                outputs.append(model(x))
            metric.update(label, outputs)
        _, test_acc = metric.get()
        metric.reset()

        temp_dict = {}
        for n, p in model.collect_params('.*weight').items():
            if n in ['conv0_weight']:
                continue
            temp = p.data().asnumpy()
            rate = numpy.flatnonzero(temp).size / temp.size
            temp_dict[n] = rate
        temp_rate = numpy.mean([one[1] for one in temp_dict.items()])
        print('epoch {}|accuracy={:.4f}|pruning_rate={:.4f}'.format(i, test_acc, 1.0-temp_rate))

    model.save_parameters('mxnet_mnist_resume')
    print('*** summary ***')
    print('pruning rate = (MACs of sparse processing)/(MACs of dense processing)')
    print('MACs = number of multiply-accumulate')
    print('final accuracy: {:.4f}, final pruning rate: {:.4f}'.format(test_acc, 1.0-temp_rate))
    print("*** model saved as mxnet_mnist_resume ***")
