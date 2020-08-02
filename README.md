## Pruning Tools by Moffett AI
**<font size='3'>This repo provides a wide spectrum of tools for neural network pruning in major deep learning platforms, including tensorflow, pytorch and mxnet.**


**<font size='3'>Two modes for using the pruning tools by Moffett AI:<font>**
1. Prune dense networks 

In order to prune their neural networks, users only need to replace the optimizer in their codes by the one provided by Moffett AI. No change in other codes is needed! For example: 

    # import Moffett optimizer in pytorch training script 
    from optimizers import pytorch_pruning as pruning
    ......
    # replace SGD optimizer by Moffett AI's optimizer  
    # optimizer = torch.optim.SGD(...)
    optimizer = pruning.SGDSparse(...)
    ......

2. Finetune the sparse networks on users' own datasets 

Users can also use the sparse networks provided by Moffett AI to finetune on their own dataset, while the sparsity is kept. 

---

**<font size='3'>This repo includes the following contents:</font>**

**<font size='3'>1. pruning optimizers using in tensorflow, pytorch and mxnet：</font>**
 * [x] mxnet_pruning.NAGSparse
 * [x] mxnet_pruning.NadamSparse
 * [x] pytorch_pruning.SGDSparse
 * [x] pytorch_pruning.AdamSparse
 * [x] tensorflow1_pruning.MomentumOptimizerSparse (not support multiple GPUs so far)
 * [ ] tensorflow1_pruning.AdamOptimizerSparse

Detailed documents for pruning optimizers:
* [pytorch pruning optimizer](./docs/pytorch_parameters.md)
* [mxnet pruning optimizer](./docs/mxnet_parameters.md)
* [tensorflow v1 pruning optimizer](./docs/tensorflow1_parameters.md)

---

**<font size='3'>2. Examples of using pruning optimizers on mnist dataset:</font>**

<font size='3'>2.1. prune dense network:</font>

 * [x] mxnet version：
    `python3.7 examples/mnist/mxnet_pruning_simple.py`

 * [x] pytorch version:
    `python3.7 examples/mnist/pytorch_pruning_simple.py`

  * [x] tensorflow version:
    `python3.7 examples/mnist/tensorflow_pruning_simple.py`

   <font size='3'>2.2. continue pruning/training from a sparse network：</font> ([results](./docs/results.md#pruning-and-finetune-results))

   * [x] mxnet version:
    `python3.7 examples/mnist/mxnet_pruning_simple_resume.py`

   * [x] pytorch version:
    `python3.7 examples/mnist/pytorch_pruning_simple_resume.py`

   * [ ] tensorflow version:
    `python3.7 examples/mnist/tensorflow1_pruning_simple_resume.py`

*notes: current optimizers only support python3.7，more versions will be released soon. [mxnet installation](https://mxnet.apache.org/get_started/?platform=macos&language=python&), [pytorch installation](https://pytorch.org/)*

---
**<font size='3'>3. finetune sparse network on users' datasets </font>**

**<font size='3'>3.1.  finetune on cifar10 dataset using the sparse network pruned on imagenet dataset:</font>**
* [x] pytorch version: ([results](./docs/results.md#task-2-finetune-sparse-network-on-classification-dataset))

    `python3.7 examples/cifar10/torch_res50_cifar10_224input.py`

    `python3.7 examples/cifar10/torch_res50_cifar10_32input.py`
* [ ] tensorflow 1.14 version
* [ ] mxnet version

**<font size='3'>3.2.  finetune on street scenes using the sparse network pruned on coco dataset:</font>** : 
* [x] pytorch version: ([results](./docs/results.md#task-3-finetune-sparse-network-on-detection-dataset))

    `python3.7 examples/cocoSubset/torch_res50_cocoSubset.py`

* [ ] tensorflow 1.14 version
* [ ] mxnet version

---
**<font size='3'>4. Moffett AI model zoo </font>**

We provide some sparse networks for users to finetune on their own datasets. More sparse networks will be constantly provided in this repo.

**<font size='3'>Classification task</font>**

|model|framework|training dataset|sparsity|top1|notes|
|-|-|-|-|-|-|
|resnet50v1d|mxnet|imagenet|95%|74.6%|pretrain model from gluoncv|
|resnet50v1d|torch|imagenet|95%|74.6%|[network in Moffett IR format<sup>1</sup>](./examples/cifar10/resnet50v1d_graph.png)|

**<font size='3'>Detection task </font>**

|model|framework|training dataset|sparsity|MAP|notes|
|-|-|-|-|-|-|
|resnet50v1d+centernet|mxnet|coco|80%|31.0%|pretrain model from gluoncv||
|resnet50v1d+centernet|torch|coco|80%|31.0%|[network in Moffett IR format<sup>1</sup>](./examples/cocoSubset/resnet50v1d_centernet.png)|

[download the sparse networks](https://drive.google.com/open?id=1xZ-lDh1CGnaFMpsQft37kyfocPf16KuR)

<sup>1</sup>**<font size='3'>[Moffett IR convert models across major deep learning platforms](./docs/reconstruct_network.md) </font>**
