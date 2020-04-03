## Pruning Tools by Moffett AI
**<font size='3'>本工程提供Moffett AI研发的神经网络剪枝工具库，用于降低模型运算量从而加快AI模型推理速度<font>**

**<font size='3'>Moffett AI剪枝工具使用模式:<font>**
1. [x] 用户将已有训练代码中的优化器替换为Moffett AI剪枝优化器，从而直接剪枝模型，而几乎不需要修改其他代码。例如：
    ```key
    # 导入剪枝优化器
    from optimizers import pytorch_pruning as pruning
    ......
    # 调用剪枝优化器替换原来的优化器
    # optimizer = torch.optim.SGD(...)
    optimizer = pruning.SGDSparse(...)
    ......
    ```

2. [x] 用户可将模型替换成Moffett AI提供的稀疏化预训练模型，在保持稀疏率的前提下finetune在用户数据。
---

**<font size='3'>本工程包含以下内容:</font>**

**<font size='3'>1. optimizer文件夹包含不同框架的多种剪枝优化器。目前包含：</font>**
 * [x] mxnet_pruning.NAGSparse
 * [x] mxnet_pruning.NadamSparse
 * [x] pytorch_pruning.SGDSparse
 * [x] pytorch_pruning.AdamSparse
 * [x] tensorflow1_pruning.MomentumOptimizerSparse (暂不支持多卡训练)
 * [ ] tensorflow1_pruning.AdamOptimizerSparse (暂不支持多卡训练)

 剪枝优化器的命名方式为原框架优化器的名字加上"Sparse"，所以"NAGSparse"是来源于原框架的"NAG"优化器。剪枝优化器是在原框架优化器的基础上加入稀疏化功能，如果使用剪枝优化器，但参数设置为不稀疏，那么优化器的行为将等价于原框架同名优化器的行为。

剪枝优化器参数请查看:
* `pytorch 优化器文档`[请点此查看](./docs/pytorch_parameters.md)
* `mxnet 优化器文档`[请点此查看](./docs/mxnet_parameters.md)
* `tensorflow v1 优化器文档`[请点此查看](./docs/tensorflow1_parameters.md)

---

**<font size='3'>2. mnist数据集上剪枝简单CNN的使用范例:</font>**


<font size='3'>2.1. 将非稀疏模型剪枝为稀疏模型:</font>


 * [x] mxnet版本：
    `python3.7 examples/mnist/mxnet_pruning_simple.py`

 * [x] pytorch版本:
    `python3.7 examples/mnist/pytorch_pruning_simple.py`

  * [x] tensorflow版本:
    `python3.7 examples/mnist/tensorflow_pruning_simple.py`

   <font size='3'>2.2. 从已经稀疏的模型继续训练：</font> ([实验结果](./docs/results.md#pruning-and-finetune-results))

   * [x] mxnet版本:
    `python3.7 examples/mnist/mxnet_pruning_simple_resume.py`

   * [x] pytorch版本:
    `python3.7 examples/mnist/pytorch_pruning_simple_resume.py`

   * [ ] tensorflow版本:
    `python3.7 examples/mnist/tensorflow1_pruning_simple_resume.py`

*注意事项: 目前剪枝优化器只适用于python3.7版本，其他版本会陆续更新。[mxnet安装连接](https://mxnet.apache.org/get_started/?platform=macos&language=python&), [pytorch 安装连接](https://pytorch.org/)*

---
**<font size='3'>3. 稀疏化模型在不同数据集的finetune </font>**

**<font size='3'>3.1.  imagenet数据集压缩后的模型在cifar10数据集上finetune:</font>**
* [x] pytorch版本: ([实验结果](./docs/results.md#task-2-finetune-sparse-network-on-classification-dataset))

    `python3.7 examples/cifar10/torch_res50_cifar10_224input.py`

    `python3.7 examples/cifar10/torch_res50_cifar10_32input.py`
* [ ] tensorflow1.14版本
* [ ] mxnet版本

**<font size='3'>3.2.  coco数据集压缩后的模型在全目标检测数据集上finetune:</font>** : 
* [x] pytorch版本: ([实验结果](./docs/results.md#task-3-finetune-sparse-network-on-detection-dataset))

    `python3.7 examples/cocoSubset/torch_res50_cocoSubset.py`

* [ ] tensorflow1.14版本
* [ ] mxnet版本

---
**<font size='3'>4. Moffett AI model zoo </font>**

我们同时提供一些已经稀疏化的预训练模型供使用，模型的稀疏率和性能见下表。模型数量会逐渐增加。目前仅提供模型，训练代码稍后也会提供。

**<font size='3'>分类模型 </font>**

|模型|框架|训练数据集|稀疏率|准确率|说明|
|-|-|-|-|-|-|
|resnet50v1d|mxnet|imagenet|95%|74.6%|同上|
|resnet50v1d|torch|imagenet|95%|74.6%|[Moffett IR生成<sup>1</sup>](./examples/cifar10/resnet50v1d_graph.png)|

**<font size='3'>检测模型 </font>**

|模型|框架|训练数据集|稀疏率|MAP|说明|
|-|-|-|-|-|-|
|resnet50v1d+centernet|mxnet|coco|80%|31.0%|基于gluoncv的预训练模型进行剪枝||
|resnet50v1d+centernet|torch|coco|80%|31.0%|[Moffett IR生成<sup>1</sup>](./examples/cocoSubset/resnet50v1d_centernet.png)|

[点击此连接Google drive下载模型](https://drive.google.com/open?id=1xZ-lDh1CGnaFMpsQft37kyfocPf16KuR)

[点击此连接Baidu drive下载](https://pan.baidu.com/s/1fL0WYtDJohzujl9AeZbY3w)，提取码：9irl。

<sup>1</sup>**<font size='3'>[Moffett IR转换不同深度学习框架模型的使用说明](./docs/reconstruct_network.md) </font>**
