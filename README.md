## Pruning Tools by Moffett AI
本工程提供Moffett AI研发的神经网络剪枝工具库，用于降低模型运算量从而加快AI模型推理速度。

**<font size='3'>Moffett AI剪枝工具使用模式:<font>**
1. [x] 用户将已有训练代码中的优化器替换为Moffett AI剪枝优化器，从而直接剪枝模型，而几乎不需要修改其他代码。例如：
    ```key
    # 导入剪枝优化器
    from optimizers import tensorflow1_pruning as pruning
    ......
    # 调用剪枝优化器替换原来的优化器
    # optimizer = tensorflow.compat.v1.train.MomentumOptimizer(...)
    optimizer = pruning.MomentumOptimizerSparse(...)
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
 * [x] tensorflow1_pruning.AdamOptimizerSparse (暂不支持多卡训练)

 剪枝优化器的命名方式为原框架优化器的名字加上"Sparse"，所以"NAGSparse"是来源于原框架的"NAG"优化器。剪枝优化器是在原框架优化器的基础上加入稀疏化功能，如果使用剪枝优化器，但参数设置为不稀疏，那么优化器的行为将等价于原框架同名优化器的行为。
 
剪枝优化器参数请查看:
* `pytorch 优化器文档`[请点此查看](./docs/pytorch_parameters.md)
* `mxnet 优化器文档`[请点此查看](./docs/mxnet_parameters.md)
* `tensorflow v1 优化器文档`[请点此查看](./docs/tensorflow1_parameters.md)

**<font size='3'>2. mnist数据集上剪枝简单CNN的使用范例:</font>**


<font size='3'>2.1. 将非稀疏模型剪枝为稀疏模型:</font>


 * [x] mxnet版本：
    `python3.7 examples/mxnet_pruning_simple.py`

 * [x] pytorch版本:
    `python3.7 examples/pytorch_pruning_simple.py`

  * [x] tensorflow版本:
    `python3.7 examples/tensorflow_pruning_simple.py`

   <font size='3'>2.2. 从已经稀疏的模型继续训练：</font>

   * [x] mxnet版本:
    `python3.7 examples/mxnet_pruning_simple_resume.py`

   * [x] pytorch版本:
    `python3.7 examples/pytorch_pruning_simple_resume.py`

   * [ ] tensorflow版本:
    `python3.7 examples/tensorflow1_pruning_simple_resume.py`

*注意事项: 目前剪枝优化器只适用于python3.7版本，其他版本会陆续更新。[mxnet安装连接](https://mxnet.apache.org/get_started/?platform=macos&language=python&), [pytorch 安装连接](https://pytorch.org/)*

**<font size='3'>3. 压缩好的模型在保持稀疏率的前提下finetune范例:</font>**
* [x] tensorflow 1.14 版本：
`python3.7 examples/tensorflow1_pruning_simple_finetune.py`

**<font size='3'>4. imagenet和coco数据集上剪枝resnet等模型的使用示例 (*to be add*)</font>**

---
我们同时提供一些已经稀疏化的预训练模型供使用，模型的稀疏率和性能见下表。模型数量会逐渐增加。目前仅提供模型，训练代码稍后也会提供。

|模型|框架|训练数据集|稀疏率|准确率|说明|
|-|-|-|-|-|-|
|resnet50_v1d|mxnet|imagenet|0%|79.1%|gluoncv提供的预训练模型|
|resnet50_v1d|mxnet|imagenet|80%|78.2%|基于gluoncv的预训练模型进行剪枝|
|resnet50_v1d|mxnet|imagenet|90%|77.0%|同上|
|resnet50_v1d|mxnet|imagenet|95%|74.6%|同上|

[点击此连接下载上述模型](https://drive.google.com/open?id=1xZ-lDh1CGnaFMpsQft37kyfocPf16KuR)