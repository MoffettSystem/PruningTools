## tensorflow v1 框架优化器文档：

稀疏优化器有两种参数。第一种参数是原框架优化器的参数，比如lr、momentum等。由于一些原因，tensorflow v1 的稀疏优化器暂时还无法做到行为等同于原框架的优化器。

另一种参数是稀疏化参数，比如target_sparsity等，这种参数设置稀疏优化算法的行为。稀疏优化器的参数不会影响原框架优化器的参数。相同框架的目稀疏优化器都有相同的稀疏化参数，且功能相同。

## 目前提供以下优化器：

* MomentumOptimizerSparse

    该优化器有以下特有参数：

    * learning_rate:

            设置学习率。

    * momentum (default=0.9):

            设置momentum。

    * use_locking (default=False):

            是否使用变量锁。

## 稀疏优化器的参数：
* init_sparsity=0 (float, default=0):

        设置初始稀疏率。

* target_sparsity (float, default=0):

        设置最终稀疏率。

* pretrain_step (int, default=0):

        执行稀疏训练之前先执行多少步预训练。如果是从随机初始化训练模型，那么pretrain_step可以设置为适合的步数。如果是从已经训练好的模型直接压缩，那么pretrain_step可以设置为0。

* sparse_step (int, default=0):

        执行稀疏训练多少步。在这个期间内，每frequency次就执行一次压缩。

* frequency (int, default=100):

        执行稀疏化算法的频率。frequency不建议设置的太小或太大，我们的实验设置为几百是比较合适的。

* keywords_no_sparse (list, default=['bias', 'beta', 'gamma']):

        设置网络参数中包含哪些关键字的变量不执行稀疏化，默认为batch norm层不稀疏，所有的bias不稀疏。

* special_sparsity_dict (dict, default={}):

        指定稀疏率不同于target_sparsity，需要设定特别稀疏率的变量。例如special_sparsity_dict={'conv2d/kernel:0':0.2}。

* enable_channel_balance (bool, default=True):

        设置分组压缩的分组策略，该选项与硬件相关，目前仅支持默认参数。另外由于硬件限制，目前要求卷积层的输入和输出通道都必须可被4整除，否则该层不支持被压缩。

* finetune (bool, default=False):

        设置是否从已经稀疏化的模型继续训练。在finetune时设置为True，此时会固定稀疏率不再变化。