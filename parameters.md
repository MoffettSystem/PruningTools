## 稀疏优化器参数：

稀疏优化器有两种参数。第一种参数是原框架优化器的参数，比如lr、momentum等，这种参数的作用与原框架同名参数等价。另一种参数是稀疏化参数，比如target_sparsity等，这种参数设置稀疏优化算法的行为。所有原框架优化器可设置的参数，本项目相应的的稀疏化版本都有该参数且行为等价。稀疏优化器的参数不会影响原框架优化器的参数。

pytorch、tensorflow、mxnet这三种框架的优化器几乎都有相同的稀疏化参数（除了pytorch有一个特殊的参数param_name），参数的名字和使用方法都完全相同。

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

        指定稀疏率不同于target_sparsity，需要设定特别稀疏率的变量。例如special_sparsity_dict={'layer1.weight':0.2}。

* sparse_block (dict, default={'1d': [1], '2d': [1, 1], '4d': [-1, 4, -1, 1]}):

        设置分组压缩的分组策略，该选项与硬件相关，目前仅支持默认参数。另外由于硬件限制，目前要求卷积层的输入和输出通道都必须可被4整除，否则该层不支持被压缩。

* resume (bool, default=False):

        设置是否从已经稀疏化的模型继续训练。在训练中断恢复或finetune时使用。

* L1_regularization (bool, default=False):

        设置是否使用L1正则化。L1正则化能诱导稀疏，但是目前实验发现准确率下降很多，暂时不推荐使用。

* param_name (iterator):

        该参数是pytorch优化器特有的，调用pytorch的稀疏优化器时，必须传入param_name=model.named_parameters()。可参考使example中的例子如何使用。