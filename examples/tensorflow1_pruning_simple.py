import os
import sys
import numpy
import tensorflow
import tensorflow_datasets
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from optimizers import tensorflow1_pruning as pruning

def network(x, training):
    with tensorflow.variable_scope('',reuse=tensorflow.AUTO_REUSE):
        x = tensorflow.layers.conv2d(x, 64, 3, 2, kernel_initializer=tensorflow.initializers.he_normal())
        x = tensorflow.nn.relu(x)

        x = tensorflow.layers.conv2d(x, 128, 3, 2, kernel_initializer=tensorflow.initializers.he_normal())
        x = tensorflow.nn.relu(x)

        x = tensorflow.layers.conv2d(x, 256, 3, 2, kernel_initializer=tensorflow.initializers.he_normal())
        x = tensorflow.nn.relu(x)

        x = tensorflow.layers.flatten(x)
        x = tensorflow.layers.dense(x, 10, kernel_initializer=tensorflow.initializers.he_normal())
        return x


if __name__ == '__main__':
    # setup training parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
    batch = 1000 
    lr = 0.1 / 256 * batch
    total_epoch = 100
    out_dir = 'tensorflow_model/mnist'

    # setup pruning parameters
    target_sparsity = 0.95 
    pretrain_epoch = 20
    pruning_epoch = 70
    assert pretrain_epoch + pruning_epoch <= total_epoch, 'reset pretrain_epoch, pruning_epoch and total_epoch'
    keywords_no_sparse = ['bias', 'beta', 'gamma', 'conv2d/kernel:0']

    # define dataset
    def parse(x):
        image = x['image']
        label = x['label']
        image = image / 255
        return {'image': image, 'label': label}

    datasets, info = tensorflow_datasets.load('mnist', with_info=True)
    train_data = datasets['train'].map(parse, -1).shuffle(batch).repeat(1).batch(batch).prefetch(1)
    test_data = datasets['test'].map(parse, -1).repeat(1).batch(batch).prefetch(1)
    step = info.splits['train'].num_examples // batch

    train_iterator = train_data.make_initializable_iterator()
    train_next = train_iterator.get_next()
    test_iterator = test_data.make_initializable_iterator()
    test_next = test_iterator.get_next()


    # defien model
    image = tensorflow.placeholder(tensorflow.float32, shape=(None, 28, 28, 1))
    label = tensorflow.placeholder(tensorflow.int64)
    training = tensorflow.placeholder(tensorflow.bool)

    logits = network(image, training)
    loss = tensorflow.losses.sparse_softmax_cross_entropy(label, logits)
    accuracy = tensorflow.metrics.accuracy(label, tensorflow.argmax(tensorflow.nn.softmax(logits), 1))

    total_parameters = 0
    nnz_dict = dict()
    for variable in tensorflow.trainable_variables():
        if any(one in variable.name for one in keywords_no_sparse):
            continue
        nonzero_rate = tensorflow.cast(tensorflow.count_nonzero(variable), tensorflow.float32) / tensorflow.cast(tensorflow.size(variable), tensorflow.float32)
        nnz_dict[variable.name] = nonzero_rate

    global_step = tensorflow.train.get_or_create_global_step()
    lr = tensorflow.train.cosine_decay(lr, global_step, step * total_epoch)

    print('*** pruning setup ***')
    minimize = pruning.MomentumOptimizerSparse(
            learning_rate=lr, 
            target_sparsity=target_sparsity, 
            pretrain_step=step * pretrain_epoch, 
            sparse_step=step * pruning_epoch, 
            frequency=step / 5, 
            keywords_no_sparse=keywords_no_sparse).minimize(loss, global_step=global_step)
    
    print('pruning optimizer is based on {};\ntarget pruning rate: {};\npruning_epoch: {};\npruning frequency: {};\nops without pruning: {};'.format(
        'momentum', 
        target_sparsity, 
        pruning_epoch, 
        step/5, 
        keywords_no_sparse)
        )

    Saver = tensorflow.train.Saver()
    Session = tensorflow.Session(config=tensorflow.ConfigProto(allow_soft_placement=True, gpu_options=tensorflow.GPUOptions(allow_growth=True), inter_op_parallelism_threads=0, intra_op_parallelism_threads=0))

    Session.run(tensorflow.global_variables_initializer())
    Session.run(tensorflow.local_variables_initializer())

    # training 
    best_acc_dense = 0
    for i in range(total_epoch):
        Session.run(train_iterator.initializer)
        while True:
            try:
                one_batch = Session.run(train_next)
                _, _global_step = Session.run([minimize, global_step], feed_dict={image: one_batch['image'], label: one_batch['label'], training: True})
            except tensorflow.errors.OutOfRangeError:
                break

        Session.run(tensorflow.variables_initializer(tensorflow.get_collection(tensorflow.GraphKeys.METRIC_VARIABLES)))
        Session.run(test_iterator.initializer)
        while True:
            try:
                one_batch = Session.run(test_next)
                _, test_acc = Session.run(accuracy, feed_dict={image: one_batch['image'], label: one_batch['label'], training: False})
            except tensorflow.errors.OutOfRangeError:
                break

        temp_rate = 1.0 - numpy.mean([v for k,v in Session.run(nnz_dict).items()])
        if temp_rate == 0.0 and test_acc > best_acc_dense:
            best_acc_dense = test_acc
        print('epoch {}|accuracy: {:.3f}|pruning_rate={:.3f}'.format(i, test_acc, temp_rate))

    Saver.save(Session, out_dir, global_step=_global_step)
    
    print('*** summary ***')
    print('pruning rate = (MACs of sparse processing)/(MACs of dense processing)')
    print('MACs = number of multiply-accumulate')
    print('best accuracy of dense model: {:.4f}'.format(best_acc_dense))
    print('final accuracy: {:.4f}, final pruning rate: {:.4f}'.format(test_acc, temp_rate))
    print('*** model saved in {} ***'.format(out_dir))
