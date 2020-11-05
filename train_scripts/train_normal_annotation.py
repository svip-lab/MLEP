import tensorflow as tf
import os

from models import prediction_networks_dict
from utils.dataloaders.only_normal_loader import NormalDataLoader
from utils.util import load, save, psnr_error
from constant import const


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPUS[0]

dataset_name = const.DATASET
train_folder = const.TRAIN_FOLDER
test_folder = const.TEST_FOLDER
frame_mask = const.FRAME_MASK
pixel_mask = const.PIXEL_MASK
k_folds = const.K_FOLDS
kth = const.KTH
interval = const.INTERVAL
multi_interval = const.MULTI_INTERVAL

batch_size = const.BATCH_SIZE
iterations = const.ITERATIONS
num_his = const.NUM_HIS
height, width = const.HEIGHT, const.WIDTH

prednet = prediction_networks_dict[const.PREDNET]

margin = const.MARGIN
lam = const.LAMBDA

model_save_freq = const.MODEL_SAVE_FREQ
summary_dir = const.SUMMARY_DIR
snapshot_dir = const.SNAPSHOT_DIR

print(const)

# define dataset
# noinspection PyUnboundLocalVariable
with tf.name_scope('dataset'):
    tf_dataset = NormalDataLoader(dataset_name, train_folder, height, width)

    train_dataset = tf_dataset(batch_size, time_steps=(num_his + 1), interval=interval)

    train_it = train_dataset.make_one_shot_iterator()
    train_tensor = train_it.get_next()
    train_tensor.set_shape([batch_size, (num_his + 1), height, width, 3])

    train_positive = train_tensor[:, 0:num_his, ...]
    train_positive_gt = train_tensor[:, -1, ...]


# define training generator function
with tf.variable_scope('generator', reuse=None):
    train_positive_output, train_positive_feature, _ = prednet(train_positive, use_decoder=True)
    train_positive_psnr = psnr_error(train_positive_output, train_positive_gt)

with tf.name_scope('training'):
    g_loss = tf.reduce_mean(tf.abs(train_positive_output - train_positive_gt))

    g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')
    g_lrate = tf.train.piecewise_constant(g_step, boundaries=const.LRATE_G_BOUNDARIES, values=const.LRATE_G)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=g_lrate, name='g_optimizer')
    g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    g_train_op = g_optimizer.minimize(g_loss, global_step=g_step, var_list=g_vars, name='g_train_op')

    # add all to summaries
    tf.summary.scalar(tensor=g_loss, name='g_loss')
    tf.summary.image(tensor=train_positive_output, name='positive_output')
    tf.summary.image(tensor=train_positive_gt, name='positive_gt')
    tf.summary.scalar(tensor=train_positive_psnr, name='positive_psnr')
    summary_op = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # summaries
    summary_writer = tf.summary.FileWriter(summary_dir, graph=sess.graph)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)
    if os.path.isdir(snapshot_dir):
        ckpt = tf.train.get_checkpoint_state(snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')
    else:
        load(loader, sess, snapshot_dir)

    print('Start training ...')

    _step, _loss, _summaries = 0, None, None
    while _step < iterations:
        try:
            _, _step, _g_loss, _p_psnr, _summaries = \
                sess.run([g_train_op, g_step, g_loss, train_positive_psnr, summary_op])

            if _step % 10 == 0:
                print('Iteration = {}, global loss = {:.6f}, positive psnr = {:.6f}'.format(_step, _g_loss, _p_psnr))

            if _step % 100 == 0:
                summary_writer.add_summary(_summaries, global_step=_step)
                print('Save summaries...')

            if _step % model_save_freq == 0:
                save(saver, sess, snapshot_dir, _step)

        except tf.errors.OutOfRangeError:
            print('Finish successfully!')
            save(saver, sess, snapshot_dir, _step)
            break
