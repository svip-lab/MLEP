import tensorflow as tf
import os

from models import prediction_networks_dict
from utils.dataloaders.tune_video_loader import DataTuneVideoGtLoaderImage
from utils.util import load, psnr_error, save
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

pretrain_model = const.PRETRAIN_MODEL
psnr_file = const.PSNR_FILE
model_save_freq = const.MODEL_SAVE_FREQ
summary_dir = const.SUMMARY_DIR
snapshot_dir = const.SNAPSHOT_DIR


print(const)

# define dataset
# noinspection PyUnboundLocalVariable
with tf.name_scope('dataset'):
    data_loader = DataTuneVideoGtLoaderImage(dataset=dataset_name, train_folder=train_folder,
                                             test_folder=test_folder,
                                             k_folds=k_folds, kth=kth,
                                             frame_mask_file=frame_mask,
                                             psnr_file=psnr_file,
                                             resize_height=height, resize_width=width)

    tf_dataset = data_loader(batch_size, time_steps=(num_his + 1), interval=interval)
    train_it = tf_dataset.make_one_shot_iterator()
    train_tensor, train_scores = train_it.get_next()
    train_tensor.set_shape([batch_size, 3, (num_his + 1), height, width, 3])
    train_scores.set_shape([batch_size, 3, (num_his + 1)])

    print(train_tensor)
    print(train_scores)

    train_anchor = train_tensor[:, 0, 0:num_his, ...]
    train_anchor_gt = train_tensor[:, 0, -1, ...]

    train_positive = train_tensor[:, 1, 0:num_his, ...]
    train_positive_gt = train_tensor[:, 1, -1, ...]
    train_positive_score = train_scores[:, 1, -1]

    train_negative = train_tensor[:, 2, 0:num_his, ...]
    train_negative_gt = train_tensor[:, 2, -1, ...]
    train_negative_score = train_scores[:, 2, -1]

# define training generator function
with tf.variable_scope('generator', reuse=None):
    train_anchor_output, train_anchor_feature, _ = prednet(train_anchor, use_decoder=True)
    train_anchor_psnr = psnr_error(train_anchor_output, train_anchor_gt)
with tf.variable_scope('generator', reuse=True):
    train_positive_output, train_positive_feature, _ = prednet(train_positive, use_decoder=True)
    train_positive_psnr = psnr_error(train_positive_output, train_positive_gt)
with tf.variable_scope('generator', reuse=True):
    train_negative_output, train_negative_feature, _ = prednet(train_negative, use_decoder=True)
    train_negative_psnr = psnr_error(train_negative_output, train_negative_gt)


with tf.name_scope('training'):
    pred_loss = tf.reduce_mean(tf.abs(train_anchor_output - train_anchor_gt)) + \
                tf.reduce_mean(train_positive_score
                               * tf.reduce_mean(tf.abs(train_positive_output - train_positive_gt), axis=[1, 2, 3]))
    # pred_loss = tf.reduce_mean(tf.abs(train_anchor_output - train_anchor_gt))

    # train_anchor_feature = tf.Print(train_anchor_feature, [tf.reduce_max(train_anchor_feature), tf.reduce_min(train_anchor_feature)])
    # train_positive_feature = tf.Print(train_positive_feature, [tf.reduce_max(train_anchor_feature), tf.reduce_min(train_anchor_feature)])
    # train_negative_feature = tf.Print(train_negative_feature, [tf.reduce_max(train_negative_feature), tf.reduce_min(train_negative_feature)])

    # inter, between different group
    intra_dis = tf.reduce_mean((train_anchor_feature - train_positive_feature) ** 2)
    inter_dis = tf.reduce_mean((train_anchor_feature - train_negative_feature) ** 2)

    margin_loss = tf.reduce_mean(tf.abs(train_positive_score - train_negative_score)) * tf.maximum(
        0.0, margin + intra_dis - inter_dis)

    g_loss = pred_loss + lam * margin_loss
    g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')
    g_lrate = tf.train.piecewise_constant(g_step, boundaries=const.LRATE_G_BOUNDARIES, values=const.LRATE_G)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=g_lrate, name='g_optimizer')
    g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    g_train_op = g_optimizer.minimize(g_loss, global_step=g_step, var_list=g_vars, name='g_train_op')

    # add all to summaries
    tf.summary.scalar(tensor=g_loss, name='g_loss')
    tf.summary.scalar(tensor=pred_loss, name='pred_loss')
    tf.summary.scalar(tensor=margin_loss, name='margin_loss')
    tf.summary.scalar(tensor=intra_dis, name='intra_dis')
    tf.summary.scalar(tensor=inter_dis, name='inter_dis')
    tf.summary.scalar(tensor=train_anchor_psnr, name='anchor_psnr')
    tf.summary.scalar(tensor=train_positive_psnr, name='positive_psnr')
    tf.summary.scalar(tensor=train_negative_psnr, name='negative_psnr')
    tf.summary.image(tensor=train_positive_output, name='positive_output')
    tf.summary.image(tensor=train_positive_gt, name='positive_gt')
    tf.summary.image(tensor=train_negative_output, name='negative_output')
    tf.summary.image(tensor=train_negative_gt, name='negative_gt')
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
        elif pretrain_model:
            load(loader, sess, pretrain_model)
            print('Pretrain model from {}.'.format(pretrain_model))
        else:
            print('No checkpoint file found.')
    else:
        load(loader, sess, snapshot_dir)

    _step, _loss, _summaries = 0, None, None
    while _step < iterations:
        try:
            _, _step, _inter_dis, _intra_dis, _pred_loss, _margin_loss, _g_loss, _a_psnr, _p_psnr, _n_psnr = \
                sess.run([g_train_op, g_step, inter_dis, intra_dis, pred_loss, margin_loss, g_loss,
                          train_anchor_psnr, train_positive_psnr, train_negative_psnr])

            print('Training, pred loss = {:.6f}, margin loss = {:.6f}'.format(_pred_loss, _margin_loss))
            if _step % 10 == 0:
                print('Iteration = {}, global     loss = {:.6f}'.format(_step, _g_loss))
                print('                     intra      dis  = {:.6f}'.format(_intra_dis))
                print('                     inter      dis  = {:.6f}'.format(_inter_dis))
                print('                     anchor     psnr = {:.6f}'.format(_a_psnr))
                print('                     positive   psnr = {:.6f}'.format(_p_psnr))
                print('                     negative   psnr = {:.6f}'.format(_n_psnr))

            if _step % 250 == 0:
                _summaries = sess.run(summary_op)
                summary_writer.add_summary(_summaries, global_step=_step)
                print('Save summaries...')

            if _step % model_save_freq == 0:
                save(saver, sess, snapshot_dir, _step)

        except tf.errors.OutOfRangeError:
            print('Finish successfully!')
            save(saver, sess, snapshot_dir, _step)
            break
