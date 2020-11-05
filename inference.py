import tensorflow as tf
import os
import time
import numpy as np
import pickle
from scipy import interpolate

from constant import const
from models import prediction_networks_dict
from utils.dataloaders.test_loader import DataTemporalGtLoader
from utils.util import psnr_error, load

import evaluate

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

batch_size = const.BATCH_SIZE
iterations = const.ITERATIONS
num_his = const.NUM_HIS
height, width = const.HEIGHT, const.WIDTH

prednet = prediction_networks_dict[const.PREDNET]
evaluate_name = const.EVALUATE

margin = const.MARGIN
lam = const.LAMBDA

summary_dir = const.SUMMARY_DIR
snapshot_dir = const.SNAPSHOT_DIR
psnr_dir = const.PSNR_DIR

print(const)

# define dataset
# noinspection PyUnboundLocalVariable
with tf.name_scope('dataset'):
    video_clips_tensor = tf.placeholder(shape=[1, (num_his + 1), height, width, 3], dtype=tf.float32)
    inputs = video_clips_tensor[:, 0:num_his, ...]
    frame_gts = video_clips_tensor[:, -1, ...]

# define training generator function
with tf.variable_scope('generator', reuse=None):
    outputs, features, _ = prednet(inputs=inputs, use_decoder=True)
    psnr_tensor = psnr_error(outputs, frame_gts)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # dataset
    data_loader = DataTemporalGtLoader(dataset=dataset_name, folder=test_folder, k_folds=k_folds, kth=kth,
                                       frame_mask_file=frame_mask, pixel_mask_file=pixel_mask,
                                       resize_height=height, resize_width=width)
    video_info = data_loader.test_videos_info
    frame_masks = data_loader.get_frame_mask
    num_videos = len(video_info)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)

    def inference_func(ckpt, dataset_name, evaluate_name):
        load(loader, sess, ckpt)

        psnr_records = []
        total = 0
        timestamp = time.time()

        if const.INTERPOLATION:
            vol_size = num_his + 1
            for v_id, (video_name, video) in enumerate(video_info.items()):
                length = video['length']
                total += length
                gts = frame_masks[v_id]

                x_ids = np.arange(0, length, vol_size)
                x_ids[-1] = length - 1
                psnrs_ids = np.empty(shape=(len(x_ids),), dtype=np.float32)

                for i, t in enumerate(x_ids):
                    if t == length - 1:
                        start = length - vol_size
                        end = length
                    else:
                        start = t
                        end = t + vol_size

                    video_clip = data_loader.get_video_clip(video_name, start, end)
                    psnr = sess.run(psnr_tensor, feed_dict={video_clips_tensor: video_clip[np.newaxis, ...]})
                    psnrs_ids[i] = psnr

                    print('video = {} / {}, i = {} / {}, psnr = {:.6f}, gt = {}'.format(
                        video_name, num_videos, t, length, psnr, gts[end - 1]))

                # interpretation
                inter_func = interpolate.interp1d(x_ids, psnrs_ids)
                ids = np.arange(0, length)
                psnrs = inter_func(ids)
                psnr_records.append(psnrs)

        else:
            for v_id, (video_name, video) in enumerate(video_info.items()):
                length = video['length']
                total += length
                psnrs = np.empty(shape=(length,), dtype=np.float32)
                gts = frame_masks[v_id]

                for i in range(num_his, length):
                    video_clip = data_loader.get_video_clip(video_name, i - num_his, i + 1)
                    psnr = sess.run(psnr_tensor, feed_dict={video_clips_tensor: video_clip[np.newaxis, ...]})
                    psnrs[i] = psnr

                    print('video = {} / {}, i = {} / {}, psnr = {:.6f}, gt = {}'.format(
                        video_name, num_videos, i, length, psnr, gts[i]))

                psnrs[0:num_his] = psnrs[num_his]
                psnr_records.append(psnrs)

        result_dict = {'dataset': dataset_name, 'psnr': psnr_records, 'diff_mask': [], 'frame_mask': frame_masks}

        used_time = time.time() - timestamp
        print('total time = {}, fps = {}'.format(used_time, total / used_time))

        # TODO specify what's the actual name of ckpt.
        pickle_path = os.path.join(psnr_dir, os.path.split(ckpt)[-1])
        with open(pickle_path, 'wb') as writer:
            pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)

        results = evaluate.evaluate(evaluate_name, pickle_path)
        print(results)


    if os.path.isdir(snapshot_dir):
        def check_ckpt_valid(ckpt_name):
            is_valid = False
            ckpt = ''
            if ckpt_name.startswith('model.ckpt-'):
                ckpt_name_splits = ckpt_name.split('.')
                ckpt = str(ckpt_name_splits[0]) + '.' + str(ckpt_name_splits[1])
                ckpt_path = os.path.join(snapshot_dir, ckpt)
                if os.path.exists(ckpt_path + '.index') and os.path.exists(ckpt_path + '.meta') and \
                        os.path.exists(ckpt_path + '.data-00000-of-00001'):
                    is_valid = True

            return is_valid, ckpt

        def scan_psnr_folder():
            tested_ckpt_in_psnr_sets = set()
            for test_psnr in os.listdir(psnr_dir):
                tested_ckpt_in_psnr_sets.add(test_psnr)
            return tested_ckpt_in_psnr_sets

        def scan_model_folder():
            saved_models = set()
            for ckpt_name in os.listdir(snapshot_dir):
                is_valid, ckpt = check_ckpt_valid(ckpt_name)
                if is_valid:
                    saved_models.add(ckpt)
            return saved_models

        tested_ckpt_sets = scan_psnr_folder()
        while True:
            all_model_ckpts = scan_model_folder()
            new_model_ckpts = all_model_ckpts - tested_ckpt_sets

            for ckpt_name in new_model_ckpts:
                # inference
                ckpt = os.path.join(snapshot_dir, ckpt_name)
                inference_func(ckpt, dataset_name, evaluate_name)

                tested_ckpt_sets.add(ckpt_name)

            print('waiting for models...')
            evaluate.evaluate('compute_auc', psnr_dir)
            time.sleep(300)
    else:
        inference_func(snapshot_dir, dataset_name, evaluate_name)




