import os
from collections import OrderedDict
import numpy as np
import tensorflow as tf

from utils.util import load_frame
from utils.dataloaders import BaseDataAbstractLoader, LazyProperty, RNG



class DataTuneVideoGtLoaderImage(BaseDataAbstractLoader):
    def __init__(self, dataset, train_folder, test_folder, k_folds, kth,
                 frame_mask_file, pixel_mask_file='', psnr_file='',
                 resize_height=256, resize_width=256):

        super().__init__(dataset=dataset, folder=train_folder,
                         resize_height=resize_height, resize_width=resize_width)

        self.k_folds = k_folds
        self.kth = kth
        self.frame_mask_file = frame_mask_file
        self.pixel_mask_file = pixel_mask_file
        self.psnr_file = psnr_file
        self.test_folder = test_folder

        self.score_min = 0.5
        self.score_max = 0.5

        self.train_videos_info = None
        self.val_videos_info = None
        self.test_videos_info = None

        self._setup()

    def __call__(self, batch_size, time_steps, interval=1, is_training=True):
        if is_training:
            train_clips_dict = self.sample_normal_abnormal_clips_scores(self.train_videos_info, time_steps, interval)
            val_clips_dict = self.sample_normal_abnormal_clips_scores(self.val_videos_info, time_steps, interval,
                                                                      max_scores=self.score_max,
                                                                      min_scores=self.score_min)
            dataset = self.convert_to_tf_dataset_training(train_clips_dict, val_clips_dict, batch_size, time_steps)
        else:
            test_clips_dict = self.sample_normal_abnormal_clips_scores(self.test_videos_info, time_steps, interval)
            dataset = self.convert_to_tf_dataset_testing(test_clips_dict, batch_size, time_steps)
        return dataset

    def debug(self, batch_size, time_steps, interval=1, is_training=True):
        train_clips_dict = self.sample_normal_abnormal_clips_scores(self.train_videos_info, time_steps, interval)
        val_clips_dict = self.sample_normal_abnormal_clips_scores(self.val_videos_info, time_steps, interval)

        train_normal_clips = train_clips_dict['normal']
        train_normal_number_list = train_clips_dict['normal_numbers']
        train_normal_scores = train_clips_dict['normal_scores']

        val_abnormal_clips = val_clips_dict['abnormal']
        val_abnormal_scores = val_clips_dict['abnormal_scores']
        val_abnormal_number_list = val_clips_dict['abnormal_numbers']

        val_normal_clips = val_clips_dict['normal']
        val_normal_scores = val_clips_dict['normal_scores']
        val_normal_number_list = val_clips_dict['normal_numbers']

        num_train_normal_videos = len(train_normal_clips)
        num_val_normal_videos = len(val_normal_clips)
        num_val_abnormal_videos = len(val_abnormal_clips)

        height, width = self.resize_height, self.resize_width

        def _sample_video_clip_score(clips_list, length, scores_list):
            clip = np.empty(shape=[time_steps, height, width, 3], dtype=np.float32)

            sample_idx = RNG.randint(length)
            for t, filename in enumerate(clips_list[sample_idx]):
                clip[t, ...] = load_frame(filename, height, width)
            return clip, np.array(scores_list[sample_idx])

        def video_clip_generator():
            while True:
                batch_video_clips = []
                batch_video_scores = []

                for i in range(batch_size):
                    a_vid = RNG.randint(num_train_normal_videos)
                    p_vid = RNG.randint(num_val_normal_videos)
                    n_vid = RNG.randint(num_val_abnormal_videos)

                    video_clips = np.empty(shape=[3, time_steps, height, width, 3], dtype=np.float32)
                    video_scores = np.empty(shape=[3, time_steps], dtype=np.float32)
                    video_clips[0, ...], video_scores[0, ...] = _sample_video_clip_score(train_normal_clips[a_vid],
                                                                                         train_normal_number_list[a_vid],
                                                                                         train_normal_scores[a_vid])
                    video_clips[1, ...], video_scores[1, ...] = _sample_video_clip_score(val_normal_clips[p_vid],
                                                                                         val_normal_number_list[p_vid],
                                                                                         val_normal_scores[p_vid])
                    video_clips[2, ...], video_scores[2, ...] = _sample_video_clip_score(val_abnormal_clips[n_vid],
                                                                                         val_abnormal_number_list[n_vid],
                                                                                         val_abnormal_scores[n_vid])

                    batch_video_clips.append(video_clips)
                    batch_video_scores.append(video_scores)

                batch_video_clips = np.stack(batch_video_clips, axis=0)
                batch_video_scores = np.stack(batch_video_scores, axis=0)

                yield batch_video_clips, batch_video_scores

        return video_clip_generator

    def convert_to_tf_dataset_training(self, train_clips_dict, val_clips_dict, batch_size, time_steps):
        train_normal_clips = train_clips_dict['normal']
        train_normal_number_list = train_clips_dict['normal_numbers']
        train_normal_scores = train_clips_dict['normal_scores']

        val_abnormal_clips = val_clips_dict['abnormal']
        val_abnormal_scores = val_clips_dict['abnormal_scores']
        val_abnormal_number_list = val_clips_dict['abnormal_numbers']

        val_normal_clips = val_clips_dict['normal']
        val_normal_scores = val_clips_dict['normal_scores']
        val_normal_number_list = val_clips_dict['normal_numbers']

        num_train_normal_videos = len(train_normal_clips)
        num_val_normal_videos = len(val_normal_clips)
        num_val_abnormal_videos = len(val_abnormal_clips)

        height, width = self.resize_height, self.resize_width

        def _sample_video_clip_score(clips_list, length, scores_list):
            clip = np.empty(shape=[time_steps, height, width, 3], dtype=np.float32)

            sample_idx = RNG.randint(length)
            for t, filename in enumerate(clips_list[sample_idx]):
                clip[t, ...] = load_frame(filename, height, width)
            return clip, np.array(scores_list[sample_idx])

        def video_clip_generator():
            while True:
                a_vid = RNG.randint(num_train_normal_videos)
                p_vid = RNG.randint(num_val_normal_videos)
                n_vid = RNG.randint(num_val_abnormal_videos)

                video_clips = np.empty(shape=[3, time_steps, height, width, 3], dtype=np.float32)
                video_scores = np.empty(shape=[3, time_steps], dtype=np.float32)
                video_clips[0, ...], video_scores[0, ...] = _sample_video_clip_score(train_normal_clips[a_vid],
                                                                                     train_normal_number_list[a_vid],
                                                                                     train_normal_scores[a_vid])
                video_clips[1, ...], video_scores[1, ...] = _sample_video_clip_score(val_normal_clips[p_vid],
                                                                                     val_normal_number_list[p_vid],
                                                                                     val_normal_scores[p_vid])
                video_clips[2, ...], video_scores[2, ...] = _sample_video_clip_score(val_abnormal_clips[n_vid],
                                                                                     val_abnormal_number_list[n_vid],
                                                                                     val_abnormal_scores[n_vid])

                # show_triplet_video_clips(video_clips)
                yield video_clips, video_scores

        # video clip paths
        dataset = tf.data.Dataset.from_generator(generator=video_clip_generator,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=([3, time_steps, height, width, 3], [3, time_steps]))
        dataset = dataset.prefetch(buffer_size=128)
        dataset = dataset.shuffle(buffer_size=128).batch(batch_size)
        return dataset

    def convert_to_tf_dataset_testing(self, val_clips_dict, batch_size, time_steps):
        val_abnormal_clips = val_clips_dict['abnormal']
        val_abnormal_scores = val_clips_dict['abnormal_scores']
        val_abnormal_number_list = val_clips_dict['abnormal_numbers']

        val_normal_clips = val_clips_dict['normal']
        val_normal_scores = val_clips_dict['normal_scores']
        val_normal_number_list = val_clips_dict['normal_numbers']

        num_val_normal_videos = len(val_normal_clips)
        num_val_abnormal_videos = len(val_abnormal_clips)

        height, width = self.resize_height, self.resize_width

        def _sample_video_clip_score(clips_list, length, scores_list):
            clip = np.empty(shape=[time_steps, height, width, 3], dtype=np.float32)

            sample_idx = RNG.randint(length)
            for t, filename in enumerate(clips_list[sample_idx]):
                clip[t, ...] = load_frame(filename, height, width)
            return clip, np.array(scores_list[sample_idx])

        def video_clip_generator():
            while True:
                a_vid, p_vid = RNG.choice(num_val_normal_videos, size=2, replace=False)
                n_vid = RNG.randint(num_val_abnormal_videos)

                video_clips = np.empty(shape=[3, time_steps, height, width, 3], dtype=np.float32)
                video_scores = np.empty(shape=[3, time_steps], dtype=np.float32)
                video_clips[0, ...], video_scores[0, ...] = _sample_video_clip_score(val_normal_clips[a_vid],
                                                                                     val_normal_number_list[a_vid],
                                                                                     val_normal_scores[a_vid])
                video_clips[1, ...], video_scores[1, ...] = _sample_video_clip_score(val_normal_clips[p_vid],
                                                                                     val_normal_number_list[p_vid],
                                                                                     val_normal_scores[p_vid])
                video_clips[2, ...], video_scores[2, ...] = _sample_video_clip_score(val_abnormal_clips[n_vid],
                                                                                     val_abnormal_number_list[n_vid],
                                                                                     val_abnormal_scores[n_vid])

                # show_triplet_video_clips(video_clips)
                yield video_clips, video_scores

        # video clip paths
        dataset = tf.data.Dataset.from_generator(generator=video_clip_generator,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=([3, time_steps, height, width, 3], [3, time_steps]))
        dataset = dataset.prefetch(buffer_size=128)
        dataset = dataset.shuffle(buffer_size=128).batch(batch_size)
        return dataset

    def _setup(self):
        videos_info = self.parse_videos_folder(self.test_folder)
        frame_mask = self.load_frame_mask(videos_info, self.frame_mask_file)
        pixel_mask_list = self.load_pixel_mask_file_list(videos_info, self.pixel_mask_file)
        scores = self.load_frame_scores(self.psnr_file)

        num_videos = len(videos_info)
        k_folds_ids = np.array_split(np.arange(num_videos), self.k_folds)

        val_ids = k_folds_ids[self.kth - 1].tolist()
        test_ids = []
        for k in range(self.k_folds):
            if k != (self.kth - 1):
                test_ids += k_folds_ids[k].tolist()

        def add_gt_to_info(ids):
            videos_info_gt = OrderedDict()
            videos_names = list(videos_info.keys())

            for i in ids:
                v_name = videos_names[i]
                pixel_mask = pixel_mask_list[i] if pixel_mask_list else []

                videos_info_gt[v_name] = {
                    'length': videos_info[v_name]['length'],
                    'images': videos_info[v_name]['images'],
                    'frame_mask': frame_mask[i],
                    'pixel_mask': pixel_mask,
                    'scores': scores[i]
                }

            return videos_info_gt

        val_videos_info = add_gt_to_info(val_ids)
        test_videos_info = add_gt_to_info(test_ids)

        del videos_info

        self.val_videos_info = val_videos_info
        self.test_videos_info = test_videos_info

        self.train_videos_info = self.parse_videos_folder(self.folder)

    def get_video_clip(self, video, start, end, interval=1):
        # assert video_name in self._videos_info, 'video {} is not in {}!'.format(video_name, self._videos_info.keys())
        # assert (start >= 0) and (start <= end) and (end < self._videos_info[video_name]['length'])

        video_idx = np.arange(start, end, interval)
        video_clip = np.empty(shape=[len(video_idx), self.resize_height, self.resize_width, 3], dtype=np.float32)
        for idx, v_idx in enumerate(video_idx):
            filename = self.test_videos_info[video]['images'][v_idx]
            video_clip[idx, ...] = load_frame(filename, self.resize_height, self.resize_width)

        return video_clip

    def get_video_names(self):
        return list(self.test_videos_info)

    @LazyProperty
    def get_frame_mask(self):
        frame_mask = []
        for video_info in self.test_videos_info.values():
            frame_mask.append(video_info['frame_mask'])

        return frame_mask

    @LazyProperty
    def get_total_frames(self):
        total = 0
        for video_info in self.test_videos_info.values():
            total += video_info['length']
        return total
