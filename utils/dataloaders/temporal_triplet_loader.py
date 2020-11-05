import os
from collections import OrderedDict
import numpy as np
import tensorflow as tf

from utils.util import load_frame
from utils.dataloaders import BaseDataAbstractLoader, LazyProperty, RNG


class DataTemporalGtLoader(BaseDataAbstractLoader):

    def __init__(self, dataset, folder, resize_height, resize_width, k_folds, kth,
                 frame_mask_file, pixel_mask_file=''):
        super().__init__(dataset=dataset, folder=folder, resize_height=resize_height, resize_width=resize_width)

        self.k_folds = k_folds
        self.kth = kth
        self.frame_mask_file = frame_mask_file
        self.pixel_mask_file = pixel_mask_file

        self.val_videos_info = None
        self.test_videos_info = None

        self._setup()

    def __call__(self, batch_size, time_steps, interval=1):
        val_clips_list = self.sample_normal_abnormal_clips(self.val_videos_info, time_steps, interval)
        test_clips_list = self.sample_normal_abnormal_clips(self.test_videos_info, time_steps, interval)

        val_dataset = self.convert_to_tf_dataset(val_clips_list, batch_size, time_steps=time_steps)
        test_dataset = self.convert_to_tf_dataset(test_clips_list, batch_size, time_steps=time_steps)

        return val_dataset, test_dataset

    def convert_to_tf_dataset(self, videos_clips_dict, batch_size, time_steps):
        normal_clips = videos_clips_dict['normal']
        normal_number_list = videos_clips_dict['normal_numbers']
        abnormal_clips = videos_clips_dict['abnormal']
        abnormal_number_list = videos_clips_dict['abnormal_numbers']

        num_normal_videos = len(normal_clips)
        num_abnormal_videos = len(abnormal_clips)

        height, width = self.resize_height, self.resize_width

        def _sample_video_clip(clips_list, length):
            clip = np.empty(shape=[time_steps, height, width, 3], dtype=np.float32)

            sample_idx = RNG.randint(length)
            for t, filename in enumerate(clips_list[sample_idx]):
                clip[t, ...] = load_frame(filename, height, width)

            return clip

        def video_clip_generator():
            while True:
                a_vid, p_vid = RNG.choice(num_normal_videos, size=2)
                n_vid = RNG.randint(num_abnormal_videos)

                video_clips = np.empty(shape=[3, time_steps, height, width, 3], dtype=np.float32)
                video_clips[0, ...] = _sample_video_clip(normal_clips[a_vid], normal_number_list[a_vid])
                video_clips[1, ...] = _sample_video_clip(normal_clips[p_vid], normal_number_list[p_vid])
                video_clips[2, ...] = _sample_video_clip(abnormal_clips[n_vid], abnormal_number_list[n_vid])

                # show_triplet_video_clips(video_clips)
                yield video_clips

        # video clip paths
        dataset = tf.data.Dataset.from_generator(generator=video_clip_generator,
                                                 output_types=tf.float32,
                                                 output_shapes=[3, time_steps, height, width, 3])
        print('generator dataset, {}'.format(dataset))
        dataset = dataset.prefetch(buffer_size=256)
        dataset = dataset.shuffle(buffer_size=256).batch(batch_size)
        print('epoch dataset, {}'.format(dataset))

        return dataset

    def convert_to_tf_dataset_debug(self, videos_clips_dict, batch_size, time_steps):
        normal_clips = videos_clips_dict['normal']
        normal_number_list = videos_clips_dict['normal_numbers']
        abnormal_clips = videos_clips_dict['abnormal']
        abnormal_number_list = videos_clips_dict['abnormal_numbers']

        num_normal_videos = len(normal_clips)
        num_abnormal_videos = len(abnormal_clips)

        height, width = self.resize_height, self.resize_width

        def _sample_video_clip(clips_list, length):
            clip = np.empty(shape=[time_steps, height, width, 3], dtype=np.float32)

            sample_idx = RNG.randint(length)
            for t, filename in enumerate(clips_list[sample_idx]):
                clip[t, ...] = load_frame(filename, height, width)

            return clip

        def video_clip_generator():
            while True:
                a_vid, p_vid = RNG.choice(num_normal_videos, size=2)
                n_vid = RNG.randint(num_abnormal_videos)

                video_clips = np.empty(shape=[3, time_steps, height, width, 3], dtype=np.float32)
                video_clips[0, ...] = _sample_video_clip(normal_clips[a_vid], normal_number_list[a_vid])
                video_clips[1, ...] = _sample_video_clip(normal_clips[p_vid], normal_number_list[p_vid])
                video_clips[2, ...] = _sample_video_clip(abnormal_clips[n_vid], abnormal_number_list[n_vid])

                # show_triplet_video_clips(video_clips)
                yield video_clips

        batch_video_clips = []
        for i in range(batch_size):
            batch_video_clips.append(next(video_clip_generator()))
        batch_video_clips = np.stack(batch_video_clips, axis=0)
        return batch_video_clips

    def _setup(self):
        videos_info = self.parse_videos_folder(self.folder)
        frame_mask = self.load_frame_mask(videos_info, self.frame_mask_file)
        pixel_mask_list = self.load_image_mask_file_list(videos_info, self.pixel_mask_file)

        num_videos = len(videos_info)
        if self.k_folds != 0:
            k_folds_ids = np.array_split(np.arange(num_videos), self.k_folds)
            val_ids = k_folds_ids[self.kth - 1].tolist()
            test_ids = []
            for k in range(self.k_folds):
                if k != (self.kth - 1):
                    test_ids += k_folds_ids[k].tolist()
        else:
            val_ids = []
            test_ids = np.arange(num_videos)

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
                    'pixel_mask': pixel_mask
                }

            return videos_info_gt

        val_videos_info = add_gt_to_info(val_ids)
        test_videos_info = add_gt_to_info(test_ids)

        del videos_info

        self.val_videos_info = val_videos_info
        self.test_videos_info = test_videos_info

    def read_video_clip(self, images_paths):
        video_clip = []
        for filename in images_paths:
            video_clip.append(load_frame(filename, self.resize_height, self.resize_width))

        video_clip = np.stack(video_clip, axis=0)
        return video_clip

    def get_video_clip(self, video, start, end, interval=1):
        # assert video_name in self._videos_info, 'video {} is not in {}!'.format(video_name, self._videos_info.keys())
        # assert (start >= 0) and (start <= end) and (end < self._videos_info[video_name]['length'])

        video_idx = np.arange(start, end, interval)
        video_clip = np.empty(shape=[len(video_idx), self.resize_height, self.resize_width, 3], dtype=np.float32)
        for idx, v_idx in enumerate(video_idx):
            filename = self.test_videos_info[video]['images'][v_idx]
            video_clip[idx, ...] = load_frame(filename, self.resize_height, self.resize_width)

        return video_clip

    def get_val_video_clip(self, video, start, end, interval=1):
        video_idx = np.arange(start, end, interval)
        video_clip = np.empty(shape=[len(video_idx), self.resize_height, self.resize_width, 3], dtype=np.float32)
        for idx, v_idx in enumerate(video_idx):
            filename = self.val_videos_info[video]['images'][v_idx]
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


class DataTemporalTripletLoader(DataTemporalGtLoader):
    def __init__(self, dataset, train_folder, test_folder, k_folds, kth,
                 frame_mask_file, pixel_mask_file='',
                 resize_height=256, resize_width=256):
        super().__init__(dataset, test_folder, resize_height, resize_width,
                         k_folds=k_folds, kth=kth, frame_mask_file=frame_mask_file, pixel_mask_file=pixel_mask_file)

        self.train_folder = train_folder
        self.train_videos_info = self.parse_videos_folder(self.train_folder)

    def __call__(self, batch_size, time_steps, interval=1, is_training=True):
        train_clips_dict = self.sample_normal_abnormal_clips(self.train_videos_info, time_steps, interval)
        val_clips_dict = self.sample_normal_abnormal_clips(self.val_videos_info, time_steps, interval)
        test_clips_dict = self.sample_normal_abnormal_clips(self.test_videos_info, time_steps, interval)

        train_val_clips_dict = {
            'normal': train_clips_dict['normal'] + val_clips_dict['normal'],
            'normal_numbers': train_clips_dict['normal_numbers'] + val_clips_dict['normal_numbers'],
            'abnormal': val_clips_dict['abnormal'],
            'abnormal_numbers': train_clips_dict['abnormal_numbers'] + val_clips_dict['abnormal_numbers']
        }

        if is_training:
            dataset = self.convert_to_tf_dataset(train_val_clips_dict, batch_size, time_steps)
        else:
            dataset = self.convert_to_tf_dataset(test_clips_dict, batch_size, time_steps)
        return dataset
