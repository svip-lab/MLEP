import os
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

from utils.util import load_frame, multi_scale_crop_load_frame
from utils.dataloaders import BaseDataAbstractLoader, RNG


class NormalDataLoader(BaseDataAbstractLoader):

    def __init__(self, dataset, folder, resize_height, resize_width):
        super().__init__(dataset=dataset, folder=folder, resize_height=resize_height, resize_width=resize_width)

        self.videos_info = None
        self._setup()

    def __call__(self, batch_size, time_steps, interval=1, multi_scale_crop=False):
        if multi_scale_crop:
            dataset = self.tf_multi_scale_crop_dataset(batch_size, time_steps, interval)
        else:
            dataset = self.tf_dataset(batch_size, time_steps, interval)
        return dataset

    def tf_multi_scale_crop_dataset(self, batch_size, time_steps, interval):
        videos_clips_list = self.sample_normal_clip_list(self.videos_info, time_steps, interval)
        num_video_clips = len(videos_clips_list)

        crop_size = 224
        short_size_list = [224, 256, 384, 480]

        height, width = self.resize_height, self.resize_width
        scale = width / height

        def video_clip_generator():
            i = 0
            while True:
                video_clips_paths = videos_clips_list[i]
                video_clips = np.empty(shape=[time_steps, crop_size, crop_size, 3], dtype=np.float32)

                resize_height = RNG.choice(short_size_list)
                resize_width = int(scale * resize_height)
                start_h = RNG.randint(0, resize_height - crop_size + 1)
                start_w = RNG.randint(0, resize_width - crop_size + 1)
                crop_bbox = ((start_h, start_h + crop_size), (start_w, start_w + crop_size))
                # print(crop_bbox, resize_height, resize_width)

                with ThreadPoolExecutor(max_workers=time_steps * 5) as pool:
                    for t_idx, frame in enumerate(pool.map(multi_scale_crop_load_frame, video_clips_paths,
                                                           [resize_height] * time_steps,
                                                           [resize_width] * time_steps,
                                                           [crop_bbox] * time_steps)):
                        video_clips[t_idx, ...] = frame

                i = (i + 1) % num_video_clips
                yield video_clips

        # video clip paths
        dataset = tf.data.Dataset.from_generator(generator=video_clip_generator,
                                                 output_types=tf.float32,
                                                 output_shapes=[time_steps, crop_size, crop_size, 3])
        print('generator dataset, {}'.format(dataset))
        dataset = dataset.prefetch(buffer_size=256)
        dataset = dataset.shuffle(buffer_size=256).batch(batch_size)
        print('epoch dataset, {}'.format(dataset))
        return dataset

    def tf_dataset(self, batch_size, time_steps, interval):
        videos_clips_list = self.sample_normal_clip_list(self.videos_info, time_steps, interval)
        num_video_clips = len(videos_clips_list)

        height, width = self.resize_height, self.resize_width

        def video_clip_generator():
            i = 0
            while True:
                video_clips_paths = videos_clips_list[i]
                video_clips = np.empty(shape=[time_steps, height, width, 3], dtype=np.float32)
                for t, filename in enumerate(video_clips_paths):
                    video_clips[t, ...] = load_frame(filename, height, width)

                i = (i + 1) % num_video_clips
                yield video_clips

        # video clip paths
        dataset = tf.data.Dataset.from_generator(generator=video_clip_generator,
                                                 output_types=tf.float32,
                                                 output_shapes=[time_steps, height, width, 3])
        print('generator dataset, {}'.format(dataset))
        dataset = dataset.prefetch(buffer_size=256)
        dataset = dataset.shuffle(buffer_size=256).batch(batch_size)
        print('epoch dataset, {}'.format(dataset))
        return dataset

    def debug(self, batch_size, time_steps, interval=1):
        videos_clips_list = self.sample_normal_clip_list(self.videos_info, time_steps, interval)
        num_video_clips = len(videos_clips_list)

        crop_size = 224
        short_size_list = [224, 256, 384, 480]

        height, width = self.resize_height, self.resize_width
        scale = width / height

        def video_clip_generator():
            i = 0
            while True:
                batch_video_clips = []
                for b_idx in range(batch_size):
                    video_clips_paths = videos_clips_list[i]
                    video_clips = np.empty(shape=[time_steps, crop_size, crop_size, 3], dtype=np.float32)

                    resize_height = RNG.choice(short_size_list)
                    resize_width = int(scale * resize_height)
                    start_h = RNG.randint(0, resize_height - crop_size + 1)
                    start_w = RNG.randint(0, resize_width - crop_size + 1)
                    crop_bbox = ((start_h, start_h + crop_size), (start_w, start_w + crop_size))
                    # print(crop_bbox, resize_height, resize_width)

                    with ThreadPoolExecutor(max_workers=time_steps * 5) as pool:
                        for t_idx, frame in enumerate(pool.map(multi_scale_crop_load_frame, video_clips_paths,
                                                               [resize_height] * time_steps,
                                                               [resize_width] * time_steps,
                                                               [crop_bbox] * time_steps)):
                            video_clips[t_idx, ...] = frame

                    i = (i + 1) % num_video_clips

                    batch_video_clips.append(video_clips)
                batch_video_clips = np.stack(batch_video_clips, axis=0)
                yield batch_video_clips

        return video_clip_generator

    def _setup(self):
        self.videos_info = self.parse_videos_folder(self.folder)

    def get_video_clip(self, video, start, end, interval=1):
        # assert video_name in self._videos_info, 'video {} is not in {}!'.format(video_name, self._videos_info.keys())
        # assert (start >= 0) and (start <= end) and (end < self._videos_info[video_name]['length'])

        video_idx = np.arange(start, end, interval)
        video_clip = np.empty(shape=[len(video_idx), self.resize_height, self.resize_width, 3], dtype=np.float32)
        for idx, v_idx in enumerate(video_idx):
            filename = self.videos_info[video]['images'][v_idx]
            video_clip[idx, ...] = load_frame(filename, self.resize_height, self.resize_width)

        return video_clip

    def get_video_names(self):
        return list(self.videos_info.keys())

