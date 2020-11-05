"""
This module define and implements all the dataloaders.
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.io as scio
from collections import OrderedDict
import os
import glob
import json
import imageio
import pickle

RNG = np.random.RandomState(2017)


class LazyProperty(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


class BaseDataAbstractLoader(object):
    __metaclass__ = ABCMeta

    def __init__(self, dataset, folder, resize_height, resize_width):
        self.dataset = dataset
        self.folder = folder
        self.resize_height = resize_height
        self.resize_width = resize_width

    @abstractmethod
    def get_video_clip(self, video, start, end, interval=1):
        pass

    @abstractmethod
    def get_video_names(self):
        pass

    @staticmethod
    def sample_normal_clip_list(videos_info, time_steps, interval=1):
        video_clips_list = []

        for video_name, video_info in videos_info.items():
            length = video_info['length']
            images_paths = video_info['images']

            for t in range(1, interval + 1):
                inv = t * time_steps

                for start in range(0, length):
                    end = start + inv
                    if end > length:
                        break

                    video_clips = images_paths[start:end:t]
                    video_clips_list.append(video_clips)

                    # flip sequence
                    video_clips_list.append(list(reversed(video_clips)))

                print('sample video {} at time {}.'.format(video_name, t))

        np.random.shuffle(video_clips_list)
        return video_clips_list

    @staticmethod
    def sample_normal_abnormal_clips(videos_info, time_steps, interval=1):
        normal_clips_list = []
        normal_number_list = []
        abnormal_clips_list = []
        abnormal_number_list = []

        # for shanghaitech and avenue which has been trained
        # at_least_anomalies = 2
        # at_least_anomalies = time_steps // 3 + 1
        at_least_anomalies = time_steps

        for v_name, v_info in videos_info.items():
            length = v_info['length']
            images_paths = v_info['images']
            frame_mask = v_info['frame_mask']

            video_normal_clips = []
            video_abnormal_clips = []

            for t in range(1, interval + 1):
                inv = t * time_steps

                for start in range(0, length - inv):
                    end = start + inv
                    video_clips = images_paths[start:end:t]
                    reversed_video_clips = list(reversed(video_clips))

                    # check is normal or abnormal
                    if len(frame_mask) != 0 and np.count_nonzero(frame_mask[start:end:t]) >= at_least_anomalies:
                        video_abnormal_clips.append(video_clips)
                        video_abnormal_clips.append(reversed_video_clips)
                    else:
                        video_normal_clips.append(video_clips)
                        video_normal_clips.append(reversed_video_clips)

                print('sample video {} at time {}.'.format(v_name, t))

            if len(video_normal_clips) != 0:
                normal_clips_list.append(video_normal_clips)
                normal_number_list.append(len(video_normal_clips))
            if len(video_abnormal_clips) != 0:
                abnormal_clips_list.append(video_abnormal_clips)
                abnormal_number_list.append(len(video_abnormal_clips))

        video_clips_dict = {
            'normal': normal_clips_list,
            'normal_numbers': normal_number_list,
            'abnormal': abnormal_clips_list,
            'abnormal_numbers': abnormal_number_list
        }
        return video_clips_dict

    @staticmethod
    def sample_normal_abnormal_clips_scores(videos_info, time_steps, interval=1, max_scores=0.5, min_scores=0.5):
        normal_clips_list = []
        normal_scores_list = []
        normal_number_list = []

        abnormal_clips_list = []
        abnormal_scores_list = []
        abnormal_number_list = []

        at_least_anomalies = min_scores * time_steps
        at_most_anomalies = max_scores * time_steps

        for v_name, v_info in videos_info.items():
            length = v_info['length']
            images_paths = v_info['images']

            if 'scores' in v_info:
                scores = v_info['scores']
            else:
                scores = np.ones(shape=(length,), dtype=np.float32)

            video_normal_clips = []
            video_normal_scores = []
            video_abnormal_clips = []
            video_abnormal_scores = []

            for t in range(1, interval + 1):
                inv = t * time_steps

                for start in range(0, length - inv):
                    end = start + inv
                    video_clips = images_paths[start:end:t]
                    reversed_video_clips = list(reversed(video_clips))

                    video_scores = scores[start:end:t]
                    reversed_video_scores = list(reversed(video_scores))

                    # check is normal or abnormal
                    total_scores = np.sum(scores[start:end:t])
                    if total_scores < at_least_anomalies:
                        video_abnormal_clips.append(video_clips)
                        video_abnormal_clips.append(reversed_video_clips)

                        video_abnormal_scores.append(video_scores)
                        video_abnormal_scores.append(reversed_video_scores)

                    elif total_scores >= at_most_anomalies:
                        video_normal_clips.append(video_clips)
                        video_normal_clips.append(reversed_video_clips)

                        video_normal_scores.append(video_scores)
                        video_normal_scores.append(reversed_video_scores)

                print('sample video {} at time {}.'.format(v_name, t))

            if len(video_normal_clips) != 0:
                normal_clips_list.append(video_normal_clips)
                normal_number_list.append(len(video_normal_clips))
                normal_scores_list.append(video_normal_scores)
            if len(video_abnormal_clips) != 0:
                abnormal_clips_list.append(video_abnormal_clips)
                abnormal_scores_list.append(video_abnormal_scores)
                abnormal_number_list.append(len(video_abnormal_clips))

        video_clips_dict = {
            'normal': normal_clips_list,
            'normal_scores': normal_scores_list,
            'normal_numbers': normal_number_list,
            'abnormal': abnormal_clips_list,
            'abnormal_scores': abnormal_scores_list,
            'abnormal_numbers': abnormal_number_list
        }
        return video_clips_dict

    @staticmethod
    def sample_normal_abnormal_clips_masks(videos_info, time_steps, interval=1):
        normal_clips_list = []
        normal_pixels_list = []
        normal_number_list = []

        abnormal_clips_list = []
        abnormal_pixels_list = []
        abnormal_number_list = []

        at_least_anomalies = time_steps

        for v_name, v_info in videos_info.items():
            length = v_info['length']
            images_paths = v_info['images']
            frame_mask = v_info['frame_mask']
            pixel_mask = v_info['pixel_mask']

            has_abnormal = len(frame_mask) > 0

            video_normal_clips = []
            video_normal_pixels = []
            video_abnormal_clips = []
            video_abnormal_pixels = []

            for t in range(1, interval + 1):
                inv = t * time_steps

                for start in range(0, length - inv):
                    end = start + inv
                    video_clips = images_paths[start:end:t]
                    reversed_video_clips = list(reversed(video_clips))

                    video_pixels = pixel_mask[start:end:t]
                    reversed_video_pixels = list(reversed(video_pixels))

                    # check is normal or abnormal
                    if has_abnormal and np.sum(frame_mask[start:end:t]) >= at_least_anomalies:
                        video_abnormal_clips.append(video_clips)
                        video_abnormal_clips.append(reversed_video_clips)

                        video_abnormal_pixels.append(video_pixels)
                        video_abnormal_pixels.append(reversed_video_pixels)

                    else:
                        video_normal_clips.append(video_clips)
                        video_normal_clips.append(reversed_video_clips)

                        video_normal_pixels.append(video_pixels)
                        video_normal_pixels.append(reversed_video_pixels)

                print('sample video {} at time {}.'.format(v_name, t))

            if len(video_normal_clips) != 0:
                normal_clips_list.append(video_normal_clips)
                normal_pixels_list.append(video_normal_pixels)
                normal_number_list.append(len(video_normal_clips))
            if len(video_abnormal_clips) != 0:
                abnormal_clips_list.append(video_abnormal_clips)
                abnormal_pixels_list.append(video_abnormal_pixels)
                abnormal_number_list.append(len(video_abnormal_clips))

        video_clips_dict = {
            'normal': normal_clips_list,
            'normal_pixels': normal_pixels_list,
            'normal_numbers': normal_number_list,
            'abnormal': abnormal_clips_list,
            'abnormal_pixels': abnormal_pixels_list,
            'abnormal_numbers': abnormal_number_list
        }
        return video_clips_dict

    @staticmethod
    def sample_fragments_seq_num(videos_info, time_steps, interval=1, fragments=20):
        fragment_clip_list = []
        video_path_list = []

        for v_info in videos_info:
            length = v_info['length']
            video_path = v_info['path']

            fragment_clip = []
            fragment_size = int(np.ceil(length / fragments))
            for frag_start in range(0, length, fragment_size):
                frag_end = min(frag_start + fragment_size, length)

                itv_fragment = []
                for t in range(1, interval + 1):
                    inv = t * time_steps
                    if inv > frag_end - frag_start:
                        break
                    start = RNG.randint(frag_start, frag_end - inv)

                    clip = list(range(start, start + inv, t))
                    reversed_clip = list(reversed(clip))

                    itv_fragment.append(clip)
                    itv_fragment.append(reversed_clip)
                if len(itv_fragment) > 0:
                    fragment_clip.append(itv_fragment)

            if len(fragment_clip) == fragments:
                fragment_clip_list.append(fragment_clip)
                video_path_list.append(video_path)

        return fragment_clip_list, video_path_list

    @staticmethod
    def sample_fragments_seq_path(videos_info, time_steps, interval=1, multi_interval=False, fragments=20):
        if multi_interval:
            interval_list = range(1, interval + 1)
        else:
            interval_list = range(interval, interval + 1)

        fragment_clip_list = []

        for v_info in videos_info:
            length = v_info['length']
            images_paths = v_info['images']

            fragment_clip = []
            fragment_size = int(np.ceil(length / fragments))
            for frag_start in range(0, length, fragment_size):
                frag_end = min(frag_start + fragment_size, length)

                itv_fragment = []

                for t in interval_list:
                    inv = t * time_steps
                    if inv > frag_end - frag_start:
                        break
                    for start in range(frag_start, frag_end - inv):
                        clip = images_paths[start: start + inv: t]
                        reversed_clip = list(reversed(clip))

                        itv_fragment.append(clip)
                        itv_fragment.append(reversed_clip)
                if len(itv_fragment) > 0:
                    fragment_clip.append(itv_fragment)

            if len(fragment_clip) == fragments:
                fragment_clip_list.append(fragment_clip)

        return fragment_clip_list

    @staticmethod
    def parse_videos_folder(folder):
        print('parsing video folder = {}'.format(folder))

        videos_info = OrderedDict()
        for video_name in sorted(os.listdir(folder)):
            images_paths = glob.glob(os.path.join(folder, video_name, '*'))
            images_paths.sort()
            length = len(images_paths)

            videos_info[video_name] = {
                'length': length,
                'images': images_paths,
                'frame_mask': [],
                'pixel_mask': []
            }

        print('parsing video successfully...')
        return videos_info

    @staticmethod
    def parser_videos_images_txt_split_classes(folder, txt_file, frame_mask_folder=''):
        print('parsing video txt = {}'.format(txt_file))

        videos_info = OrderedDict()
        with open(txt_file, 'r') as reader:
            def add_to_videos_info(video, video_class, video_path):
                # video_folder/Abuse/Abuse001_x264
                assert os.path.exists(video_path), 'video = {} dose not exist!'.format(video_path)

                # check and load temporal annotation
                if frame_mask_folder:
                    # frame_folder/Abuse001_x264
                    frame_mat = os.path.join(frame_mask_folder, os.path.split(video_path)[-1] + '.mat')
                    load_frame = scio.loadmat(frame_mat)
                    frame_mask = load_frame['Annotation_file']['Anno'][0][0]
                else:
                    frame_mask = []

                images_paths = glob.glob(os.path.join(video_path, '*'))
                images_paths.sort()
                length = len(images_paths)

                info = {
                    'length': length,
                    'images': images_paths,
                    'frame_mask': frame_mask
                }

                if 'normal' in video_class.lower():
                    class_label = 'Normal'
                else:
                    class_label = video_class

                if class_label in videos_info:
                    videos_info[class_label][video] = info
                else:
                    videos_info[class_label] = {video: info}

            for line in reader:
                line = line.strip()

                # Abuse/Abuse001_x264.mp4
                splits = line.split('/')
                video_class = str(splits[0])

                # video_folder/Abuse/Abuse001_x264
                video_path = os.path.join(folder, video_class, str(splits[-1].split('.')[0]))
                add_to_videos_info(line, video_class, video_path)
                print(txt_file, line)

        return videos_info

    @staticmethod
    def parser_videos_images_txt(folder, txt_file, frame_mask_folder=''):
        print('parsing video txt = {}'.format(txt_file))

        videos_info = OrderedDict()
        with open(txt_file, 'r') as reader:
            def add_to_videos_info(video, video_class, video_path):
                # video_folder/Abuse/Abuse001_x264
                assert os.path.exists(video_path), 'video = {} dose not exist!'.format(video_path)

                # check and load temporal annotation
                if frame_mask_folder:

                    # frame_folder/Abuse001_x264
                    frame_mat = os.path.join(frame_mask_folder, os.path.split(video_path)[-1] + '.mat')
                    load_frame = scio.loadmat(frame_mat)
                    frame_mask = load_frame['Annotation_file']['Anno'][0][0]
                else:
                    frame_mask = []

                images_paths = glob.glob(os.path.join(video_path, '*'))
                images_paths.sort()
                length = len(images_paths)

                info = {
                    'length': length,
                    'images': images_paths,
                    'frame_mask': frame_mask
                }
                videos_info[video] = info

            for line in reader:
                line = line.strip()

                # Abuse/Abuse001_x264.mp4
                splits = line.split('/')
                video_class = str(splits[0])

                # video_folder/Abuse/Abuse001_x264
                video_path = os.path.join(folder, video_class, str(splits[-1].split('.')[0]))
                add_to_videos_info(line, video_class, video_path)
                print(txt_file, line)

        return videos_info

    @staticmethod
    def parser_videos_images_json(folder, frame_mask_file=''):
        print('parsing video json = {}'.format(frame_mask_file))

        videos_info = OrderedDict()
        with open(frame_mask_file, 'r') as file:
            data = json.load(file)

            for video_name in sorted(os.listdir(folder)):
                images_paths = glob.glob(os.path.join(folder, video_name, '*'))
                images_paths.sort()
                length = len(images_paths)

                assert length == data[video_name]['length']
                anomalies = data[video_name]['anomalies']

                frame_mask = []
                for event in anomalies:
                    for name, annotation in event.items():
                        frame_mask.append(annotation)

                videos_info[video_name] = {
                    'length': length,
                    'images': images_paths,
                    'frame_mask': frame_mask,
                    'pixel_mask': []
                }

        print('parsing video successfully...')
        return videos_info

    @staticmethod
    def parser_videos_paths_txt(folder, txt_file, frame_mask_file=''):
        print('parsing video txt = {}'.format(txt_file))

        videos_info = OrderedDict()
        with open(txt_file, 'r') as reader:
            def add_to_videos_info(video_class, video_path):
                assert os.path.exists(video_path), 'video = {} dose not exist!'.format(video_path)

                # get length
                vid = imageio.get_reader(video_path, 'ffmpeg')
                video_length = vid.get_length()

                # check and load temporal annotation
                if frame_mask_file:
                    frame_mat = os.path.join(frame_mask_file, os.path.split(video_path)[-1].split('.')[0] + '.mat')
                    load_frame = scio.loadmat(frame_mat)
                    frame_mask = load_frame['Annotation_file']['Anno'][0][0]
                else:
                    frame_mask = []

                info = {
                    'length': video_length,
                    'path': video_path,
                    'frame_mask': frame_mask
                }

                if 'normal' in video_class.lower():
                    class_label = 'Normal'
                else:
                    class_label = video_class

                if class_label in videos_info:
                    videos_info[class_label].append(info)
                else:
                    videos_info[class_label] = [info]
                vid.close()

            for line in reader:
                line = line.strip()

                # Abuse/Abuse001_x264.mp4
                video_class = str(line.split('/')[0])
                video_path = os.path.join(folder, line)

                add_to_videos_info(video_class, video_path)
                print(txt_file, line)

        return videos_info

    @staticmethod
    def parser_paths_txt(folder, txt_file):
        print('parsing video txt = {}'.format(txt_file))

        videos_info = {}
        with open(txt_file, 'r') as reader:
            def add_to_videos_info(video_class, video_path):
                assert os.path.exists(video_path), 'video = {} dose not exist!'.format(video_path)
                if 'normal' in video_class.lower():
                    class_label = 'Normal'
                else:
                    class_label = video_class

                if class_label in videos_info:
                    videos_info[class_label].append(video_path)
                else:
                    videos_info[class_label] = [video_path]

            for line in reader:
                line = line.strip()

                # Abuse/Abuse001_x264.mp4
                video_class = str(line.split('/')[0])
                video_path = os.path.join(folder, line)

                add_to_videos_info(video_class, video_path)
                print(txt_file, line)

        return videos_info

    @staticmethod
    def load_frame_scores(psnr_file):
        scores = []
        with open(psnr_file, 'rb') as reader:
            # results {
            #   'dataset': the name of dataset
            #   'psnr': the psnr of each testing videos,
            # }

            # psnr_records['psnr'] is np.array, shape(#videos)
            # psnr_records[0] is np.array   ------>     01.avi
            # psnr_records[1] is np.array   ------>     02.avi
            #               ......
            # psnr_records[n] is np.array   ------>     xx.avi

            results = pickle.load(reader)
            psnr_records = results['psnr']

            for psnr in psnr_records:
                score = (psnr - psnr.min()) / (psnr.max() - psnr.min())
                scores.append(score)

        return scores

    @staticmethod
    def load_frame_psnrs(psnr_file, threshold=None):
        psnrs_records = []
        with open(psnr_file, 'rb') as reader:

            results = pickle.load(reader)
            psnr_records = results['psnr']

            for i, psnrs in enumerate(psnr_records):
                if threshold:
                    invalid_index = np.logical_or(np.isnan(psnrs), np.isinf(psnrs))
                    psnrs[invalid_index] = threshold + 1

                    too_big_index = np.logical_or(invalid_index, psnrs > threshold)
                    not_too_big_index = np.logical_not(too_big_index)

                    psnr_max = np.max(psnrs[not_too_big_index])
                    psnrs[too_big_index] = psnr_max

                psnrs_records.append(psnrs)
        return psnrs_records

    @staticmethod
    def filter_caption_psnrs(x):
        length = len(x)
        # p = np.zeros(shape=(length,), dtype=np.float32)

        x_mean = np.max(x)
        mid_idx = length // 2
        # p[mid_idx] = x[mid_idx]
        delta = length - mid_idx
        for i in range(mid_idx + 1, length):
            alpha = (i - mid_idx) / delta
            x[i] = alpha * x_mean + (1 - alpha) * x[i]

        for i in range(mid_idx - 1, -1, -1):
            alpha = (mid_idx - i) / delta
            x[i] = alpha * x_mean + (1 - alpha) * x[i]

        return x

    def load_frame_mask(self, videos_info, gt_file_path):
        # initialize the load frame mask function
        if self.dataset in ['ped1', 'ped2', 'avenue', 'enter', 'exit']:
            if gt_file_path.endswith('.json'):
                frame_mask = self._load_json_gt_file(gt_file_path)
            else:
                frame_mask = self._load_ucsd_avenue_subway_gt(videos_info, gt_file_path)
        elif self.dataset == 'shanghaitech':
            frame_mask = self._load_shanghaitech_gt(gt_file_path)
        else:
            print('Warning, dataset {} is not in {}, be careful when loading the frame mask and '
                  'here, we use _load_uscd_avenue_subway_gt()'.format(self.dataset,
                                                                      ['ped1', 'ped2', 'avenue', 'enter', 'exit',
                                                                       'shanghaitech']))
            frame_mask = self._load_ucsd_avenue_subway_gt(videos_info, gt_file_path)

        return frame_mask

    @staticmethod
    def load_semantic_frame_mask(json_file):
        semantic_gts = []
        anomalies_names = []
        anomalies_names_set = set()

        with open(json_file, 'r') as file:
            data = json.load(file)
            video_names = list(sorted(data.keys()))

            for video_name in video_names:
                info = data[video_name]
                anomalies = info['anomalies']
                length = info['length']
                semantic_label = ['normal'] * length
                name_set = set()

                for event in anomalies:
                    for name, annotation in event.items():
                        for start, end in annotation:
                            semantic_label[start:end] = [name] * (end - start)

                        name_set.add(name)

                semantic_gts.append(semantic_label)
                anomalies_names.append(name_set)
                anomalies_names_set |= name_set

        print('gt json file = {}'.format(json_file))
        return semantic_gts, anomalies_names, anomalies_names_set

    @staticmethod
    def _load_json_gt_file(json_file):
        gts = []
        with open(json_file, 'r') as file:
            data = json.load(file)
            video_names = list(sorted(data.keys()))

            for video_name in video_names:
                info = data[video_name]
                anomalies = info['anomalies']
                length = info['length']
                label = np.zeros((length,), dtype=np.int8)

                for event in anomalies:
                    for name, annotation in event.items():
                        for start, end in annotation:
                            label[start - 1: end] = 1
                gts.append(label)

        print('gt json file = {}'.format(json_file))
        return gts

    @staticmethod
    def _load_ucsd_avenue_subway_gt(videos_info, gt_file_path):
        """
        :param videos_info: videos information, parsed by parse_videos_folder
        :param gt_file_path: the path of gt file
        :type videos_info: dict or OrderedDict
        :type gt_file_path: str
        :return:
        """
        assert os.path.exists(gt_file_path), 'gt file path = {} dose not exits!'.format(gt_file_path)

        abnormal_events = scio.loadmat(gt_file_path, squeeze_me=True)['gt']

        if abnormal_events.ndim == 2:
            abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0], abnormal_events.shape[1])

        num_video = abnormal_events.shape[0]
        assert num_video == len(videos_info), 'ground true does not match the number of testing videos. {} != {}' \
            .format(num_video, len(videos_info))

        # need to test [].append, or np.array().append(), which one is faster
        gt = []
        for i, video_info in enumerate(videos_info.values()):
            length = video_info['length']

            sub_video_gt = np.zeros((length,), dtype=np.int8)
            sub_abnormal_events = abnormal_events[i]
            if sub_abnormal_events.ndim == 1:
                sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))

            _, num_abnormal = sub_abnormal_events.shape

            for j in range(num_abnormal):
                # (start - 1, end - 1)
                start = sub_abnormal_events[0, j] - 1
                end = sub_abnormal_events[1, j]

                sub_video_gt[start: end] = 1

            gt.append(sub_video_gt)

        return gt

    @staticmethod
    def _load_shanghaitech_gt(gt_file_folder):
        """
        :param gt_file_folder: the folder path of test_frame_mask of ShanghaiTech dataset.
        :type gt_file_folder: str
        :return:
        """
        video_path_list = os.listdir(gt_file_folder)
        video_path_list.sort()

        gt = []
        for video in video_path_list:
            gt.append(np.load(os.path.join(gt_file_folder, video)))

        return gt

    @staticmethod
    def load_pixel_mask_file_list(videos_info, pixel_mask_folder=''):
        """
        :param videos_info: videos information, parsed by parse_videos_folder
        :param pixel_mask_folder: the path of pixel mask folder
        :type videos_info: dict or OrderedDict
        :type pixel_mask_folder: str
        :return:
        """

        if pixel_mask_folder:
            pixel_mask_file_list = os.listdir(pixel_mask_folder)
            pixel_mask_file_list.sort()

            num_videos = len(videos_info)
            assert num_videos == len(pixel_mask_file_list), \
                'ground true does not match the number of testing videos. {} != {}'.format(
                    num_videos, len(pixel_mask_file_list))

            for video_name, pixel_mask_file in zip(videos_info.keys(), pixel_mask_file_list):
                assert video_name + '.npy' == pixel_mask_file, 'video name {} does not have pixel mask {}'.format(
                    video_name, pixel_mask_file
                )

            for i in range(num_videos):
                pixel_mask_file_list[i] = os.path.join(pixel_mask_folder, pixel_mask_file_list[i])
        else:
            pixel_mask_file_list = []

        return pixel_mask_file_list

    @staticmethod
    def load_image_mask_file_list(videos_info, pixel_mask_folder=''):
        """
        :param videos_info: videos information, parsed by parse_videos_folder
        :param pixel_mask_folder: the path of pixel mask folder
        :type videos_info: dict or OrderedDict
        :type pixel_mask_folder: str
        :return:
        """

        if pixel_mask_folder:
            pixel_mask_file_list = os.listdir(pixel_mask_folder)
            pixel_mask_file_list.sort()

            num_videos = len(videos_info)
            assert num_videos == len(pixel_mask_file_list), \
                'ground true does not match the number of testing videos. {} != {}'.format(
                    num_videos, len(pixel_mask_file_list))

            for video_name, pixel_mask_file in zip(videos_info.keys(), pixel_mask_file_list):
                assert video_name == pixel_mask_file, 'video name {} does not have pixel mask {}'.format(
                    video_name, pixel_mask_file
                )

            image_mask_files = []
            for i in range(num_videos):
                mask_files = glob.glob(os.path.join(pixel_mask_folder, pixel_mask_file_list[i], '*.jpg'))
                mask_files.sort()
                image_mask_files.append(mask_files)
        else:
            image_mask_files = []

        return image_mask_files
