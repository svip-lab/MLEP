import numpy as np
import os
import argparse
import pickle
from sklearn import metrics
from scipy import interpolate
from math import factorial
import json
import re

from constant import const


NORMALIZE = const.NORMALIZE
DECIDABLE_IDX = 0
THRESHOLD = 200


class RecordResult(object):
    def __init__(self, fpr=None, tpr=None, auc=-np.inf, dataset=None, loss_file=None):
        self.fpr = fpr
        self.tpr = tpr
        self.auc = auc
        self.dataset = dataset
        self.loss_file = loss_file

    def __lt__(self, other):
        return self.auc < other.auc

    def __gt__(self, other):
        return self.auc > other.auc

    def __str__(self):
        return 'dataset = {}, loss file = {}, auc = {}'.format(self.dataset, self.loss_file, self.auc)


def temporal_annotation_to_label(annotations, length):
    label = np.zeros((length,), dtype=np.int8)

    for start, end in annotations:
        label[start - 1:end] = 1
    return label


def load_video_scene(video_scene_file, k_folds, kth):
    scenes_videos_list = []
    with open(video_scene_file, 'r') as reader:
        video_scenes_list = list(reader.readlines())
        num_videos = len(video_scenes_list)

        if k_folds > 0:
            k_folds_ids = np.array_split(np.arange(num_videos), k_folds)
            test_ids = []
            for k in range(k_folds):
                if k != (kth - 1):
                    test_ids += k_folds_ids[k].tolist()
        else:
            test_ids = np.arange(num_videos)

        for ids in test_ids:
            video_scene = video_scenes_list[ids]
            video_scene = video_scene.rstrip()
            video, scene = video_scene.split()
            scenes_videos_list.append((video, scene))

    return scenes_videos_list


def load_gt_by_json(json_file, k_folds, kth):
    gts = []
    with open(json_file, 'r') as file:
        data = json.load(file)
        video_names = list(sorted(data.keys()))

        num_videos = len(video_names)

        if k_folds > 0:
            k_folds_ids = np.array_split(np.arange(num_videos), k_folds)
            test_ids = []
            for k in range(k_folds):
                if k != (kth - 1):
                    test_ids += k_folds_ids[k].tolist()
        else:
            test_ids = np.arange(num_videos)

        for ids in test_ids:
            info = data[video_names[ids]]
            anomalies = info['anomalies']
            length = info['length']
            label = np.zeros((length,), dtype=np.int8)

            for event in anomalies:
                for name, annotation in event.items():
                    for start, end in annotation:
                        label[start-1: end] = 1
            gts.append(label)

    return gts


def load_semantic_frame_mask(json_file, k_folds, kth):
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

        num_videos = len(data.keys())
        if k_folds != 0:
            k_folds_ids = np.array_split(np.arange(num_videos), k_folds)
            val_ids = k_folds_ids[kth - 1].tolist()
            test_ids = []
            for k in range(k_folds):
                if k != (kth - 1):
                    test_ids += k_folds_ids[k].tolist()
        else:
            val_ids = []
            test_ids = np.arange(num_videos)

        # seen anomalies
        seen_anomalies = set()
        for v in val_ids:
            seen_anomalies |= anomalies_names[v]

        gts = []
        for v in test_ids:
            gt = np.zeros(len(semantic_gts[v]))
            for t, label in enumerate(semantic_gts[v]):
                if label == 'normal':
                    continue

                if label in seen_anomalies:
                    gt[t] = 1
                else:
                    gt[t] = 2
            gts.append(gt)

    return gts


def load_psnr_gt(loss_file):
    with open(loss_file, 'rb') as reader:
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

    dataset = results['dataset']
    psnr_records = results['psnr']
    gts = results['frame_mask']

    return dataset, psnr_records, gts


def load_features_gt(loss_file):
    with open(loss_file, 'rb') as reader:
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

    dataset = results['dataset']
    gts = results['frame_mask']
    features = results['visualize']['feature']

    f_means = []
    for i, f in enumerate(features):
        length = len(gts[i])
        sub_len = len(f)
        vol_size = int(np.ceil(length / sub_len))
        f_m = np.mean(f, axis=(1, 2, 3))
        # interpretation
        x_ids = np.arange(0, length, vol_size)
        x_ids[-1] = length - 1
        print(len(x_ids), sub_len)
        inter_func = interpolate.interp1d(x_ids, f_m)
        ids = np.arange(0, length)
        f_means.append(inter_func(ids))

    return dataset, f_means, gts


def load_psnr(loss_file):
    """
    load image psnr or optical flow psnr.
    :param loss_file: loss file path
    :return:
    """
    with open(loss_file, 'rb') as reader:
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
    psnrs = results['psnr']
    return psnrs


def get_scores_labels(loss_file):
    # the name of dataset, loss, and ground truth
    dataset, psnr_records, gt = load_psnr_gt(loss_file=loss_file)

    # the number of videos
    num_videos = len(psnr_records)

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    # video normalization
    for i in range(num_videos):
        distance = psnr_records[i]

        if NORMALIZE:
            distance -= distance.min()  # distances = (distance - min) / (max - min)
            distance /= distance.max()
            # distance = 1 - distance

        scores = np.concatenate((scores[:], distance[DECIDABLE_IDX:]), axis=0)
        labels = np.concatenate((labels[:], gt[i][DECIDABLE_IDX:]), axis=0)
    return dataset, scores, labels


def precision_recall_auc(loss_file, *args):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        dataset, scores, labels = get_scores_labels(sub_loss_file)
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=0)
        auc = metrics.auc(recall, precision)

        results = RecordResult(recall, precision, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model = {}'.format(optimal_results))
    return optimal_results


def cal_eer(fpr, tpr):
    # makes fpr + tpr = 1
    eer = fpr[np.nanargmin(np.absolute((fpr + tpr - 1)))]
    return eer


def compute_eer(loss_file, *args):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult(auc=np.inf)
    for sub_loss_file in loss_file_list:
        dataset, scores, labels = get_scores_labels(sub_loss_file)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        eer = cal_eer(fpr, tpr)

        results = RecordResult(fpr, tpr, eer, dataset, sub_loss_file)

        if optimal_results > results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model = {}'.format(optimal_results))
    return optimal_results


def compute_auc(loss_file, *args):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        # the name of dataset, loss, and ground truth
        dataset, psnr_records, gt = load_psnr_gt(loss_file=sub_loss_file)

        # the number of videos
        num_videos = len(psnr_records)

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        # video normalization
        for i in range(num_videos):
            distance = psnr_records[i]

            if NORMALIZE:
                distance -= distance.min()  # distances = (distance - min) / (max - min)
                distance /= distance.max()
                # distance -= np.mean(distance)
                # distance /= np.std(distance)
                # print(distance.max(), distance.min())

            scores = np.concatenate((scores, distance[DECIDABLE_IDX:]), axis=0)
            labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        auc = metrics.auc(fpr, tpr)

        results = RecordResult(fpr, tpr, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        # if os.path.isdir(loss_file):
        #     print(results)
    print('##### optimal result and model = {}'.format(optimal_results))
    return optimal_results


def calculate_auc(psnrs_records, gts):
    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    for psnrs, gt in zip(psnrs_records, gts):
        # invalid_index = np.logical_or(np.isnan(psnrs), np.isinf(psnrs))
        # psnrs[invalid_index] = THRESHOLD + 1
        #
        # too_big_index = np.logical_or(invalid_index, psnrs > THRESHOLD)
        # not_too_big_index = np.logical_not(too_big_index)
        #
        # psnr_min = np.min(psnrs[not_too_big_index])
        # psnr_max = np.max(psnrs[not_too_big_index])
        #
        # psnrs[too_big_index] = psnr_max

        score = (psnrs - psnrs.min()) / (psnrs.max() - psnrs.min())
        scores = np.concatenate((scores, score))
        labels = np.concatenate((labels, gt))

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    # print('auc = {}'.format(auc))


def compute_valid_auc(loss_file, *args):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        # the name of dataset, loss, and ground truth
        dataset, psnr_records, gt = load_psnr_gt(loss_file=sub_loss_file)

        # the number of videos
        num_videos = len(psnr_records)

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        # video normalization
        for i in range(num_videos):
            psnrs = psnr_records[i]
            invalid_index = np.logical_or(np.isnan(psnrs), np.isinf(psnrs))
            psnrs[invalid_index] = THRESHOLD + 1

            too_big_index = np.logical_or(invalid_index, psnrs > THRESHOLD)
            not_too_big_index = np.logical_not(too_big_index)

            psnr_min = np.min(psnrs[not_too_big_index])
            psnr_max = np.max(psnrs[not_too_big_index])

            psnrs[too_big_index] = psnr_max
            psnrs = filter_psnrs(psnrs)

            if NORMALIZE:
                psnrs = (psnrs - psnr_min) / (psnr_max - psnr_min)  # distances = (distance - min) / (max - min)
                # distance -= np.mean(distance)
                # distance /= np.std(distance)
                # print(distance.max(), distance.min())

            scores = np.concatenate((scores, psnrs[DECIDABLE_IDX:]), axis=0)
            labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        auc = metrics.auc(fpr, tpr)

        results = RecordResult(fpr, tpr, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model = {}'.format(optimal_results))
    return optimal_results


def filter_psnrs(x):
    length = len(x)
    p = np.zeros(shape=(length,), dtype=np.float32)

    x_mean = np.max(x)
    mid_idx = length // 2
    p[mid_idx] = x[mid_idx]
    delta = length - mid_idx
    for i in range(mid_idx + 1, length):
        alpha = (i - mid_idx) / delta
        p[i] = alpha * x_mean + (1 - alpha) * x[i]

    for i in range(mid_idx - 1, -1,  -1):
        alpha = (mid_idx - i) / delta
        p[i] = alpha * x_mean + (1 - alpha) * x[i]

    return p


def filter_window(x, length):
    mid_idx = length // 2
    max_psnr = np.max(x)
    # max_psnr = np.mean(x)

    p = np.zeros(shape=x.shape, dtype=np.float32)
    p[mid_idx] = x[mid_idx]

    delta = length - mid_idx
    for i in range(mid_idx + 1, length):
        alpha = (i - mid_idx) / delta
        p[i] = alpha * max_psnr + (1 - alpha) * x[i]

    for i in range(mid_idx - 1, -1,  -1):
        alpha = (mid_idx - i) / delta
        p[i] = alpha * max_psnr + (1 - alpha) * x[i]

    return p


def filter_psnrs_2(x, window_size=128):
    length = len(x)

    window_size = min(length, window_size)
    p = np.empty(shape=(length,), dtype=np.float32)
    w_num = int(np.ceil(length / window_size))

    for w in range(w_num):
        start = w * window_size
        end = min((w + 1) * window_size, length)
        p[start:end] = filter_window(x[start:end], length=end-start)

    return p


def filter_psnrs_3(x, window_size=128):
    length = len(x)

    window_size = min(length, window_size)
    p = np.empty(shape=(length,), dtype=np.float32)
    w_num = int(np.ceil(length / window_size))

    for w in range(w_num):
        start = w * window_size
        end = min((w + 1) * window_size, length)
        p[start:end] = filter_window(x[start:end], length=end-start)
        p[start:end] = (p[start:end] - p[start:end].min()) / (p[start:end].max() - p[start:end].min())

    return p


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        print("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def compute_filter_auc(loss_file, *args):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        # the name of dataset, loss, and ground truth
        dataset, psnr_records, gt = load_psnr_gt(loss_file=sub_loss_file)

        # the number of videos
        num_videos = len(psnr_records)

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        # video normalization
        for i in range(num_videos):
            psnrs = psnr_records[i]
            invalid_index = np.logical_or(np.isnan(psnrs), np.isinf(psnrs))
            psnrs[invalid_index] = THRESHOLD + 1

            too_big_index = np.logical_or(invalid_index, psnrs > THRESHOLD)
            not_too_big_index = np.logical_not(too_big_index)

            psnr_max = np.max(psnrs[not_too_big_index])

            psnrs[too_big_index] = psnr_max
            psnrs = filter_psnrs(psnrs)

            # avenue
            # psnrs = filter_psnrs_3(psnrs, window_size=25)

            # shanghaitech
            # psnrs = filter_psnrs_2(psnrs, window_size=500)

            psnr_min = np.min(psnrs)
            psnr_max = np.max(psnrs)

            if NORMALIZE:
                psnrs = (psnrs - psnr_min) / (psnr_max - psnr_min)  # distances = (distance - min) / (max - min)

            scores = np.concatenate((scores, psnrs[DECIDABLE_IDX:]), axis=0)
            labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        auc = metrics.auc(fpr, tpr)

        results = RecordResult(fpr, tpr, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model = {}'.format(optimal_results))
    return optimal_results


def smooth_psnrs(x):
    length = x.shape[0]
    for i in range(1, length):
        x[i] = 0.99 * x[i-1] + 0.01 * x[i]
    return x


def compute_scene_auc(loss_file, *args):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    video_scene_file = args[0]
    pattern = re.compile('folds_([0-9]+)_kth_([0-9]+)')
    folds, kth = pattern.findall(loss_file)[0]
    folds, kth = int(folds), int(kth)

    video_scene_list = load_video_scene(video_scene_file, folds, kth)

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        # the name of dataset, loss, and ground truth
        dataset, psnr_records, gt = load_psnr_gt(loss_file=sub_loss_file)

        # the number of videos
        num_videos = len(psnr_records)

        scene_psnrs_dict = {}
        scene_labels_dict = {}
        # video normalization
        for i in range(num_videos):
            psnrs = psnr_records[i]

            video, scene = video_scene_list[i]
            # print(video, scene)
            if scene not in scene_psnrs_dict:
                scene_psnrs_dict[scene] = []
                scene_labels_dict[scene] = []

            psnrs = filter_psnrs(psnrs)
            scene_psnrs_dict[scene].append(psnrs)
            scene_labels_dict[scene].append(gt[i])

        # print(len(scene_psnrs_dict), len(scene_labels_dict))

        if NORMALIZE:
            scores_list = []
            labels_list = []
            for scene in scene_psnrs_dict:
                psnrs = np.concatenate(scene_psnrs_dict[scene], axis=0)
                labels = np.concatenate(scene_labels_dict[scene], axis=0)
                # psnrs = filter_psnrs_2(psnrs, window_size=500)
                # psnrs = filter_psnrs(psnrs)
                # psnrs = savitzky_golay(psnrs, window_size=51, order=3)

                scores = (psnrs - psnrs.min()) / (psnrs.max() - psnrs.min())
                scores_list.append(scores)
                labels_list.append(labels)

            scores = np.concatenate(scores_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
            auc = metrics.auc(fpr, tpr)
            results = RecordResult(fpr, tpr, auc, dataset, sub_loss_file)
        else:
            auc_list = []
            for scene in scene_psnrs_dict:
                psnrs = np.concatenate(scene_psnrs_dict[scene], axis=0)
                labels = np.concatenate(scene_labels_dict[scene], axis=0)
                # psnrs = filter_psnrs_2(psnrs, window_size=51)
                # psnrs = filter_psnrs(psnrs)
                # psnrs = savitzky_golay(psnrs, window_size=51, order=3)
                # psnrs = smooth_psnrs(psnrs)
                scores = psnrs

                fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
                auc = metrics.auc(fpr, tpr)
                auc_list.append(auc)

            auc = np.mean(auc_list)

            results = RecordResult([], [], auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model = {}'.format(optimal_results))
    return optimal_results


def compute_seen_unseen_auc(loss_file, *args):
    # ipdb.set_trace()

    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    gt_file = args[0]
    pattern = re.compile('folds_([0-9]+)_kth_([0-9]+)')
    folds, kth = pattern.findall(loss_file)[0]
    folds, kth = int(folds), int(kth)
    gt = load_semantic_frame_mask(gt_file, folds, kth)

    seen_optimal_results = RecordResult()
    unseen_optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        # the name of dataset, loss, and ground truth
        dataset, psnr_records, _ = load_psnr_gt(loss_file=sub_loss_file)

        # the number of videos
        num_videos = len(psnr_records)

        seen_scores = np.array([], dtype=np.float32)
        seen_labels = np.array([], dtype=np.int8)

        unseen_scores = np.array([], dtype=np.float32)
        unseen_labels = np.array([], dtype=np.int8)

        # video normalization
        for i in range(num_videos):
            distance = psnr_records[i]

            if NORMALIZE:
                distance -= distance.min()  # distances = (distance - min) / (max - min)
                distance /= distance.max()

            # seen idx
            seen_idx = gt[i] != 2
            # unseen idx
            unseen_idx = gt[i] != 1
            # seen_idx[0:DECIDABLE_IDX] = False
            # unseen_idx[0:DECIDABLE_IDX] = False

            seen_scores = np.concatenate((seen_scores, distance[seen_idx]), axis=0)
            seen_labels = np.concatenate((seen_labels, gt[i][seen_idx]), axis=0)

            unseen_scores = np.concatenate((unseen_scores, distance[unseen_idx]), axis=0)
            unseen_labels = np.concatenate((unseen_labels, gt[i][unseen_idx]), axis=0)

        seen_fpr, seen_tpr, _ = metrics.roc_curve(seen_labels, seen_scores, pos_label=0)
        seen_auc = metrics.auc(seen_fpr, seen_tpr)

        unseen_fpr, unseen_tpr, _ = metrics.roc_curve(unseen_labels, unseen_scores, pos_label=0)
        unseen_auc = metrics.auc(unseen_fpr, unseen_tpr)

        seen_results = RecordResult(seen_fpr, seen_tpr, seen_auc, dataset, sub_loss_file)
        unseen_results = RecordResult(unseen_fpr, unseen_tpr, unseen_auc, dataset, sub_loss_file)

        if seen_optimal_results < seen_results:
            seen_optimal_results = seen_results

        if unseen_optimal_results < unseen_results:
            unseen_optimal_results = unseen_results

        if os.path.isdir(loss_file):
            print('seen {}'.format(seen_results))
            print('unseen {}'.format(unseen_results))

    print('##### seen optimal result and model = {}'.format(seen_optimal_results))
    print('##### unseen optimal result and model = {}'.format(unseen_optimal_results))
    return seen_optimal_results


def compute_auc_with_gt_file(loss_file, *args):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    gt_file = args[0]
    pattern = re.compile('folds_([0-9]+)_kth_([0-9]+)')
    folds, kth = pattern.findall(loss_file)[0]
    folds, kth = int(folds), int(kth)
    gt = load_gt_by_json(gt_file, folds, kth)

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        # the name of dataset, loss, and ground truth
        dataset, psnr_records, _ = load_psnr_gt(loss_file=sub_loss_file)

        # the number of videos
        num_videos = len(psnr_records)

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        # video normalization
        for i in range(num_videos):
            distance = psnr_records[i]

            if NORMALIZE:
                distance -= distance.min()  # distances = (distance - min) / (max - min)
                distance /= distance.max()

            scores = np.concatenate((scores, distance[DECIDABLE_IDX:]), axis=0)
            labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        auc = metrics.auc(fpr, tpr)

        results = RecordResult(fpr, tpr, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model = {}'.format(optimal_results))
    return optimal_results


def compute_auc_with_threshold(loss_file, *args):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    psnr = 30
    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        # the name of dataset, loss, and ground truth
        dataset, psnr_records, gt = load_psnr_gt(loss_file=sub_loss_file)

        # the number of videos
        num_videos = len(psnr_records)

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        # video normalization
        for i in range(num_videos):
            distance = psnr_records[i]
            # less_thresholds = distance < psnr
            # distance[less_thresholds] = distance.min()

            if NORMALIZE:
                    distance -= distance.min()  # distances = (distance - min) / (max - min)
                    distance /= distance.max()

            scores = np.concatenate((scores, distance[DECIDABLE_IDX:]), axis=0)
            labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        auc = metrics.auc(fpr, tpr)

        results = RecordResult(fpr, tpr, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model = {}'.format(optimal_results))
    return optimal_results


def compute_feature_auc(loss_file, *args):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        # the name of dataset, loss, and ground truth
        dataset, f_means, gt = load_features_gt(loss_file=sub_loss_file)

        # the number of videos
        num_videos = len(f_means)

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        # video normalization
        for i in range(num_videos):
            distance = f_means[i]

            if NORMALIZE:
                distance -= distance.min()  # distances = (distance - min) / (max - min)
                distance /= distance.max()
                # distance -= np.mean(distance)
                # distance /= np.std(distance)
                # print(distance.max(), distance.min())

            scores = np.concatenate((scores, distance[DECIDABLE_IDX:]), axis=0)
            labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        auc = metrics.auc(fpr, tpr)

        results = RecordResult(fpr, tpr, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model = {}'.format(optimal_results))
    return optimal_results


def average_psnr(loss_file, *args):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    max_avg_psnr = -np.inf
    max_file = ''
    for file in loss_file_list:
        psnr_records = load_psnr(file)

        psnr_records = np.concatenate(psnr_records, axis=0)
        avg_psnr = np.mean(psnr_records)
        if max_avg_psnr < avg_psnr:
            max_avg_psnr = avg_psnr
            max_file = file
        print('{}, average psnr = {}'.format(file, avg_psnr))

    print('max average psnr file = {}, psnr = {}'.format(max_file, max_avg_psnr))


def calculate_psnr(loss_file, *args):
    optical_result = compute_auc(loss_file)
    print('##### optimal result and model = {}'.format(optical_result))

    mean_psnr = []
    for file in os.listdir(loss_file):
        file = os.path.join(loss_file, file)
        dataset, psnr_records, gt = load_psnr_gt(file)

        psnr_records = np.concatenate(psnr_records, axis=0)
        gt = np.concatenate(gt, axis=0)

        mean_normal_psnr = np.mean(psnr_records[gt == 0])
        mean_abnormal_psnr = np.mean(psnr_records[gt == 1])
        mean = np.mean(psnr_records)
        print('mean normal psrn = {}, mean abnormal psrn = {}, mean = {}'.format(
            mean_normal_psnr,
            mean_abnormal_psnr,
            mean)
        )
        mean_psnr.append(mean)
    print('max mean psnr = {}'.format(np.max(mean_psnr)))


def calculate_score(loss_file, *args):
    if not os.path.isdir(loss_file):
        loss_file_path = loss_file
    else:
        optical_result = compute_auc(loss_file)
        loss_file_path = optical_result.loss_file
        print('##### optimal result and model = {}'.format(optical_result))
    dataset, psnr_records, gt = load_psnr_gt(loss_file=loss_file_path)

    # the number of videos
    num_videos = len(psnr_records)

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    # video normalization
    for i in range(num_videos):
        distance = psnr_records[i]

        distance = (distance - distance.min()) / (distance.max() - distance.min())

        scores = np.concatenate((scores, distance[DECIDABLE_IDX:]), axis=0)
        labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)

    mean_normal_scores = np.mean(scores[labels == 0])
    mean_abnormal_scores = np.mean(scores[labels == 1])
    print('mean normal scores = {}, mean abnormal scores = {}, '
          'delta = {}'.format(mean_normal_scores, mean_abnormal_scores, mean_normal_scores - mean_abnormal_scores))


eval_type_function = {
    'compute_auc': compute_auc,
    'compute_eer': compute_eer,
    'compute_valid_auc': compute_valid_auc,
    'compute_filter_auc': compute_filter_auc,
    'compute_scene_auc': compute_scene_auc,
    'compute_seen_unseen_auc': compute_seen_unseen_auc,
    'precision_recall_auc': precision_recall_auc,
    'calculate_psnr': calculate_psnr,
    'calculate_score': calculate_score,
    'average_psnr': average_psnr,
    'average_psnr_sample': average_psnr,
    'compute_auc_with_gt_file': compute_auc_with_gt_file,
    'compute_feature_auc': compute_feature_auc,
    'compute_auc_with_threshold': compute_auc_with_threshold
}


def evaluate(eval_type, save_file, gt_file=''):
    assert eval_type in eval_type_function, 'there is no type of evaluation {}, please check {}' \
        .format(eval_type, eval_type_function.keys())
    eval_func = eval_type_function[eval_type]
    optimal_results = eval_func(save_file, gt_file)
    return optimal_results

