import os
import argparse
import configparser

import ipdb


def get_dir(directory):
    """
    get the directory, if no such directory, then make it.

    @param directory: The new directory.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


def parser_args():
    parser = argparse.ArgumentParser(description='Options to run the network.')
    parser.add_argument('-g', '--gpu', type=str, nargs='*', choices=['0', '1', '2', '3',
                                                                     '4', '5', '6', '7', '8', '9'], required=True,
                        help='the device id of gpu.')
    parser.add_argument('-i', '--iters', type=int, default=1,
                        help='set the number of iterations, default is 1')
    parser.add_argument('-b', '--batch', type=int, default=4,
                        help='set the batch size, default is 4.')

    parser.add_argument('-d', '--dataset', type=str,
                        help='the name of dataset.')

    parser.add_argument('-o', '--output_dir', type=str, default="./data/pretrains",
                        help='the path of the output directory')

    parser.add_argument('--num_his', type=int, default=4,
                        help='set the time steps, default is 4.')

    parser.add_argument('--prednet', type=str, default='cyclegan_convlstm',
                        choices=['resnet_convlstm', 'cyclegan_convlstm', 'cyclegan_conv2d',
                                 'resnet_conv3d', 'unet_conv2d', 'conv2d_deconv2d', 'MCNet',
                                 'two_cyclegan_convlstm_classifier',
                                 'unet_conv2d_instance_norm', 'cyclegan_convlstm_deconv1',
                                 'two_cyclegan_convlstm_focal_loss',
                                 'MLE_2_NN', 'MLE_2_SVM', 'MLE_1_SVM', 'Pred_1_SVM', 'TRI_1_SVM'],
                        help='set the name of prediction network, default is cyclegan_convlstm')

    parser.add_argument('--label_level', type=str, default='temporal', choices=['normal', 'video', 'tune_video',
                                                                                'temporal', 'tune_temporal',
                                                                                'tune_video_temporal',
                                                                                'temporal_mle_nn',
                                                                                'temporal_mle_svm',
                                                                                'pixel'],
                        help='set the label level.')

    parser.add_argument('--k_folds', type=int, default=5,
                        help='set the number of folds.')
    parser.add_argument('--kth', type=int, default=1,
                        help='choose the kth fold.')
    parser.add_argument('--margin', type=float, default=1.0, help='value of margin.')

    parser.add_argument('--pretrain', type=str, default='',
                        help='pretrained MLE-FFP, only using for feature extraction and training MLE-2NN,'
                             'MLE-2-SVM and MLE-1-SVM')
    parser.add_argument('--snapshot_dir', type=str, default='',
                        help='if it is folder, then it is the directory to save models, '
                             'if it is a specific model.ckpt-xxx, then the system will load it for testing.')
    parser.add_argument('--summary_dir', type=str, default='', help='the directory to save summaries.')
    parser.add_argument('--psnr_dir', type=str, default='', help='the directory to save psnrs results in testing.')

    parser.add_argument('--evaluate', type=str, default='compute_auc',
                        help='the evaluation metric, default is compute_auc')

    parser.add_argument('--interpolation', action='store_true', help='use interpolation to increase fps or not.')
    parser.add_argument('--multi', action='store_true', help='use multi scale and crop  or not')

    return parser.parse_args()


class Const(object):
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const.{}".format(name))
        if not name.isupper():
            raise self.ConstCaseError('const name {} is not all uppercase'.format(name))

        self.__dict__[name] = value

    def __str__(self):
        _str = '<================ Constants information ================>\n'
        for name, value in self.__dict__.items():
            _str += '\t{}\t{}\n'.format(name, value)

        return _str

    def set_margin(self, margin):
        self.__dict__['MARGIN'] = margin


args = parser_args()
const = Const()

# inputs constants
const.OUTPUT_DIR = args.output_dir
const.DATASET = args.dataset
const.K_FOLDS = args.k_folds
const.KTH = args.kth
const.LABEL_LEVEL = args.label_level

const.GPUS = args.gpu

const.BATCH_SIZE = args.batch
const.NUM_HIS = args.num_his
const.ITERATIONS = args.iters
const.PREDNET = args.prednet
const.EVALUATE = args.evaluate
const.INTERPOLATION = args.interpolation
const.MULTI = args.multi


# set training hyper-parameters of different datasets
config = configparser.ConfigParser()
assert config.read(os.path.join('./data/hyper_params', '{}.ini'.format(const.LABEL_LEVEL)))

const.NORMALIZE = config.getboolean(const.DATASET, 'NORMALIZE')
const.HEIGHT = config.getint(const.DATASET, 'HEIGHT')
const.WIDTH = config.getint(const.DATASET, 'WIDTH')
const.TRAIN_FOLDER = config.get(const.DATASET, 'TRAIN_FOLDER')
const.TEST_FOLDER = config.get(const.DATASET, 'TEST_FOLDER')
const.FRAME_MASK = config.get(const.DATASET, 'FRAME_MASK')
const.PIXEL_MASK = config.get(const.DATASET, 'PIXEL_MASK')

if args.pretrain:
    const.PRETRAIN_MODEL = args.pretrain
else:
    const.PRETRAIN_MODEL = config.get(const.DATASET, 'PRETRAIN_MODEL')

const.PSNR_FILE = config.get(const.DATASET, 'PSNR_FILE')


# const.MARGIN = config.getfloat(const.DATASET, 'MARGIN')
const.MARGIN = args.margin
const.LAMBDA = config.getfloat(const.DATASET, 'LAMBDA')

const.LRATE_G = eval(config.get(const.DATASET, 'LRATE_G'))
const.LRATE_G_BOUNDARIES = eval(config.get(const.DATASET, 'LRATE_G_BOUNDARIES'))

const.INTERVAL = config.getint(const.DATASET, 'INTERVAL')
const.MULTI_INTERVAL = config.getboolean(const.DATASET, 'MULTI_INTERVAL')

const.MODEL_SAVE_FREQ = config.getint(const.DATASET, 'MODEL_SAVE_FREQ')

if const.LABEL_LEVEL == 'normal':
    const.SAVE_DIR = '{label_level}/{dataset}/prednet_{PREDNET}'.format(
        label_level=const.LABEL_LEVEL, dataset=const.DATASET, PREDNET=const.PREDNET
    )
else:
    const.SAVE_DIR = '{label_level}/{dataset}/prednet_{PREDNET}_folds_{K_FOLDS}_kth_{KTH}_/MARGIN_{MARGIN}_' \
                     'LAMBDA_{LAMBDA}'.format(label_level=const.LABEL_LEVEL,
                                              dataset=const.DATASET, PREDNET=const.PREDNET,
                                              MARGIN=const.MARGIN, LAMBDA=const.LAMBDA,
                                              K_FOLDS=const.K_FOLDS, KTH=const.KTH)

if args.snapshot_dir:
    # if the snapshot_dir is model.ckpt-xxx, which means it is the single model for testing.
    if os.path.exists(args.snapshot_dir + '.meta') or os.path.exists(args.snapshot_dir + '.data-00000-of-00001') or \
            os.path.exists(args.snapshot_dir + '.index'):
        const.SNAPSHOT_DIR = args.snapshot_dir
    else:
        const.SNAPSHOT_DIR = get_dir(args.snapshot_dir)
else:
    const.SNAPSHOT_DIR = get_dir(os.path.join(const.OUTPUT_DIR, 'checkpoints', const.SAVE_DIR))

if args.summary_dir:
    const.SUMMARY_DIR = get_dir(args.summary_dir)
else:
    const.SUMMARY_DIR = get_dir(os.path.join(const.OUTPUT_DIR, 'summary', const.SAVE_DIR))

if args.psnr_dir:
    const.PSNR_DIR = get_dir(args.psnr_dir)
else:
    if const.INTERPOLATION:
        const.PSNR_DIR = get_dir(os.path.join(const.OUTPUT_DIR, 'psnrs', const.SAVE_DIR + '_interpolation'))
    else:
        const.PSNR_DIR = get_dir(os.path.join(const.OUTPUT_DIR, 'psnrs', const.SAVE_DIR))


