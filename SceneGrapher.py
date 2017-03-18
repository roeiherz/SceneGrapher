from __future__ import print_function

import cv2
import numpy
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense

from keras_frcnn.Lib.Zoo import ModelZoo
from tfFasterRcnn.lib.datasets.Imdb import Imdb
from tfFasterRcnn.lib.datasets.factory import get_imdb
from tfFasterRcnn.lib.model.config import cfg_from_file, cfg
from tfFasterRcnn.lib.model.train_val import get_training_roidb

__author__ = 'roeih'

TRAIN_IMDB = "voc_2007_trainval"
TEST_IMDB = "voc_2007_test"
ITERS = 400000
CFG_FILE = "vgg16.yml"
NET = "vgg16"
WEIGHT = "Data/imagenet_weights/vgg16.ckpt"


# --weight Data/imagenet_weights/vgg16.ckpt --imdb voc_2007_trainval --imdbval voc_2007_test  --iters 400000 --cfg ../experiments/cfgs/vgg16.yml --net vgg16

def combined_roidb(imdb_names):
    """
  Combine multiple roidbs
  """

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset2 `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = Imdb(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


def train():
    """
    This function is training the model
    :return:
    """
    # Train Net
    imdb, roidb = combined_roidb(TRAIN_IMDB)
    cfg_from_file(full_config_file)
    print('debug')
    # net = Resnet101(batch_size=cfg.TRAIN.IMS_PER_BATCH)
    # train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
    #           pretrained_model=args.weight,
    #           max_iters=args.max_iters)


if __name__ == '__main__':
    # Path for config file
    path = "SceneGrapher/tfFasterRcnn/experiments/cfgs/"
    full_config_file = path + CFG_FILE
    output_folder = '/home/roeih/SceneGrapher/results'
    # data_dest = '/home/roeih/SceneGrapher/Deep/datasets/ccru/315_newdataset/ccru_315_newdataset.p'

    im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(numpy.float32)
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    im = im.transpose((2, 0, 1))
    im = numpy.expand_dims(im, axis=0)

    model = ModelZoo().vgg19(convolution_only=True)
    model.add(Convolution2D(2048, 1, 1, activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))

    # tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # tfconfig.gpu_options.allow_growth = True
    # sess = tf.Session(config=tfconfig)
    # net = vgg16(batch_size=1)
    # anchors = [8, 16, 32]
    #
    # net.create_architecture(sess, mode="TEST", num_classes=imdb.num_classes, caffe_weight_path=args.weight,
    #                         tag='default', anchor_scales=anchors)





    print('debug')

    # --weight Data/imagenet_weights/vgg16.ckpt --imdb voc_2007_trainval --imdbval voc_2007_test  --iters 400000 --cfg ../experiments/cfgs/vgg16.yml --net vgg16
    # --weight Data/imagenet_weights/res101.ckpt --imdb coco_2014_train+coco_2014_valminusminival --imdbval voc_2007_test  --iters 400000 --cfg ../experiments/cfgs/res101.yml --net res101
