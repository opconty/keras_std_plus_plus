#-*- coding:utf-8 -*-
#'''
# Created on 19-5-11 下午1:44
#
# @Author: Greg Gao(laygin)
#'''
import os
from keras.utils import Sequence
import numpy as np
import cv2
from config import Config
from .img_aug import CenterCrop, RandomCrop, Resize
from .data_utils import generate_targets_skew, readxml_skew_v2
from .data_utils import preprocess_img


class Rctw17Config(Config):
    Name = 'rctw17'
    data_dir = os.path.join(Config.basedir, 'icdar2017rctw')
    assert os.path.exists(data_dir), Exception('data directory does not exist..')

    img_dir_trainval = os.path.join(data_dir, 'trainval')
    annot_dir_trainval = os.path.join(data_dir, 'std_annotations_skew')
    train_txt = os.path.join(data_dir, 'train.txt')
    val_txt = os.path.join(data_dir, 'val.txt')
    assert os.path.exists(train_txt) and os.path.exists(val_txt)

    img_dir_test = os.path.join(data_dir, 'testing', 'images')
    submit_dir = os.path.join(data_dir, 'testing', 'submit')
    if not os.path.exists(submit_dir):
        os.mkdir(submit_dir)

    batch_size = 16
    input_size = 384
    stride_size = 4
    scale = 'h'
    center_line_stride = 1
    region = 2
    alpha = 0.999
    model = 'vgg16'
    mode = 'caffe'  # preprocess mode
    shrink_ratio = 1.5


class Rctw17Dataset(Sequence):
    def __init__(self, cfg: Rctw17Config,
                 shuffle=False,
                 vis=False,
                 is_train=False,
                 augs=None):
        self.cfg = cfg
        self.is_train = is_train
        self.shuffle = shuffle
        if self.is_train:
            self.txt = self.cfg.train_txt
        else:
            self.txt = self.cfg.val_txt

        self.annot_names = self._read_txt(self.txt)
        self.vis = vis
        self.augs = augs
        self.si = self._size()
        self.center_crop = CenterCrop(self.cfg.input_size)
        self.random_crop = RandomCrop(self.cfg.input_size)
        self.resize = Resize((cfg.input_size, cfg.input_size))
        self.on_epoch_end()

    @staticmethod
    def _read_txt(txt_file):
        mmp = []
        with open(txt_file, 'r') as f:
            for i in f:
                mmp.append(i.strip())
        return mmp

    def __len__(self):
        return len(self.annot_names) // self.cfg.batch_size

    def _size(self):
        return len(self.annot_names)

    def __getitem__(self, idx):
        lb = idx * self.cfg.batch_size
        rb = (idx + 1) * self.cfg.batch_size
        if rb > self.si:
            rb = self.si
            lb = rb - self.cfg.batch_size
        b = rb - lb

        b_img = np.zeros(
            (b, self.cfg.input_size, self.cfg.input_size, 3), dtype=np.float32)
        b_center_map = np.zeros(
            (b, self.cfg.input_size//self.cfg.stride_size, self.cfg.input_size//self.cfg.stride_size, 3))
        b_scale_map = np.zeros(
            (b, self.cfg.input_size // self.cfg.stride_size, self.cfg.input_size // self.cfg.stride_size, 2))
        b_offset_map = np.zeros(
            (b, self.cfg.input_size // self.cfg.stride_size, self.cfg.input_size // self.cfg.stride_size, 3))
        b_tcl_mask = np.zeros((b, self.cfg.input_size // self.cfg.center_line_stride,
                              self.cfg.input_size // self.cfg.center_line_stride, 1), dtype=np.uint8)
        b_angle_map = np.zeros(
            (b, self.cfg.input_size // self.cfg.stride_size, self.cfg.input_size // self.cfg.stride_size, 2))

        for i, ann_name in enumerate(self.annot_names[lb:rb]):
            a = self._aug_img(ann_name)
            if a is not None:
                img, tcl_mask, center_map, scale_map, offset_map, angle_map = a
                b_img[i] = img
                b_center_map[i] = center_map
                b_scale_map[i] = scale_map
                b_offset_map[i] = offset_map
                b_angle_map[i] = angle_map
                b_tcl_mask[i] = np.expand_dims(tcl_mask, axis=-1)
            else:
                print(ann_name)

        return [b_img], [b_tcl_mask, b_center_map, b_scale_map, b_offset_map, b_angle_map]

    def _aug_img(self, ann_name):
        try:
            ann_path = os.path.join(self.cfg.annot_dir_trainval, ann_name)
            bboxes, img_name = readxml_skew_v2(ann_path)
            img_path = os.path.join(self.cfg.img_dir_trainval, img_name)

            img = cv2.imread(img_path)
            assert img is not None, Exception(f'img path does not exists: {img_path}')
            h, w = img.shape[:2]

            # if image minimum larger than target size, crop it directly
            # elif image maximum larger than target size, resize it to maximum then crop
            # else resize to n times target size and crop, which n larger than 1
            minimum = np.minimum(h, w)
            maximum = np.maximum(h, w)
            resize_size = None
            if self.cfg.input_size * self.cfg.shrink_ratio < minimum:
                resize_size = self.cfg.input_size * self.cfg.shrink_ratio
            elif maximum > self.cfg.input_size >= minimum:
                resize_size = maximum
            elif self.cfg.input_size >= maximum:
                resize_size = int(self.cfg.shrink_ratio * self.cfg.input_size)

            if resize_size:
                img, bboxes = self.resize(img, bboxes, size=int(resize_size))

            if self.is_train:
                img, bboxes = self.random_crop(img, bboxes)
            else:
                img, bboxes = self.center_crop(img, bboxes)

            # apply augments
            if self.augs:
                for a in self.augs:
                    img, bboxes = a(img, bboxes)

            img = img.astype(np.float32)
            if not self.vis:
                img = preprocess_img(img[...,::-1], self.cfg.model)

            tcl_mask, center_map, scale_map, offset_map, angle_map = generate_targets_skew(self.cfg, bboxes)
            return img, tcl_mask, center_map, scale_map, offset_map, angle_map
        except Exception as e:
            import traceback
            traceback.print_exc()

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.annot_names)


if __name__ == '__main__':

    pass


