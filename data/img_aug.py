#-*- coding:utf-8 -*-
#'''
# Created on 19-5-11 下午2:25
#
# @Author: Greg Gao(laygin)
#'''
import numpy as np
import cv2


def quads_area(quads):
    '''

    :param quads:(n, 4, 2) for quadrilateral points
    :return:
    '''
    p0, p1, p2, p3 = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    a1 = np.abs(np.cross(p0-p1, p1-p2)) / 2
    a2 = np.abs(np.cross(p0-p3, p3-p2)) / 2
    return a1+a2


def clip_quads(quads, clip_box, alpha=0.25):
    '''

    :param quads: shape is (n, 4, 2)
    :param clip_box:[0, 0, w, h]
    :param alpha:
    :return:
    '''
    areas_ = quads_area(quads)
    quads[:,:,0] = np.minimum(np.maximum(quads[:,:,0], clip_box[0]), clip_box[2])  # 0<= x <= w
    quads[:, :, 1] = np.minimum(np.maximum(quads[:, :, 1], clip_box[1]), clip_box[3])  # 0<= y <= h

    delta_area = (areas_ - quads_area(quads)) / (areas_ + 1e-6)
    mask = (delta_area < (1-alpha)) & (quads_area(quads)>0)
    return quads[mask, :, :]


class RandomCrop(object):
    """Crop randomly the image in a sample.

        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, coors):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,
                left: left + new_w]

        for i in range(len(coors)):
            coors[i] -= [left, top]
            coors[i] = clip_quads(coors[i], [0, 0, new_w, new_h], 0.25)

        return image, coors


class CenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, coors):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = int((h - new_h) // 2)
        left = int((w - new_w) // 2)
        image = image[top: top + new_h,
                left: left + new_w]

        for i in range(len(coors)):
            coors[i] -= [left, top]
            coors[i] = clip_quads(coors[i], [0, 0, new_w, new_h], 0.25)

        return image, coors


class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, coors, size=None):
        h, w = image.shape[:2]
        if size is not None:
            self.output_size = size

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))

        for i in range(len(coors)):
            coors[i][:, :, 0] = coors[i][:, :, 0] * new_w / w
            coors[i][:,:,1] = coors[i][:,:,1] * new_h / h

        return img, coors


'''augmentation'''
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert 255 >= delta >= 0, 'delta is invalid'
        self.delta = delta

    def __call__(self, img, coors=None):
        img = img.astype(np.float32)
        if np.random.randint(0,2):
            delta = np.random.uniform(-self.delta, self.delta)
            img += delta
        return np.clip(img, 0, 255), coors


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, img, coors=None):
        img = img.astype(np.float32)
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            img *= alpha
        return np.clip(img, 0, 255), coors

