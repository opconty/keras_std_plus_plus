#-*- coding:utf-8 -*-
#'''
# Created on 19-7-18 上午10:55
#
# @Author: Greg Gao(laygin)
#'''
import numpy as np


def compute_distance_of_2pts(p0, p1):
    return np.sqrt(np.square(p0[0] - p1[0]) + np.square(p0[1] - p1[1]))


def compute_angle_of_2pts(p0, p1):
    return np.arctan((p0[1] - p1[1]) / (p1[0] - p0[0] + 1e-6))


def resize_image(im, max_img_size):
    im_width = np.minimum(im.shape[1], max_img_size)
    if im_width == max_img_size < im.shape[1]:
        im_height = int((im_width / im.shape[1]) * im.shape[0])
    else:
        im_height = im.shape[0]
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_width = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_width, d_height

