#-*- coding:utf-8 -*-
#'''
# Created on 19-5-15 上午11:48
#
# @Author: Greg Gao(laygin)
#'''
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from skimage.draw import polygon as drawpoly
from functools import reduce
from utils import compute_distance_of_2pts, compute_angle_of_2pts
from keras.applications.imagenet_utils import preprocess_input


# with each text instances
def readxml_skew_v2(path):
    gtboxes = []
    img_file = ''
    xml = ET.parse(path)
    for elem in xml.iter():
        if 'filename' in elem.tag:
            img_file = elem.text
        text = []
        if 'text' in elem.tag:
            for aa in list(elem):
                if 'object' in aa.tag:
                    for attr in list(aa):
                        if 'bndbox' in attr.tag:
                            x0 = float(attr.find('x0').text)
                            y0 = float(attr.find('y0').text)
                            x1 = float(attr.find('x1').text)
                            y1 = float(attr.find('y1').text)
                            x2 = float(attr.find('x2').text)
                            y2 = float(attr.find('y2').text)
                            x3 = float(attr.find('x3').text)
                            y3 = float(attr.find('y3').text)

                            text.append([[x0,y0], [x1,y1],[x2,y2],[x3,y3]])
            gtboxes.append(np.array(text))

    return gtboxes, img_file


class TextInstance(object):
    '''
    a single whole text box class
    '''
    def __init__(self, points):
        '''

        :param points: numpy array, (n, 4, 2), each quad contains four points
        '''
        self.points = points

    def get_center_points(self):
        p0 = self.get_top_points()
        p1 = self.get_bottom_points()

        return (p0 + p1) / 2

    def get_top_points(self):
        top_points1 = self.points[:, [0, 1], :].reshape(-1, 2)
        top_points2 = self.points[:, [0, 1], :].reshape(-1, 2)
        if len(top_points1) > len(top_points2):
            top_points = top_points1
        else:
            top_points = top_points2
        return top_points

    def get_bottom_points(self):
        bottom_points1 = self.points[:, [3, 2], :].reshape(-1, 2)
        bottom_points2 = self.points[:, [3, 2], :].reshape(-1, 2)
        if len(bottom_points1) > len(bottom_points2):
            bottom_points = bottom_points1
        else:
            bottom_points = bottom_points2
        return bottom_points

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


def fill_polygon(mask, polygon, value):
    """
    fill polygon in the mask with value
    :param mask: input mask
    :param polygon: polygon to draw
    :param value: fill value
    """
    rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=(mask.shape[0], mask.shape[1]))
    mask[rr, cc] = value


def make_center_line_mask(top_points,
                          bottom_points,
                          center_points,
                          tcl_mask,
                          expand=0.3):
    shrink = 0
    for i in range(shrink, len(center_points) - 1 - shrink):
        c1 = center_points[i]
        c2 = center_points[i + 1]
        top1 = top_points[i]
        top2 = top_points[i + 1]
        bottom1 = bottom_points[i]
        bottom2 = bottom_points[i + 1]

        p1 = c1 + (top1 - c1) * expand
        p2 = c1 + (bottom1 - c1) * expand
        p3 = c2 + (bottom2 - c2) * expand
        p4 = c2 + (top2 - c2) * expand
        polygon = np.stack([p1, p2, p3, p4])

        fill_polygon(tcl_mask, polygon, value=1)


def gaussian(kernel):
    sigma = ((kernel - 1) * 0.5 -1) * 0.3 + 0.8
    s = 2*(sigma**2)
    dx = np.exp(-np.square(np.arange(kernel) - int(kernel/2)) / s)
    return np.reshape(dx, (-1, 1))


def _warp_gaumap_box(imgsize, quad, forshow=False):
    mask = np.zeros((imgsize[0], imgsize[1]), dtype=np.uint8)
    side_len = int(compute_distance_of_2pts(quad[0], quad[1]))
    dx, dy = gaussian(side_len), gaussian(side_len)
    gau_map = np.multiply(dy, dx.T)
    if forshow:
        mask[:side_len, :side_len] = gau_map * 255
    else:
        mask[:side_len, :side_len] = gau_map

    # 使用透视变换将高斯map转换为小方框形状
    gao_pts = np.array([[0, 0], [side_len, 0], [side_len, side_len], [0, side_len]], dtype=np.float32)
    dst_pts = np.array(quad, dtype=np.float32)
    M = cv2.getPerspectiveTransform(gao_pts, dst_pts)
    dst_mask = cv2.warpPerspective(mask, M, (imgsize[1], imgsize[0]))
    return dst_mask


# get the whole image gaussian mask
def warp_img_gaussian_mask(imgsize, quads, stride=1):
    all_masks = []
    for text in quads:
        for quad in text:
            m = _warp_gaumap_box(imgsize, quad/stride, forshow=False)
            all_masks.append(m)

    # reduce to single mask
    if len(all_masks):
        img_mask = reduce(lambda a, b: cv2.bitwise_or(a, b), all_masks)
        return img_mask
    else:
        return np.zeros(imgsize, dtype=np.uint8)


# for arbitrary shape text
def generate_targets_skew(cfg, annots):
    quads = np.copy(annots)
    scale_map = np.zeros((cfg.input_size//cfg.stride_size,cfg.input_size//cfg.stride_size, 2))
    offset_map = np.zeros((cfg.input_size//cfg.stride_size, cfg.input_size//cfg.stride_size,3))
    center_map = np.zeros((cfg.input_size//cfg.stride_size, cfg.input_size//cfg.stride_size, 3))
    center_map[:,:,1] = 1
    # angle map for miniboxes' orientation
    angle_map = np.zeros((cfg.input_size//cfg.stride_size,cfg.input_size//cfg.stride_size, 2))

    # below mask is the same size as input image, no stride
    tcl_mask = np.zeros((cfg.input_size, cfg.input_size), dtype=np.uint8)

    if len(quads):
        center_map[:, :, 0] = warp_img_gaussian_mask(
            (cfg.input_size // cfg.stride_size, cfg.input_size // cfg.stride_size), quads, stride=cfg.stride_size)
        for text_idx in range(len(quads)):
            if len(quads[text_idx]) == 0:
                continue
            text = TextInstance(quads[text_idx])

            make_center_line_mask(text.get_top_points(),
                                  text.get_bottom_points(),
                                  text.get_center_points(),
                                  tcl_mask)
            region = cfg.region
            boxes = quads[text_idx] / cfg.stride_size
            boxes[boxes <= 0] = 0
            for idx in range(boxes.shape[0]):
                p0, p1, p2, p3 = boxes[idx]
                # quads could be triangle after random cropped
                cx, cy = boxes[idx].mean(0).astype(np.int32)
                center_map[cy, cx, 2] = 1

                scale_map[cy - region:cy + region + 1, cx - region:cx + region + 1, 0] = max(np.log(compute_distance_of_2pts(p0, p1)), 1e-5)
                scale_map[cy - region:cy + region + 1, cx - region:cx + region + 1, 1] = 1

                angle_map[cy - region:cy + region + 1, cx - region:cx + region + 1, 0] = compute_angle_of_2pts(quads[text_idx][idx][-1],
                                                                                                               quads[text_idx][idx][-2])
                angle_map[cy - region:cy + region + 1, cx - region:cx + region + 1, 1] = 1

                offset_map[cy, cx, 0] = (p0[1]+p2[1]) / 2 - cy - 0.5
                offset_map[cy, cx, 1] = (p0[0]+p2[0]) / 2 - cx - 0.5
                offset_map[cy, cx, 2] = 1

    return tcl_mask, center_map, scale_map, offset_map, angle_map


def preprocess_img(rgb_img, mode='caffe'):
    return preprocess_input(rgb_img, mode=mode)


