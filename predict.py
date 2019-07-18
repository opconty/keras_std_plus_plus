#-*- coding:utf-8 -*-
#'''
# Created on 19-7-8 上午11:28
#
# @Author: Greg Gao(laygin)
#'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.bbox_process import DetectSkew
from data.data_utils import preprocess_img
from models import create_model
from utils import resize_image


score = 0.1
center_line_score = 0.3
max_size = 768  # 608,768, 960, 1024
resize_version = 0  # 0-resize_image, 1-resize_image_v2, 2-cv2.resize
area_cont = 10

resize = True

vggmode = 'deconv'  # only for vgg16,  'deconv', 'upsample'
# config visualization
plot_centerline = True
vis_center_mask = True

weights_path = 'path/to/pretrained_weights'
assert os.path.exists(weights_path), 'weight path does not exist'

Detector = DetectSkew(cls_score=score,
                      center_line_score=center_line_score,
                      area_cont=area_cont)

model, _ = create_model(mode=vggmode)
model.load_weights(weights_path)


def plot(img):
    plt.imshow(img)
    plt.show()
    return


def predictions(model, img_path):
    img = cv2.imread(img_path)
    assert img is not None, 'image path does not exists'
    print(f'ori size: {img.shape}', end='\t')
    if resize:
        w, h = resize_image(img, max_size)
        img = cv2.resize(img, dsize=(w, h))
        print(f'resize size: {img.shape}')
    img_dis = img.copy()
    img = preprocess_img(img[..., ::-1])
    preds = model.predict(np.expand_dims(img, 0))
    qqs = Detector.detect(preds, img_dis.shape[:2])
    if len(qqs) == 0:
        return

    quads, scores = qqs[:, :8].reshape(-1, 4, 2), qqs[:, -1]

    # plot center line mask
    tcl_mask = Detector.make_center_mask()
    mask, _ = Detector.build_mask_from_mini_boxes(img_dis.shape[:2], Detector.get_miniboxes(img_dis.shape[:2]))
    tcl_mask = ((mask > 0) * (tcl_mask > 0)).astype(np.uint8) * 255
    if plot_centerline:
        plot(tcl_mask)

    # visualize center mask individual quads
    if vis_center_mask and len(quads):
        img_dis = Detector.draw_quads(img_dis, quads.astype(int), color=(255, 0, 0))

    # blending
    alpha = 0.6
    channel = 2
    img_dis[..., channel] = cv2.addWeighted(img_dis[..., channel], alpha, tcl_mask, 1 - alpha, 0)
    plot(img_dis[..., ::-1])


if __name__ == '__main__':
    from data import Rctw17Config

    cfg = Rctw17Config()

    for i in os.listdir(cfg.img_dir_test):
        predictions(model, os.path.join(cfg.img_dir_test, i))

