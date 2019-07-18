#-*- coding:utf-8 -*-
#'''
# Created on 19-5-13 下午4:09
#
# @Author: Greg Gao(laygin)
#'''
import cv2
import numpy as np
from skimage.draw import polygon as draw_polygon


class Quad:
    def __init__(self, box):
        '''
        transfer box to quads
        :param box:model output box: [xmin, ymin, xmax, ymax, angle, score]
        '''
        self.box = box

    @property
    def score(self):
        return self.box[-1]

    @property
    def degree(self):
        return self.box[-2] * 180 / np.pi

    @property
    def centerX(self):
        xmin, _, xmax, _, _, _ = self.box
        return (xmin + xmax) / 2

    @property
    def centerY(self):
        _, ymin, _, ymax, _, _ = self.box
        return (ymin + ymax) / 2

    def box2quad(self):
        xmin, ymin, xmax, ymax, angle, score = self.box
        cx = self.centerX
        cy = self.centerY

        diagonal = np.square(xmin - xmax) + np.square(ymax - ymin)
        side = np.sqrt(diagonal / 2)
        rect = ((cx, cy), (side, side), self.degree)

        quad = np.int0(cv2.boxPoints(rect))  # left down, left up, right up, right down
        return quad


class DetectSkew():
    def __init__(self,
                 cls_score=0.1,
                 center_line_score=0.5,
                 area_cont=0):
        self.cls_score = cls_score  # to filter mini boxes
        self.aspect_ratio = 1
        self.center_line_score = center_line_score
        self.area_cont = area_cont

        # predictions
        self.center_line = None
        self.center_cls = None
        self.scale_regr = None
        self.offset = None
        self.angle = None

    @staticmethod
    def is_in_contour(cont, point):
        x, y = point
        return cv2.pointPolygonTest(cont, (x, y), False) > 0

    def _extract_prediction(self, Y):
        self.center_line, self.center_cls, self.scale_regr, self.offset, self.angle = Y

    def get_miniboxes(self, imgsize):
        '''

        :param imgsize:
        :return: all boxes, each box is a instance of Quad
        '''
        stride_size = 4
        seman = self.center_cls[0, :, :, 0]
        height = self.scale_regr[0, :, :, 0]
        angle = self.angle[0, :, :, 0]
        offset_y = self.offset[0, :, :, 0]
        offset_x = self.offset[0, :, :, 1]
        y_c, x_c = np.where(seman >= self.cls_score)
        boxs = []
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * stride_size
            w = self.aspect_ratio * h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * stride_size - w / 2), max(0, (
                        y_c[i] + o_y + 0.5) * stride_size - h / 2)
            a = angle[y_c[i], x_c[i]]
            boxs.append(Quad([x1, y1, min(x1 + w, imgsize[1]), min(y1 + h, imgsize[0]), a, s]))

        return boxs

    def make_center_mask(self):
        center_line = self.center_line[0,:,:,0]
        center_line = (center_line >= self.center_line_score)
        center_mask = (center_line * 255).astype(np.uint8)
        return center_mask

    @staticmethod
    def build_mask_from_mini_boxes(img_size, boxes):
        def clip_quad(quads, img_size):
            quads[:, 0] = np.minimum(np.maximum(quads[:, 0], 0), img_size[1])  # 0<= x <= w
            quads[:, 1] = np.minimum(np.maximum(quads[:, 1], 0), img_size[0])  # 0<= y <= h
            return quads

        mask = np.zeros(img_size, dtype=np.uint8)
        scores = []
        for b in boxes:
            quads = clip_quad(quads=b.box2quad(), img_size=img_size)
            rr, cc = draw_polygon(quads[:, 1], quads[:, 0])
            mask[rr, cc] = 255

            scores.append(b.score)

        return mask, np.mean(scores)

    @staticmethod
    def _find_contours(mask, chain=cv2.CHAIN_APPROX_SIMPLE):
        try:
            _, conts, _ = cv2.findContours(mask, cv2.RETR_TREE, chain)
        except:
            conts, _ = cv2.findContours(mask, cv2.RETR_TREE, chain)

        return conts

    def classify_boxes_to_a_contour(self, miniboxes, contour):
        text = []
        for b in miniboxes:
            cx, cy = b.centerX, b.centerY
            if self.is_in_contour(contour, (cx, cy)):
                text.append(b)

        return text

    def _get_words_boxes_from_mask(self, mask, score):
        '''

        :param mask: binary mask
        :return: words level bounding boxes
        '''
        conts = self._find_contours(mask)
        rects = []
        for c in conts:
            r = cv2.minAreaRect(c)  # (x,y),(w,h), a = rect
            b = np.int0(cv2.boxPoints(r))   # left down, left up, right up, right down
            rects.append(np.append(b.flatten(), score))

        return np.array(rects)

    def detect(self, Y, imgsize:(tuple, list)):
        assert len(imgsize) == 2, 'img size must be a tuple or list which includes height and width'
        self._extract_prediction(Y)
        mini_boxes = self.get_miniboxes(imgsize)
        center_mask = self.make_center_mask()
        mask, _ = self.build_mask_from_mini_boxes(imgsize, mini_boxes)
        # make center line mask from mask and center score mask
        tcl_mask = ((mask>0) * (center_mask>0)).astype(np.uint8) * 255

        # find contours from tcl mask, to split mini boxes into different text lines
        conts = self._find_contours(tcl_mask)

        all_quads = []
        for c in conts:
            if cv2.contourArea(c) > self.area_cont:
                text = self.classify_boxes_to_a_contour(mini_boxes, c)
                if len(text):
                    text_mask, score = self.build_mask_from_mini_boxes(imgsize, text)
                    rects = self._get_words_boxes_from_mask(text_mask, float(score))

                    all_quads.extend(rects)

        if len(all_quads):
            all_quads = np.stack(all_quads)

        return all_quads

    @staticmethod
    def draw_quads(img_to_be_plotted, all_quads, color: tuple = (0, 0, 255)):
        if len(all_quads):
            img_to_be_plotted = cv2.drawContours(img_to_be_plotted, all_quads, -1, color, 2)

        return img_to_be_plotted

