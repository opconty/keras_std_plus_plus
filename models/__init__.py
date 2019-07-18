#-*- coding:utf-8 -*-
#'''
# Created on 19-7-16 下午6:53
#
# @Author: Greg Gao(laygin)
#'''
from .std_vgg16_skew import StdVGG16


def create_model(mode='deconv'):
    M = StdVGG16
    model = M(mode=mode).std_net()
    return model, M

