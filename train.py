#-*- coding:utf-8 -*-
#'''
# Created on 19-5-11 下午3:56
#
# @Author: Greg Gao(laygin)
#'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from models.losses import cls_center, regr_h, regr_offset, cl,angle
from data import Rctw17Dataset, Rctw17Config
from data.img_aug import RandomContrast,RandomBrightness
from models import create_model


cfg = Rctw17Config()
cfg.input_size = 384  # 384,512,608,768
cfg.batch_size = 2
vggmode = 'deconv'  # only for vgg16,  'deconv', 'upsample'

optimizer = 'adam'  # sgd, adam
lr = 1e-4
monitor = 'val_loss'

cfg.shrink_ratio = 1.5

random_contrast = RandomContrast()
random_bright = RandomBrightness()
augs = [random_bright, random_contrast]


pre_weights ='path_to_pretrained_weights'
print('pretrained weightd: {}'.format(pre_weights))
init_ep = 0


def create_callbacks(mm, monitor='val_loss'):
    dd = os.path.join(cfg.checkpoints_dir, cfg.Name)
    if not os.path.exists(dd):
        os.mkdir(dd)

    nn = '%s_%s_ep{epoch:02d}_{loss:.3f}_{val_loss:.3f}.h5' % (mm, cfg.input_size)
    checkpoint = ModelCheckpoint(os.path.join(dd, nn),
                                 monitor=monitor, save_best_only=True, save_weights_only=True, verbose=1)
    earlystop = EarlyStopping(patience=10, monitor=monitor, verbose=1)
    reduce = ReduceLROnPlateau(monitor=monitor, patience=2)

    return [checkpoint, earlystop, reduce]


def _main():
    datagen_train = Rctw17Dataset(cfg, shuffle=True, is_train=True, augs=augs)
    datagen_test = Rctw17Dataset(cfg, shuffle=False, is_train=False)
    print('datagen train length: ', len(datagen_train) * cfg.batch_size)

    model, M = create_model(mode=vggmode)
    print('count_params: ', model.count_params())
    if os.path.exists(pre_weights):
        print('using pretrained weights: ', pre_weights)
        model.load_weights(pre_weights)
    else:
        print('training from scratch...')

    if optimizer == 'adam':
        opt = Adam(lr=lr)
    else:
        opt =SGD(lr=lr, momentum=0.99, decay=1e-6)

    loss = [cl, cls_center, regr_h, regr_offset, angle]
    model.compile(optimizer=opt,
                  loss=loss)

    model.fit_generator(datagen_train,
                        steps_per_epoch=len(datagen_train) ,
                        validation_data=datagen_test,
                        validation_steps=len(datagen_test),
                        epochs=cfg.epochs + init_ep,
                        initial_epoch=init_ep,
                        callbacks=create_callbacks(mm=M.__name__, monitor=monitor),
                        verbose=1
                        )


if __name__ == '__main__':

    _main()

    pass

