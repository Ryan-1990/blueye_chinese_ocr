# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from crnn import LiteCrnn, CRNNHandle
from psenet import PSENet, PSENetHandel
from crnn.keys import alphabetChinese as alphabet


def crop_rect(img, rect, alph=0.6):
    img = np.asarray(img)
    center, size, angle = rect[0], rect[1], rect[2]
    min_size = min(size)
    if (angle > -45):
        center, size = tuple(map(int, center)), tuple(map(int, size))
        size = (int(size[0] + min_size * alph), int(size[1] + min_size * alph))
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    else:
        center = tuple(map(int, center))
        size = tuple([int(rect[1][1]), int(rect[1][0])])
        size = (int(size[0] + min_size * alph), int(size[1] + min_size * alph))
        angle -= 270
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    img_crop = Image.fromarray(img_crop)
    #img_crop.show()
    return img_crop


def crnnRec(im, rects_re, f=1.0):
    results = []
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    for index, rect in enumerate(rects_re):
        degree, w, h, cx, cy = rect
        partImg = crop_rect(im, ((cx, cy), (h, w), degree))
        newW, newH = partImg.size
        partImg_array = np.uint8(partImg)

        if newH > 1.5 * newW:
            partImg_array = np.rot90(partImg_array, 1)
        partImg = Image.fromarray(partImg_array).convert("RGB")
        partImg_ = partImg.convert('L')

        try:
            simPred = crnn_handle.predict(partImg_)
        except:
            continue
        if simPred.strip() != u'':
            results.append({'cx': cx * f, 'cy': cy * f, 'text': simPred, 'w': newW * f, 'h': newH * f,
                            'degree': degree})
        
        if abs(degree) > 0:
            x0 = rect[3] - rect[1]/2
            y0 = rect[4] - rect[2]/2
            x1 = rect[3] + rect[1]/2
            y1 = rect[4] + rect[2]/2
        else:
            x0 = rect[3] - rect[2]/2
            y0 = rect[4] - rect[1]/2
            x1 = rect[3] + rect[2]/2
            y1 = rect[4] + rect[1]/2
        draw.rectangle(((x0, y0), (x1, y1)), outline ="red")
    im.show()

    return results


def text_predict(img):
    preds, boxes_list, rects_re, t = text_handle.predict(img, long_size=pse_long_size)
    result = crnnRec(np.array(img), rects_re)
    return result

# psenet
pse_scale = 1
pse_long_size = 720
pse_model_type = "mobilenetv2"
pse_model_path = "models/psenet.pth"
text_detect_net = PSENet(backbone=pse_model_type, pretrained=False, result_num=6, scale=pse_scale)
text_handle = PSENetHandel(pse_model_path, text_detect_net, pse_scale)

# crnn
nh = 256
crnn_model_path = "models/crnn_lstm.pth"
crnn_net = LiteCrnn(32, 1, len(alphabet) + 1, nh, n_rnn=2, leakyRelu=False, lstmFlag=True)
crnn_handle = CRNNHandle(crnn_model_path, crnn_net)

if __name__ == '__main__':
    img = './test/test.png'
    img = Image.open(img).convert('RGB')
    #img.show()
    img = np.array(img)
    text = text_predict(img)
    text = sorted(text, key=lambda k: (k['cy'], k['cx']))
    print('Results:', list(map(lambda x: x['text'], text)))
