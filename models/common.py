# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import platform
import warnings
from copy import copy
from pathlib import Path



import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torchvision.ops
from PIL import Image
from torch.cuda import amp
from functools import partial
from torch import Tensor
from torch.cuda import amp
from torch import einsum
from einops.layers.torch import Rearrange
from einops import rearrange
from utils.activations import MetaAconC

from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, colorstr, increment_path, make_divisible,
                           non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
# from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.utils import _pair
from utils.torch_utils import time_sync
import torch.nn.functional as F





def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))



class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True): 
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)  


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))



class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):  ## ch_in, ch_out
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        # self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.m3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m1(x)
            y2 = self.m2(y1)
            y3 = self.m3(y2)

            return self.cv2(torch.cat([x, y1, y2, y3], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)  # 1*1卷积的作用：降低通道数
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 池化没有改变特征图的大小

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            out = self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

            return out



class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)







class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=None, dnn=True):
        # Usage:
        #   PyTorch:      weights = *.pt
        #   TorchScript:            *.torchscript.pt
        #   CoreML:                 *.mlmodel
        #   TensorFlow:             *_saved_model
        #   TensorFlow:             *.pb
        #   TensorFlow Lite:        *.tflite
        #   ONNX Runtime:           *.onnx
        #   OpenCV DNN:             *.onnx with dnn=True
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        suffix, suffixes = Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '', '.mlmodel']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, onnx, tflite, pb, saved_model, coreml = (suffix == x for x in suffixes)  # backend booleans
        jit = pt and 'torchscript' in w.lower()
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

        if jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif pt:  # PyTorch
            from models.experimental import attempt_load  # scoped to avoid circular import
            model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
            stride = int(model.stride.max())  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        elif coreml:  # CoreML *.mlmodel
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
            import onnxruntime
            session = onnxruntime.InferenceSession(w, None)
        else:  # TensorFlow model (TFLite, pb, saved_model)
            import tensorflow as tf
            if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                                   tf.nest.map_structure(x.graph.as_graph_element, outputs))

                LOGGER.info(f'Loading {w} for TensorFlow *.pb inference...')
                graph_def = tf.Graph().as_graph_def()
                graph_def.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
            elif saved_model:
                LOGGER.info(f'Loading {w} for TensorFlow saved_model inference...')
                model = tf.keras.models.load_model(w)
            elif tflite:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                if 'edgetpu' in w.lower():
                    LOGGER.info(f'Loading {w} for TensorFlow Edge TPU inference...')
                    import tflite_runtime.interpreter as tfli
                    delegate = {'Linux': 'libedgetpu.so.1',  # install https://coral.ai/software/#edgetpu-runtime
                                'Darwin': 'libedgetpu.1.dylib',
                                'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = tfli.Interpreter(model_path=w, experimental_delegates=[tfli.load_delegate(delegate)])
                else:
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt:  # PyTorch
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
            return y if val else y[0]
        elif self.coreml:  # CoreML *.mlmodel
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
            conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
            y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
        elif self.onnx:  # ONNX
            im = im.cpu().numpy()  # torch to numpy
            if self.dnn:  # ONNX OpenCV DNN
                self.net.setInput(im)
                y = self.net.forward()
            else:  # ONNX Runtime
                y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        else:  # TensorFlow model (TFLite, pb, saved_model)
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.pb:
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            elif self.saved_model:
                y = self.model(im, training=False).numpy()
            elif self.tflite:
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., 0] *= w  # x
            y[..., 1] *= h  # y
            y[..., 2] *= w  # w
            y[..., 3] *= h  # h
        y = torch.tensor(y)
        return (y, []) if val else y


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    multi_label = False  # NMS multiple labels per box
    max_det = 1000  # maximum number of detections per image

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def autoshape(self):
        LOGGER.info('AutoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model.model[-1]  # Detect()
        m.stride = fn(m.stride)
        m.grid = list(map(fn, m.grid))
        if isinstance(m.anchor_grid, list):
            m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes,
                                    multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


# 注意力机制


class conv(nn.Module):
    def __init__(self, in_planes, map_reduce=4):  # 通道数降低为原来的8分之一
        super(conv, self).__init__()
        out_planes = in_planes // map_reduce
        self.conv = BasicConv(in_planes, out_planes, kernel_size=1, stride=1, groups=1, relu=False)

    def forward(self, x):
        out = self.conv(x)
        return out





class Sub(nn.Module):
    def __init__(self, c1):
        super(Sub, self).__init__()

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        out = x1 - x2
        return out

class Mul(nn.Module):
    def __init__(self, c1):
        super(Mul, self).__init__()

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        out = x1 * x2
        return out


class CARAFE(nn.Module):
    # CARAFE: Content-Aware ReAssembly of FEatures       https://arxiv.org/pdf/1905.02188.pdf
    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(c1, c1 // 4, 1)
        self.encoder = nn.Conv2d(c1 // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        N, C, H, W = x.size()
        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W,
                                              self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        x = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                          self.kernel_size // 2, self.kernel_size // 2),
                  mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        x = x.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        x = x.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        x = x.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
        x = x.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(x, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        # print("up shape:",out_tensor.shape)
        return out_tensor
    
    
    
class SpatialGate(nn.Module):
    def __init__(self,
                 gate_channel,
                 reduction_ratio=8,
                 dilation_conv_num=2,
                 dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()

        self.gate_s.add_module(
            'gate_s_conv_reduce0',
            nn.Conv2d(gate_channel,
                      gate_channel // reduction_ratio,
                      kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0',
                               nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())

        # 进行多个空洞卷积，丰富感受野
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                'gate_s_conv_di_%d' % i,
                nn.Conv2d(gate_channel // reduction_ratio,
                          gate_channel // reduction_ratio,
                          kernel_size=3,
                          padding=dilation_val,
                          dilation=dilation_val))
            self.gate_s.add_module(
                'gate_s_bn_di_%d' % i,
                nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())

        self.gate_s.add_module(
            'gate_s_conv_final',
            nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, x):
        return self.gate_s(x).expand_as(x)


params= {
    "global_context":{
        "weighted_gc": True,
        "gc_reduction": 1,
        "compete": False,
        "head": 1,
    },
    "spatial_mixer":{
        "use_globalcontext":True,
        "useSecondTokenMix": True,
        "mix_size_1": 5,
        "mix_size_2": 7,
        "fc_factor": 8,
        "fc_min_value": 16,
        "useSpatialAtt": True
    },
}



class GlobalContext(nn.Module):
    def __init__(self, dim ,act_layer=nn.GELU, params=params):
        super().__init__()
        # bottleneck information
        # "Compete to compute." NeurIPS 2013
        self.compete = params["global_context"]["compete"]
        if self.compete:
            self.fc1 = nn.Linear(dim,2*dim//params["global_context"]["gc_reduction"])
            self.fc2 = nn.Linear(dim//params["global_context"]["gc_reduction"], dim)
        else:
            self.fc = nn.Sequential(
                nn.Linear( dim,dim//params["global_context"]["gc_reduction"]),
                act_layer(),
                nn.Linear(dim//params["global_context"]["gc_reduction"], int(dim*0.5))
            )
        self.weight_gc = params["global_context"]["weighted_gc"]
        if self.weight_gc:
            self.head = params["global_context"]["head"]
            self.scale = (dim//self.head) ** -0.5
            self.rescale_weight = nn.Parameter(torch.ones(self.head))
            self.rescale_bias = nn.Parameter(torch.zeros(self.head))
            self.epsilon = 1e-5

    def _get_gc(self, gap): # gap [b,c]
        if self.compete:
            b,c = gap.size()
            gc = self.fc1(gap).reshape([b,2,-1])
            gc, _ = gc.max(dim=1)
            gc = self.fc2(gc)
            return gc
        else:
            return self.fc(gap)


    def forward(self,x):
        if self.weight_gc:
            b,c,w,h = x.size()
            x = rearrange(x,"b c x y -> b c (x y)")
            gap = x.mean(dim=-1, keepdim=True)
            q, g = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h = self.head), [x,gap])  #[b,head, hdim, n]
            sim = einsum('bhdi,bhjd->bhij', q, g.transpose(-1, -2)).squeeze(dim=-1) * self.scale  #[b,head, w*h]
            std, mean = torch.std_mean(sim, dim=[1,2], keepdim=True)
            sim = (sim-mean)/(std+self.epsilon)
            sim = sim * self.rescale_weight.unsqueeze(dim=0).unsqueeze(dim=-1) + self.rescale_bias.unsqueeze(dim=0).unsqueeze(dim=-1)
            sim = sim.reshape(b,self.head,1, w, h) # [b, head, 1, w, h]
            gc = self._get_gc(gap.squeeze(dim=-1)).reshape(b,self.head,-1).unsqueeze(dim=-1).unsqueeze(dim=-1)  # [b, head, hdim, 1, 1]
            gc = rearrange(sim*gc, "b h d x y -> b (h d) x y")  # [b, head, hdim, w, h] - > [b,c,w,h]
        else:
            gc = self._get_gc(x.mean(dim=-1).mean(dim=-1)).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return gc  # [b,c,w,h] for weighted or [b,c,1,1]

class DeepWise_PointWise_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeepWise_PointWise_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
    
class SELayer(nn.Module):
    def __init__(self, c1,  r=16,act_layer=nn.GELU, params=params):
        super(SELayer, self).__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.context = GlobalContext(dim=c1, act_layer=act_layer, params=params)
        self.conv3 =  nn.Conv2d(c1//2, c1, 1, 1)
        self.dw1 = DeepWise_PointWise_Conv(c1, c1)
        self.conv1 = nn.Conv2d(c1, c1//r, kernel_size=1, stride=1,padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c1//r)
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.l1 = nn.Linear(c1, c1 // r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c1//r, c1, kernel_size=1, stride=1,padding=0, bias=False)
        # self.pconv = Partial_conv3(c1)
        # self.dconv = DWConv(c1, c1, 1)
        # self.dpconv = DeepWise_PointWise_Conv(c1, c1)
        # self.conv3 = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=False)
        # self.l2 = nn.Linear(c1 // r, c1, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        # print(x.size())
        # b, c, _, _ = x.size()
        # y = self.avgpool(x)
        v = self.context(x)
        y = x + self.conv3(v)
        y = self.dw1(y)
        y = self.conv1(y)
        y = self.bn(y)
        # y = self.l1(y)
        y = self.relu(y)
        y = self.conv2(y)
        wei = self.sig(y)
        z = x * wei
        # print(z.size())
        # z = self.pconv(z)
        # z = self.dconv(z)
        # z = self.dpconv(z)
        # z = self.conv3(z)
        # y = self.l2(y)
        # y = self.sig(y)
        # y = y.view(b, c, 1, 1)
        return z


class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


# 三个分支concat操作
class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)







