# -*- coding: utf-8 -*-
__author__ = 'huangyf'

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Logger(object):
    def __init__(self, name):
        self.reset()
        self.name = name

    def reset(self):
        self.step_logger = []
        self.loss_logger = []
        self.error_logger = []
        self.step_logger_test = []
        self.loss_logger_test = []
        self.error_logger_test = []

    def log_train(self, step, loss, error):
        self.step_logger.append(step)
        self.loss_logger.append(loss)
        self.error_logger.append(error)

    def log_test(self, step, loss, error):
        self.step_logger_test.append(step)
        self.loss_logger_test.append(loss)
        self.error_logger_test.append(error)

    # margin definition? 
    # margin[sample_i]= prediction_y_i[label] - \max(k!=label) prediction_y_i[k]


def get_margin(output, label):
    top2 = output.topk(2)[1].cpu().data.numpy()
    output = output.cpu().data.numpy()
    label = label.cpu().data.numpy()
    pred = np.argmax(output, axis=1)
    margin = output[
        np.arange(len(output)), label]  ## correct prediction needs minus the second largest prediction for margin.
    error_idx = (label != pred)
    margin[error_idx] = output[error_idx, label[error_idx]] - output[error_idx, pred[error_idx]]
    margin[~error_idx] = output[~error_idx, label[~error_idx]] - output[~error_idx, top2[~error_idx][:, 1]]
    return margin


def ramp_loss(margin, gamma):
    loss = 1 - margin / gamma
    loss[np.where(margin > gamma)[0]] = 0
    loss[np.where(margin < 0)[0]] = 1
    return loss.mean()


def margin_error(margin, gamma):
    return np.where(margin < gamma, 1, 0).mean()


def get_Lip(model, type, s=1e4):
    def get_MidLip(conv, bn=None):
        size = conv.weight.size()
        w_cnn = conv.weight.view(size[0], -1)
        if bn is not None:
            scale = bn.weight / (bn.running_var.sqrt() + bn.eps)
            w_cnn_new = w_cnn * scale.view(-1, 1)
        else:
            w_cnn_new = w_cnn.detach()
        return w_cnn_new.norm(1).item() / s

    lip = 1
    if type == 'resnet18':
        lip *= get_MidLip(model.conv1, model.bn1)

        for module in [model.layer1, model.layer2,
                       model.layer3, model.layer4]:
            for i in range(2):
                lip_1 = 1
                lip_1 *= get_MidLip(module[i].conv1, module[i].bn1) * s
                lip_1 *= get_MidLip(module[i].conv2, module[i].bn2)
                if len(list(zip(*module[i].shortcut.named_children()))) != 0:
                    lip_1 += get_MidLip(module[i].shortcut[0],
                                        module[i].shortcut[1])

                lip *= lip_1
        lip *= model.linear.weight.norm(2).item() / s
    elif type == 'vgg16':
        for i in range(len(model.features)):
            if isinstance(model.features[i], nn.Conv2d):
                lip *= get_MidLip(model.features[i],
                                  model.features[i + 1])
        lip *= model.classifier.weight.norm(2).item() / s

    elif type == 'alex':
        for i in range(len(model.features)):
            if isinstance(model.features[i], nn.Conv2d):
                lip *= get_MidLip(model.features[i])
        lip *= model.classifier.weight.norm(2).item() / s

    return lip
