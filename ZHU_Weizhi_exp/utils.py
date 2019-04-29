import numpy as np
import math
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import transforms


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


# def get_margin(output, label):
#     top2 = output.topk(2)[1].cpu().data.numpy()
#     output = output.cpu().data.numpy()
#     label = label.cpu().data.numpy()
#     pred = np.argmax(output, axis=1)
#     margin = output[
#         np.arange(len(output)), label]  ## correct prediction needs minus the second largest prediction for margin.
#     error_idx = (label != pred)
#     margin[error_idx] = output[error_idx, label[error_idx]] - output[error_idx, pred[error_idx]]
#     margin[~error_idx] = output[~error_idx, label[~error_idx]] - output[~error_idx, top2[~error_idx][:, 1]]
#     return margin

def get_margin(output, label):
    top2 = output.topk(2)[1]
    pred = torch.argmax(output, dim=1)
    error_idx = (label != pred)
    margin = torch.zeros(len(output)).cuda()

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


def l2normalize(v, eps=1e-6):
    return v/(v.norm() + eps)

def get_spectral(w, iters, u, v):
    ## w should be detached
    height = w.shape[0]
    width = w.view(height, -1).shape[1]

    if u is None:
        u = torch.randn(height).cuda()
        v = torch.randn(width).cuda()
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)

    with torch.no_grad():
        for _ in range(iters):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sig = u.dot(w.view(height, -1).mv(v)).item()

    return sig, u, v

def get_21norm(w):
    height = w.shape[0] ##output channel
    norm_2 = w.view(height, -1).norm(p=2, dim=0)
    norm_21 = norm_2.norm(p=1).item()
    return norm_21

def get_12norm(w):
    height = w.shape[0] ##output channel
    norm_2 = w.view(height, -1).norm(p=1, dim=0)
    norm_21 = norm_2.norm(p=2).item()
    return norm_21

# def get_Lip_pow(resnet, u, v, block='BasicBlock'):
#     ## u & v are dictionary

#     def get_MidLip(conv, bn, u, v):
#         size = conv.weight.size()
#         w_cnn = conv.weight.view(size[0], -1)
#         scale = bn.weight / (bn.running_var.sqrt() + bn.eps)
#         w_cnn_new = w_cnn * scale.view(-1, 1)
#         pow_iters = 25 if u is None else 5
#         return get_spectral(w_cnn_new.data, pow_iters, u, v)

#     ## BasicBlock or Bottleneck
#     lip = 1
#     ## conv-1 & bn-1
#     w_norm, u['c1'], v['c1'] = get_MidLip(resnet.conv1, resnet.bn1, u['c1'], v['c1'])
#     lip *= w_norm

#     for m_, module in enumerate([resnet.layer1, resnet.layer2,
#                                  resnet.layer3, resnet.layer4]):
#         # module is nn.Sequential() object with num of block len(module)
#         lip_mod = 1
#         if block == 'BasicBlock':
#             for b_ in range(len(module)):
#                 lip_block = 1
#                 w_norm, u['m%db%dc1'%(m_,b_)], v['m%db%dc1'%(m_,b_)] = get_MidLip(module[b_].conv1, module[b_].bn1, 
#                                                                 u['m%db%dc1'%(m_,b_)], v['m%db%dc1'%(m_,b_)])
#                 lip_block *= w_norm
#                 w_norm, u['m%db%dc2'%(m_,b_)], v['m%db%dc2'%(m_,b_)] = get_MidLip(module[b_].conv2, module[b_].bn2, 
#                                                             u['m%db%dc2'%(m_,b_)], v['m%db%dc2'%(m_,b_)])                
#                 lip_block *= w_norm
#                 if len(list(zip(*module[b_].shortcut.named_children()))) != 0:
#                     w_norm, u['m%db%dc3'%(m_,b_)], v['m%db%dc3'%(m_,b_)] = get_MidLip(module[b_].shortcut[0], module[b_].shortcut[1], 
#                                                                 u['m%db%dc3'%(m_,b_)], v['m%db%dc3'%(m_,b_)])
#                     lip_block += w_norm
#                 else:
#                     lip_block += 1 ## need more careful analysis here!
#                 lip_mod *= (lip_block/10)
#         lip *= lip_mod

#     pow_iters = 25 if u['w_fc'] is None else 5
#     w_norm, u['w_fc'], v['w_fc'] = get_spectral(resnet.linear.weight.data, iters=pow_iters,
#                                                 u=u['w_fc'], v=v['w_fc'])
#     lip *= w_norm
#     return lip, u, v

def get_Lip_pow(resnet, u, v, block='BasicBlock'):
    ## u & v are dictionary

    def get_MidLip(conv, bn, u, v):
        size = conv.weight.size()
        w_cnn = conv.weight.view(size[0], -1)
        scale = bn.weight / (bn.running_var.sqrt() + bn.eps)
        w_cnn_new = w_cnn * scale.view(-1, 1)
        pow_iters = 30 if u is None else 8
        return get_spectral(w_cnn_new.data, pow_iters, u, v)

    ## BasicBlock or Bottleneck
    lip = 1
    ## conv-1 & bn-1
    w_norm, u['c1'], v['c1'] = get_MidLip(resnet.conv1, resnet.bn1, u['c1'], v['c1'])
    lip *= w_norm

    for m_, module in enumerate([resnet.layer1, resnet.layer2,
                                 resnet.layer3, resnet.layer4]):
        # module is nn.Sequential() object with num of block len(module)
        lip_mod = 1
        if block == 'BasicBlock':
            for b_ in range(len(module)):
                w_norm1, u['m%db%dc1'%(m_,b_)], v['m%db%dc1'%(m_,b_)] = get_MidLip(module[b_].conv1, module[b_].bn1, 
                                                                u['m%db%dc1'%(m_,b_)], v['m%db%dc1'%(m_,b_)])
                w_norm2, u['m%db%dc2'%(m_,b_)], v['m%db%dc2'%(m_,b_)] = get_MidLip(module[b_].conv2, module[b_].bn2, 
                                                            u['m%db%dc2'%(m_,b_)], v['m%db%dc2'%(m_,b_)])                
                if len(list(zip(*module[b_].shortcut.named_children()))) != 0:
                    w_norm3, u['m%db%dc3'%(m_,b_)], v['m%db%dc3'%(m_,b_)] = get_MidLip(module[b_].shortcut[0], module[b_].shortcut[1], 
                                                                u['m%db%dc3'%(m_,b_)], v['m%db%dc3'%(m_,b_)])
                    lip_block = (math.sqrt(w_norm1*w_norm2) + w_norm3)/10
                else:
                    lip_block = 1 ## need more careful analysis here!
                lip_mod *= lip_block
        lip *= lip_mod

    pow_iters = 30 if u['w_fc'] is None else 8
    w_norm, u['w_fc'], v['w_fc'] = get_spectral(resnet.linear.weight.data, iters=pow_iters,
                                                u=u['w_fc'], v=v['w_fc'])
    lip *= w_norm
    return lip, u, v

def get_Lip_L1(resnet, block='BasicBlock',s=1000):

    def get_MidLip(conv, bn, s):
        size = conv.weight.size()
        w_cnn = conv.weight.view(size[0], -1)
        scale = bn.weight / (bn.running_var.sqrt() + bn.eps)
        w_cnn_new = w_cnn * scale.view(-1, 1)
        return w_cnn_new.norm(1).item() / s

    ## BasicBlock or Bottleneck
    lip = 1
    ## conv-1 & bn-1
    lip *= get_MidLip(resnet.conv1, resnet.bn1, s=s)

    for module in [resnet.layer1, resnet.layer2,
                   resnet.layer3, resnet.layer4]:
        ## module is nn.Sequential() object with num of block len(module)
        lip_mod = 1
        if block == 'BasicBlock':
            for i in range(len(module)):
                lip_block = 1
                lip_block *= get_MidLip(module[i].conv1, module[i].bn1, s=1)
                lip_block *= get_MidLip(module[i].conv2, module[i].bn2, s=1)
                if len(list(zip(*module[i].shortcut.named_children()))) != 0:
                    lip_block += get_MidLip(module[i].shortcut[0],
                                            module[i].shortcut[1], s=1)
                lip_mod *= (lip_block/s)
        lip *= lip_mod
    
    lip *= resnet.linear.weight.norm(2).item() / s
    return lip

