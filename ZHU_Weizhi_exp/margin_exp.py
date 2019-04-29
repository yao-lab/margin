import os
import re
import sys
import pickle
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from utils import *
import seaborn as sns

use_gpu = torch.cuda.is_available()

## This is for experiment-2(qMargins)
def train_model(model, criterion, optimizer, loaders, num_epochs=100, 
                verbose=True, net='Res18', s=10000, normalization='pow'):
## net: Res18 / CNN / Res34
## normalization: two ways to calculate spectral norm of conv kernal, 
##                'l1' L1-bound and 'power' power iterations
##
    since = time.time()
    steps = 0

    log_saver = {  #'num_params': [],
                   'train_mle': [],
                   'train_nmle': [],
                   'train_error': [],
                   #'train_margin_error': [],
                   #'train_ramp_qmar': [],
                   #'test_ramp_qmar': [],
                   'test_mle': [],
                   'test_nmle': [],
                   'test_error': [],
                   #'test_margin_error': [],
                   'lip_l1': [],
                   'lip_pow': [],
                   'lip_12': [],
                   'norm_21': [],
                   #'qmargin':[],
                   'train_dist_margin':[],
                   'train_dist_nmargin':[],
                   'test_dist_margin':[],
                   'test_dist_nmargin':[],
                   #'train_nmargin': [], 
                   #'test_nmargin': [],
                   }
    if net == 'CNN':
        u = {}; v = {};
        for i in [0,1,2,3,4]:
            log_saver['w%d' % i] = []
            u['w%d'%i] = None
            v['w%d'%i] = None
        log_saver['w_fc'] = []
        u['w_fc'] = None
        v['w_fc'] = None

    if net == 'Res18':
        u = {}; v = {};
        u['c1'] = None; v['c1'] = None
        for m_ in [0,1,2,3]: ##module
            for b_ in [0,1]: ##block
                for c_ in [1,2,3]: ##conv layer
                    u['m%db%dc%d'%(m_, b_, c_)] = None
                    v['m%db%dc%d'%(m_, b_, c_)] = None
        u['w_fc'] = None; v['w_fc'] = None


    for epoch in range(num_epochs):
        if (epoch % verbose == 0):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

        for phase in ['train', 'train1', 'test']:
            margin_ep = []
            nmargin_ep = []

            mle_meter = AverageMeter()
            nmle_meter = AverageMeter()
            ramp_qmar_meter = AverageMeter()
            error_meter = AverageMeter()
            margin_error_meter = AverageMeter()

            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            for i, data in enumerate(loaders[re.findall('[a-zA-Z]+', phase)[0]]):
                inputs, labels = data
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    steps += 1

                N = outputs.size(0)

                if phase != 'train':
                    margin_temp = get_margin(outputs, labels).detach().cpu().numpy()
                    nmargin_temp = margin_temp/log_saver['lip_'+normalization][-1]
                    ## collect margin
                    margin_ep = np.append(margin_ep, margin_temp)
                    nmargin_ep = np.append(nmargin_ep, nmargin_temp)

                    nloss = criterion(outputs/log_saver['lip_'+normalization][-1], labels)
                    nmle_meter.update(nloss.data.item(), N)
                    mle_meter.update(loss.data.item(), N)
                    error_meter.update(accuracy(outputs.data, labels.data)[-1].item(), N)

            epoch_mle = mle_meter.avg
            epoch_nmle = nmle_meter.avg
            epoch_error = 1 - error_meter.avg / 100

            if phase == 'train':
                model.eval()
                
                if (net == 'Res18') or (net == 'Res34'):
                    lip_pow, u, v = get_Lip_pow(model, u, v)
                    log_saver['lip_pow'].append(lip_pow)
                    lip_l1 = get_Lip_L1(model)
                    log_saver['lip_l1'].append(lip_l1)
                
                elif net == 'CNN':
                    lip_l1 = 1; lip_pow=1; lip_12=1; norm_21 = 0;
                    for i in range(6):
                        if i <= 4:
                            size = eval('model.features.conv%d.weight.size()' % i)
                            w_cnn = eval('model.features.conv%d.weight.view(size[0],-1)' % i)
                            w_bn = eval('model.features.bn%d.weight' % i)
                            var_bn = eval('model.features.bn%d.running_var' % i)
                            eps_bn = eval('model.features.bn%d.eps' % i)
                            scale = w_bn / (var_bn.sqrt() + eps_bn)
                            w_cnn_new = w_cnn * scale.view(-1, 1)
                            ## L1- upper bound
                            w_norm_l1 = w_cnn_new.norm(1).data.item()
                            lip_l1 *= w_norm_l1 / 1000
                            
                            w_norm_12 = get_12norm(w_cnn_new)
                            lip_12 *= w_norm_12 / 1000

                            ## Power iterations
                            pow_iters = 25 if u['w%d'%i] is None else 5
                            w_norm_pow, u['w%d'%i], v['w%d'%i] = get_spectral(w_cnn_new.data,
                                        iters=pow_iters, u=u['w%d'%i], v=v['w%d'%i])
                            
                            w_norm_21 = get_21norm(w_cnn_new.data/w_norm_pow)
                            norm_21 += (1/6 + w_norm_21)**(2/3)
                            
                            lip_pow *= w_norm_pow
                            log_saver['w%d' % i].append([w_norm_l1, w_norm_12, w_norm_pow])
                        else:
                            #w_norm = model.classifier.weight.norm(2).data.item()
                            pow_iters = 25 if u['w_fc'] is None else 5
                            w_norm_pow, u['w_fc'], v['w_fc'] = get_spectral(model.classifier.weight.data, 
                                            iters=pow_iters, u=u['w_fc'], v=v['w_fc'])
                            
                            w_norm_21 = get_21norm(model.classifier.weight.data/w_norm_pow)
                            norm_21 += (1/6 + w_norm_21)**(2/3)

                            lip_l1 *= w_norm_pow / 1000
                            lip_pow *= w_norm_pow / 100
                            lip_12 *= w_norm_pow / 1000
                            log_saver['w_fc'].append(w_norm_pow)
                    
                    log_saver['norm_21'].append(norm_21 ** (3/2))
                    log_saver['lip_l1'].append(lip_l1)
                    log_saver['lip_pow'].append(lip_pow)
                    log_saver['lip_12'].append(lip_12)

            if phase == 'train1':
                
                log_saver['train_dist_margin'].append(margin_ep)
                log_saver['train_dist_nmargin'].append(nmargin_ep)
                log_saver['train_mle'].append(epoch_mle)
                log_saver['train_nmle'].append(epoch_nmle)
                log_saver['train_error'].append(epoch_error)

            elif phase == 'test':

                log_saver['test_dist_margin'].append(margin_ep)
                log_saver['test_dist_nmargin'].append(nmargin_ep)
                log_saver['test_mle'].append(epoch_mle)
                log_saver['test_nmle'].append(epoch_nmle)
                log_saver['test_error'].append(epoch_error)

            if (phase != 'train') & (epoch % verbose == 0):
                print('%s: CrossEntropy: %.4f, Error: %.4f, Lip(%s):%.4f' % 
                    (phase, epoch_mle, epoch_error, normalization, log_saver['lip_'+normalization][-1]))

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    ## normalized margin for training (inverse of RC est)
    for q in np.linspace(0, 1, 11):
        log_saver['train_qnmargin_%.1f' % q] = [np.quantile(nmargin_ep, q)
                                for nmargin_ep in log_saver['train_dist_nmargin']]
        log_saver['test_qnmargin_%.1f' % q] = [np.quantile(nmargin_ep, q)
                                for nmargin_ep in log_saver['test_dist_nmargin']]

    # ## train/test ramp loss/ margin error
    # for rho in [1,3,5,10,12,15,18,20,30,50]:
    #     log_saver['train_ramp_%d' % rho] = [ramp_loss(nmargin_ep, rho)
    #                             for nmargin_ep in log_saver['train_dist_nmargin']]
    #     log_saver['train_marerr_%d' % rho] = [margin_error(nmargin_ep, rho)
    #                             for nmargin_ep in log_saver['train_dist_nmargin']]
    #     log_saver['test_ramp_%d' % rho] = [ramp_loss(nmargin_ep, rho)
    #                             for nmargin_ep in log_saver['test_dist_nmargin']]
    #     log_saver['test_marerr_%d' % rho] = [margin_error(nmargin_ep, rho)
    #                             for nmargin_ep in log_saver['test_dist_nmargin']]
    return model, log_saver

def plot(log, q, result_dir, savefig=True):
    fontdict = {'size': 30}
    
    def get_fig(i, title):
        fig = plt.figure(i, figsize=(20, 10))
        ax = fig.add_subplot(111)
        plt.title(title, fontsize=30, y=1.04)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        return fig, ax

    fig1, ax1 = get_fig(1, 'CrossEntropy on Cifar10')
    fig2, ax2 = get_fig(2, 'Error on Cifar10')
    fig7, ax7 = get_fig(7, 'Normalized qMargin with q %.2f' % q)
    fig8, ax8 = get_fig(8, 'qMargin with q %.2f' % q)
    fig9, ax9 = get_fig(9, 'Rademacher Complexity')
    fig14, ax14 = get_fig(14, 'Ramp loss with q %.2f' % q)

    ax1.plot(log['train_mle'], linewidth=3, label='training')
    ax1.plot(log['test_mle'], linewidth=3, label='test')
    ax2.plot(log['train_error'], linewidth=3, label='training')
    ax2.plot(log['test_error'], linewidth=3, label='test')

    ax7.plot(np.array(log['qmargin'])/(np.array(log['Lip'])+1e-3), linewidth=3)
    ax8.plot(log['qmargin'], linewidth=3)
    ax9.plot(np.array(log['Lip'])/(np.array(log['qmargin'])+1e-3), linewidth=3)

    ax14.plot(log['train_ramp_qmar'], linewidth=3, label='training')
    ax14.plot(log['test_ramp_qmar'], linewidth=3, label='test')


    for ax in [ax1, ax2, ax14]:
        ax.set_xlabel('Number of epochs', fontdict=fontdict)
        ax.legend(loc='upper right', fontsize=20)

    if savefig:
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        fig1.savefig(result_dir + 'crossentropy_q%.2f.png' % q)
        plt.close(fig1)
        fig2.savefig(result_dir + 'error_q%.2f.png' % q)
        plt.close(fig2)
        fig7.savefig(result_dir + 'nqmar_q%.2f.png' % q)
        plt.close(fig7)
        fig8.savefig(result_dir + 'qmar_q%.2f.png' % q)
        plt.close(fig8)
        fig9.savefig(result_dir + 'rc_q%.2f.png' % q)
        plt.close(fig9)
        fig14.savefig(result_dir + 'ramp_q%.2f.png' % q)
        plt.close(fig14)

    # sns.distplot(np.array(log['dist_margin'][-1]), kde=True, label='un-normalized margin')
    # sns_kde = sns.distplot(np.array(log['dist_margin'][-1])/log['Lip'][-1], kde=True, label='normalized margin')

    # if savefig:
    #     fig15 = sns_kde.get_figure()
    #     fig15.savefig(result_dir + 'dist_margin.png')
    #     plt.close(fig15)

