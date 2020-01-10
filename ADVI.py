#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2020/01/05
@file: ADVI.py
"""
import os
import numpy as np
import pickle
from scipy import stats
import matplotlib.pyplot as plt

from dataset.UCIdataset import UCIDataset
from dataset.Facedataset import FaceDataset
from ADVI_bnn import train, test, construct_nn

def log_prob(ppc, Y_test):
    pred = ppc['out']
    sd = np.std(ppc['out'], axis=0)
    logp = [np.mean(stats.norm.logpdf(pred[:, i] - Y_test[i],
                                      loc=0, scale=sd[i])) for i in range(len(sd))]
    print ('log-likelihood mean: %.2f'%np.mean(logp))
    print('log-likelihood std: %.3f' % np.std(logp))
    return np.mean(logp), np.std(logp)


def show_inference(inference, b=0, e=100):
    plt.plot(-inference.hist[b:e], alpha=.3)
    plt.ylabel('ELBO')
    plt.xlabel('iteration')
    plt.show()
    return None


def run(dataset_name, root_path, hypers, shape='train_test'):
    np.random.seed(123)

    for time in range(hypers['times']):
        outpath = os.path.join(root_path, str(time))
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        if dataset_name == 'face':
            data = FaceDataset("./dataset", 0.9)
        else:
            data = UCIDataset(dataset_name, 0.9)

        X = np.append(data.Xtrain, data.Xtest, axis=0)
        Y = np.append(data.Ytrain, data.Ytest)
        data.Ytrain = data.Ytrain.reshape(len(data.Ytrain), )
        data.Ytest = data.Ytest.reshape(len(data.Ytest), )
        neural_network = construct_nn(
            X, Y, data.Xtrain, data.Ytrain, hypers)

        approx_file = os.path.join(outpath, "approx.pkl")
        inference_file = os.path.join(outpath, "inference.pkl")
        ppc_file = os.path.join(outpath, "ppc.pkl")
        if shape == 'train_test':
            inference, approx = train(
                neural_network, inference_file, approx_file, hypers)
            ppc = test(
                neural_network,
                approx,
                data.Xtest,
                data.Ytest,
                ppc_file,
                trace_samples=hypers['pred_samples'],
                pred_samples=hypers['pred_samples'])
        if shape == 'test':
            with open(inference_file, 'rb') as f:
                inference = pickle.load(f)
            with open(approx_file, 'rb') as f:
                approx = pickle.load(f)

            ppc = test(
                neural_network,
                approx,
                data.Xtest,
                data.Ytest,
                ppc_file,
                trace_samples=hypers['pred_samples'],
                pred_samples=hypers['pred_samples'])
        if shape == 'show':
            with open(inference_file, 'rb') as f:
                inference = pickle.load(f)
            with open(approx_file, 'rb') as f:
                approx = pickle.load(f)
            with open(ppc_file, 'rb') as f:
                ppc = pickle.load(f)

        show_inference(inference, b=0, e=hypers['n_sample'])
        # show_inference(inference, b=0, e=hypers['n_sample'])
        log_prob(ppc, data.Ytest)

    return None


if __name__ == '__main__':
    dataset_name = "face" #数据集名称
    root_path = os.path.join('/home/yunnd/project/result/ADVI', dataset_name) #存储结果的路径
    hypers = {'conc': {'sd': 1,
                       'lr': 0.01,
                       'n_hidden': 50,
                       'n_sample': 100000,
                       'pred_samples': 5000,
                       'times': 1
                       },
              'powe': {'sd': 1,
                       'lr': 0.01,
                       'n_hidden': 50,
                       'n_sample': 100000,
                       'pred_samples': 5000,
                       'times': 1
                       },
              'yach': {'sd': 1,
                       'lr': 0.01,
                       'n_hidden': 50,
                       'n_sample': 100000,
                       'pred_samples': 5000,
                       'times': 1
                       },
              'prot': {'sd': 1,
                       'lr': 0.01,
                       'n_hidden': 50,
                       'n_sample': 100000,
                       'pred_samples': 5000,
                       'times': 1
                       },
              'face': {'sd': 5,
                       'lr': 0.1,
                       'n_hidden': 50,
                       'n_sample': 10000,
                       'pred_samples': 1000,
                       'times': 1
                       },
              }
    run(dataset_name, root_path, hypers[dataset_name], 'train_test')

    pass