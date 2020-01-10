#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2019/11/24
@file: DVI.py
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
from matplotlib import rc
plt.rcParams['savefig.dpi'] = 300
rc('text', usetex=True)
rc('font', size=15)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)

import gaussian_variables as gv
import utils as u
import plot_utils as pu
import bayes_layers as bnn
from bayes_models import MLP, PointMLP, AdaptedMLP
from dataset.UCIdataset import UCIDataset
from dataset.Facedataset import FaceDataset


def make_model(hypers):
    if hypers['method'].lower().strip() == 'bayes':
        MLP_factory = MLP

        def prediction(y): return tf.reshape(y.mean[:, 0], [-1])
        loss = bnn.regression_loss
    else:
        MLP_factory = PointMLP

        def prediction(y): return tf.reshape(y.mean[:, 0], [-1])
        loss = bnn.point_regression_loss

    mlp = MLP_factory(hypers['x_dim'], hypers['y_dim'], hypers)
    mlp = AdaptedMLP(mlp)
    mlp.make_placeholders()
    ipt = mlp.placeholders['ipt_mean']
    y = mlp(ipt)

    target = tf.placeholder(tf.float32, [None])
    mlp.placeholders['target'] = target
    global_step = tf.Variable(0, trainable=False, name='global_step')
    loss, logprob, all_surprise = loss(y, target, mlp, hypers, global_step)

    accuracy = tf.reduce_mean(tf.abs(target - prediction(y)))

    return {
        'model': mlp,
        'metrics': {
            'accuracy': accuracy, 'loss': loss,
            'logprob': logprob, 'all_surprise': all_surprise
        },
        'global_step': global_step}


def train_test(
        Xtrain,
        Ytrain,
        Xtest,
        Ytest,
        paras,
        outpath):
    train_no, x_dim = Xtrain.shape
    try:
        test_no, y_dim = Ytest.shape
    except:
        test_no = Ytest.shape
        y_dim = 1

    hypers = {
        "x_dim": x_dim,
        "y_dim": y_dim,
        "hidden_dims": paras["hidden_dims"],
        "nonlinearity": "relu",
        "adapter": {'in':paras['in'],'out':paras['out']},
        "method": "bayes",
        "style": "heteroskedastic",
        "homo_logvar_scale": 2 * np.log(0.2),
        "prior_type": [
            "empirical",
            "wider_he",
            "wider_he"],
        "n_epochs": paras['n_epochs'],
        # "batch_size": 32,
        "batch_size": train_no,
        "learning_rate": paras['learning_rate'],
        "lambda": 1.0,
        "warmup_updates": {
            'lambda': 14000.0},
        "anneal_updates": {
            'lambda': 1000.0},
        "optimizer": "adam",
        "gradient_clip": 0.1,
        "data_fraction": 1.0,
        "sections_to_run": [
            "train",
            'test']}

    data = [[Xtrain, Ytrain.reshape(-1)],
            [Xtest, Ytest.reshape(-1)]]

    restricted_training_set = u.restrict_dataset_size(
        data[0], hypers['data_fraction'])
    hypers['dataset_size'] = len(restricted_training_set[0])

    device_id = 1
    device_string = u.get_device_string(device_id)
    print(hypers)
    with tf.device(device_string):
        if True:
            model_and_metrics = make_model(hypers)

            train_op = u.make_optimizer(model_and_metrics, hypers)
            sess = u.get_session()
            saver = tf.train.Saver()

            all_summaries = []
            best_valid_accuracy = np.inf

        for epoch in range(1, hypers['n_epochs'] + 1):
            verbose = (epoch % 20 == 0)
            if verbose:
                print("Epoch %i:        " % epoch, end='')

            epoch_summary, accuracies = u.train_valid_test(
                {
                    'train': restricted_training_set,
                    'test': data[1]
                },
                sess, model_and_metrics, train_op, hypers, verbose)
            # dump log file
            all_summaries.append(epoch_summary)

            if epoch % 5000 == 0:
                saver.save(
                    sess,
                    os.path.join(
                        outpath,
                        'model.ckpt'),
                    global_step=epoch)

        with open(os.path.join(outpath, "summaries.json"), 'w') as f:
            json.dump(all_summaries, f, indent=4, cls=u.NumpyEncoder)

    return None


def run_(dataset_name, dataset_path, times, paras):
    np.random.seed(123)

    for time in range(times):
        outpath = os.path.join(dataset_path, str(time))
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        if dataset_name == 'face':
            data = FaceDataset("./dataset", 0.9)
        else:
            data = UCIDataset(dataset_name, 0.9)
        print(
            data.Xtrain.shape,
            data.Ytrain.shape,
            data.Xtest.shape,
            data.Ytest.shape)

        train_test(
            data.Xtrain,
            data.Ytrain,
            data.Xtest,
            data.Ytest,
            paras,
            outpath)

    return None


def show_(datasets, root_path, times, epoch_list, shape):
    fig = plt.figure()
    for i in range(len(datasets)):
        dataset_name = datasets[i]
        print (dataset_name)
        data_path = os.path.join(root_path, dataset_name)
        b_epoch, e_epoch = epoch_list[i]
        ax = fig.add_subplot(2, 2, i + 1)
        test_mean, test_std = pu.UCI_result_plot(
            dataset_name,
            data_path,
            times,
            ax,
            b_epoch=b_epoch,
            e_epoch=e_epoch,
            shape=shape)
        if i+1 in [1, 3]:
            ax.set_ylabel(shape)
        if i+1 in [3, 4]:
            ax.set_xlabel('Epoch')

        print(np.min(test_mean),
              0.5 * test_std[np.where(test_mean == np.min(test_mean))])

    return None


def show_face(root_path, times, b_epoch, e_epoch, shape):
    fig = plt.figure()
    data_path = os.path.join(root_path, dataset_name)
    ax = fig.add_subplot(111)
    test_mean, test_std = pu.UCI_result_plot(
        dataset_name,
        data_path,
        times,
        ax,
        b_epoch=b_epoch,
        e_epoch=e_epoch,
        shape=shape)

    ax.set_ylabel(shape)
    print (np.min(test_mean),
           0.5*test_std[np.where(test_mean == np.min(test_mean))])
    return None


if __name__ == '__main__':
    times = 20
    paras = {'conc':
                 {'in': {"scale": [[1.0]], "shift": [[0.0]]},
                  'out': {"scale": [[1.0, 0.1]], "shift": [[0.0, 1.9]]},
                  'hidden_dims': [50],
                  'learning_rate': 0.1,
                  'n_epochs': 2000,
                  'epochs': [300, 700]
                  },
             'powe':
                 {'in': {"scale": [[1.0]], "shift": [[0.0]]},
                  'out': {"scale": [[1.0, 0.02]], "shift": [[0.0, -3.5]]},
                  'hidden_dims': [50],
                  'learning_rate': 0.6,
                  'n_epochs': 2000,
                  'epochs': [760, 900]
                  },
             'yach':
                 {'in': {"scale": [[1.0]], "shift": [[0.0]]},
                  'out': {"scale": [[1.0, 0.85]], "shift": [[0.0, -1.7]]},
                  'hidden_dims': [50],
                  'learning_rate': 0.001,
                  'n_epochs': 10000,
                  'epochs': [8000, 10000]
                  },
             'prot':
                 {'in': {"scale": [[1.0]], "shift": [[0.0]]},
                  'out': {"scale": [[1.0, 0.8]], "shift": [[0.0, -3.5]]},
                  'hidden_dims': [100],
                  'learning_rate': 0.08,
                  'n_epochs': 500,
                  'epochs': [5, 20]
                  },
             'face':
                 {'in': {"scale": [[1.0]], "shift": [[0.0]]},
                  'out': {"scale": [[1.0, 0.1]], "shift": [[0.0, -1.0]]},
                  'hidden_dims': [50],
                  'learning_rate': 0.3,
                  'n_epochs': 800,
                  'epochs': [225, 400]
                  }
             }

    dataset_name = 'conc'  # Change the name.
    root_path = '/home/yunnd/project/result/DVI'  # Your path.
    run_(dataset_name, os.path.join(root_path, dataset_name), times, paras[dataset_name])

    datasets=['conc', 'powe','yach','prot']
    epoch_list = [paras[dataset]['epochs'] for dataset in datasets]
    show_(datasets, root_path, times, epoch_list, 'logprob')
    plt.show()

    show_face(root_path, times, paras['face']['epochs'][0], paras['face']['epochs'][1], 'logprob')
    plt.show()
    pass