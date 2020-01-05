import os
import json
import numpy as np
import matplotlib.pyplot as plt

def toy_results_plot(data, data_generator, hypers=None, predictions=None):   
    train_x = np.arange(np.min(data[0][0].reshape(-1)),
                        np.max(data[0][0].reshape(-1)), 1/100)
    
    # plot the training data distribution
    plt.plot(train_x, data_generator['mean'](train_x), 'red', label='data mean')
    plt.fill_between(train_x,
                     data_generator['mean'](train_x) - data_generator['std'](train_x),
                     data_generator['mean'](train_x) + data_generator['std'](train_x),
                     color='orange', alpha=1, label='data 1-std')
    plt.plot(data[0][0], data[0][1], 'r.', alpha=0.2, label='train sampl')
     
    # plot the model distribution
    if predictions is not None:
        x = predictions[0]
        y_mean   = predictions[1]['mean'][:,0]
        ell_mean = predictions[1]['mean'][:,1]
        y_var    = predictions[1]['cov'][:,0,0]
        ell_var  = predictions[1]['cov'][:,1,1]
        
        if hypers['style'] != 'heteroskedastic':
            ell_mean = hypers["homo_logvar_scale"]
            ell_var = 0

        heteroskedastic_part = np.exp(0.5 * ell_mean)
        full_std = np.sqrt(y_var + np.exp(ell_mean + 0.5 * ell_var))

        plt.plot(x, y_mean, label='model mean')
        plt.fill_between(x,
                         y_mean - heteroskedastic_part,
                         y_mean + heteroskedastic_part,
                         color='g', alpha = 0.2, label='$\ell$ contrib')
        plt.fill_between(x,
                         y_mean - full_std,
                         y_mean + full_std,
                         color='b', alpha = 0.2, label='model 1-std')
        
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim([-3,2])
    plt.legend()

def UCI_result_plot(
        dataset_name,
        summary_path,
        times,
        axes,
        train=False,
        test=True,
        b_epoch=0,
        e_epoch=100,
        shape='logprob'):

    length = e_epoch - b_epoch
    train_lst = []
    test_lst = []
    for time in range(times):
        summary_file = os.path.join(summary_path, str(time), 'summaries.json')
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        train_log = []
        test_log = []
        for itera in summary[b_epoch:e_epoch]:
            train_log.append(itera['train'][0][0][shape])
            test_log.append(itera['test'][0][0][shape])
        train_lst.append(train_log)
        test_lst.append(test_log)
    train_mean = np.mean(np.array(train_lst), axis=0)
    test_mean = np.mean(np.array(test_lst), axis=0)
    train_std = np.std(np.array(train_lst), axis=0)
    test_std = np.std(np.array(test_lst), axis=0)

    epoches = np.arange(b_epoch, length + b_epoch, 1)
    if train:
        axes.plot(epoches, train_mean, label='train')
        axes.fill_between(
            epoches,
            train_mean -
            0.5 *
            train_std,
            train_mean +
            0.5 *
            train_std, alpha=0.2)
    if test:
        axes.plot(epoches, test_mean, label='test')
        axes.fill_between(
            epoches,
            test_mean -
            0.5 *
            test_std,
            test_mean +
            0.5 *
            test_std, alpha=0.2)

    axes.set_title(dataset_name)

    return test_mean, test_std