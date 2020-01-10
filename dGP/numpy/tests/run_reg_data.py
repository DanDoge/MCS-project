# python run_reg_sep.py -d dataset -hi no_hiddens -m no_pseudos -i no_iterations -b batch_size
# for example, for boston housing dataset, remember to check path to data, hard-coded in this file
# python run_reg_sep.py -d boston -hi 2 -m 50 -i 5000 -b 50
import sys
sys.path.append('../code/')
import os
import math
import numpy as np
import AEPDGP_net
from tools import *
from dataset.UCIdataset import UCIDataset
from dataset.Facedataset import FaceDataset
import argparse
import time

parser = argparse.ArgumentParser(description='run regression experiment',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset',
            action="store", dest="dataset",
            help="dataset name, eg. boston, power", default="boston")
parser.add_argument('-hi', '--hiddens', nargs='+', type=int,
            action="store", dest="n_hiddens",
            help="number of hidden dimensions, eg. 2 or 5 2", default=[])
parser.add_argument('-m', '--pseudos', type=int,
            action="store", dest="n_pseudos",
            help="number of pseudo points per layer, eg. 10", default=10)
parser.add_argument('-b', '--minibatch', type=int,
            action="store", dest="minibch_size",
            help="minibatch size, eg. 10", default=50)
parser.add_argument('-i', '--iterations', type=int,
            action="store", dest="n_iterations",
            help="number of stochastic updates, eg. 10", default=1000)
parser.add_argument('-s', '--seed', type=int,
            action="store", dest="random_seed",
            help="random seed, eg. 10", default=123)
parser.add_argument('-l', '--lrate', type=float,
            action="store", dest="lrate",
            help="adam learning rate", default=0.005)
parser.add_argument('-t', '--tied',
            action="store_true", dest="tied",
            help="tying inducing point (boolean)", default=False)

args = parser.parse_args()

name = args.dataset
n_hiddens = args.n_hiddens
n_hiddens_str = '_'.join(map(str, n_hiddens))
nolayers = len(n_hiddens) + 1
M = args.n_pseudos
n_pseudos = [M for _ in range(nolayers)]
no_iterations = args.n_iterations
no_points_per_mb = args.minibch_size
random_seed = args.random_seed
np.random.seed(random_seed)
lrate = args.lrate
tied = args.tied

fnames = {'boston': 'bost',
          'power': 'powe',
          'concrete': 'conc',
          'energy': 'ener',
          'kin8nm': 'kin8',
          'naval': 'nava',
          'protein': 'prot',
          'wine_red': 'wine',
          'yacht': 'yach',
          'year': 'YearPredictionMSD'}

if name == 'face':
    data = FaceDataset(ratio=0.9)
else:
    data = UCIDataset(name, ratio=0.5)


# prepare output files

for i in range(1):
    outname1 = './tmp/' + name + '_' + n_hiddens_str + '_' + str(M) + '_' + str(i) + '.rmse'
    if not os.path.exists(os.path.dirname(outname1)):
        os.makedirs(os.path.dirname(outname1))
    outfile1 = open(outname1, 'w')
    outname2 = './tmp/' + name + '_' + n_hiddens_str + '_' + str(M) + '_' + str(i) + '.nll'
    outfile2 = open(outname2, 'w')
    outname3 = './tmp/' + name + '_' + n_hiddens_str + '_' + str(M) + '_' + str(i) + '.time'
    outfile3 = open(outname3, 'w')
    X_train = data.Xtrain
    y_train = data.Ytrain
    X_test = data.Xtest
    y_test = data.Ytest

    # We construct the network
    net = AEPDGP_net.AEPDGP_net(X_train, y_train, n_hiddens, n_pseudos, lik='Gaussian', zu_tied=tied)
    # train
    no_epochs = no_iterations
    test_nll, test_rms, logZ = net.train(X_test, y_test, no_epochs=no_epochs,
                                   no_points_per_mb=no_points_per_mb,
                                   lrate=lrate, compute_test=True,
                                   file_time=outfile3, file_rmse=outfile1, file_llh=outfile2)

    outfile1.close()
    outfile2.close()
    outfile3.close()
