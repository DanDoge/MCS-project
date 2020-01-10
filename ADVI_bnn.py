"""
@author: yunnaidan
@time: 2019/12/10
@file: ADVI.py
"""
import pickle
import theano
floatX = theano.config.floatX
import pymc3 as pm
import numpy as np
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
sns.set_style('white')
from pymc3.theanof import set_tt_rng, MRG_RandomStreams


def construct_nn(X, Y, X_train, Y_train, hypers):
    n_hidden = hypers['n_hidden']
    sd = hypers['sd']
    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[1], n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden).astype(floatX)

    init_b_1 = np.random.randn(n_hidden).astype(floatX)
    init_b_out = np.random.randn(1).astype(floatX)

    with pm.Model() as neural_network:
        ann_input = pm.Data('ann_input', X_train)
        ann_output = pm.Data('ann_output', Y_train)

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal('w_in_1', 0, sigma=1,
                                 shape=(X.shape[1], n_hidden),
                                 testval=init_1)
        bias_1 = pm.Normal(
            'b_1', mu=0, sd=1, shape=(
                n_hidden,), testval=init_b_1)

        # Weights from hidden layer to output
        weights_1_out = pm.Normal('w_1_out', 0, sigma=1,
                                  shape=(n_hidden,),
                                  testval=init_out)
        bias_out = pm.Normal(
            'b_out', mu=0, sd=1, shape=(
                1, ), testval=init_b_out)

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1) + bias_1)
        act_out = pm.math.dot(act_1, weights_1_out) + bias_out

        sd = pm.HalfNormal('sd', sd=sd)
        target_func = pm.Normal('out', mu=act_out, sd=sd, observed=ann_output)

    return neural_network


def train(neural_network, inference_file, model_file, hypers):
    set_tt_rng(MRG_RandomStreams(42))

    with neural_network:
        inference = pm.ADVI()
        approx = pm.fit(
            n=hypers['n_sample'],
            method=inference,
            obj_optimizer=pm.adam(
                learning_rate=hypers['lr']))
        # approx = pm.fit(n=50000, method=inference, obj_optimizer=pm.adam(learning_rate=0.01))

    with open(inference_file, "wb") as f:
        pickle.dump(inference, f, pickle.HIGHEST_PROTOCOL)
    with open(model_file, "wb") as f:
        pickle.dump(approx, f, pickle.HIGHEST_PROTOCOL)

    return inference, approx


def test(
        neural_network,
        approx,
        X_test,
        Y_test,
        ppc_file,
        trace_samples=5000,
        pred_samples=5000):
    trace = approx.sample(draws=trace_samples)
    pm.set_data(
        new_data={
            'ann_input': X_test,
            'ann_output': Y_test},
        model=neural_network)
    ppc = pm.sample_posterior_predictive(
        trace,
        samples=pred_samples,
        progressbar=True,
        model=neural_network)

    with open(ppc_file, "wb") as f:
        pickle.dump(ppc, f, pickle.HIGHEST_PROTOCOL)

    return ppc