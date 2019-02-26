from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import data


# Set the random number generators' seeds for consistency
SEED = 9876
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    I do not want to shuffle the dataset. 
    Set shuffle = False
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)



def my_get_minibatches_idx(n, shuffle=False):
    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    return zip(range(len(idx_list)), idx_list)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, _p, trng):
    '''proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.8, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 1.0)'''

    proj = state_before * trng.binomial(state_before.shape,
                                        p=_p, n=1,
                                        dtype=state_before.dtype)
    return proj




def dropout_mask_1D(state, dim, _p, trng):
    return trng.binomial(size=(state.shape[dim],), p=_p, n=1, dtype=state.dtype)


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options, Wemb_value=None):
    """
    Global (not NN) parameter. For the embeding and the classifier.
    """

    rng = numpy.random.RandomState(7896)

    params = OrderedDict()


    # embeddings of tokens in sentences
    if Wemb_value is None:
        params['Wemb'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['dim_token'])),
                            high = numpy.sqrt(6. / (options['dim_token'])),
                            size=(options['n_vocb_words'], options['dim_token'])
                            )
                        )
                    ).astype(theano.config.floatX)
    else:
        params['Wemb'] = (numpy.asarray(Wemb_value)).astype(theano.config.floatX)

    # Word Embeddings with perturbations
    params['p_Wemb'] = params['Wemb'] + 1e-4



    #NN parameters
    params = param_init_lstm_0(options, params)
    params = param_init_lstm_1(options, params)
    params = param_init_attention_layer_2D_0(options, params)
    params = param_init_attention_layer_2D_1(options, params)


    # classifier softmax
    params['Ws'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['lstm_layer_0_n'] + options['lstm_layer_1_n'] + options['ydim1'])),
                            high = numpy.sqrt(6. / (options['lstm_layer_0_n'] + options['lstm_layer_1_n'] + options['ydim1'])),
                            size=(options['lstm_layer_0_n'] + options['lstm_layer_1_n'], options['ydim1'])
                            )
                        )
                    ).astype(theano.config.floatX)

    params['bs'] = numpy.zeros((options['ydim1'],)).astype(config.floatX)



    return params


'''def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params'''


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams



def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)



def init_my_param(size):

    rng = numpy.random.RandomState(7896)

    p = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (size[0] + size[1])),
                            high = numpy.sqrt(6. / (size[0] + size[1])),
                            size = size))
                    ).astype(theano.config.floatX)

    return p


def param_init_lstm_0(options, myparams, prefix='lstm_layer_0'):
    """
    Init the LSTM_0 parameter:

    :see: init_params
    """

    rng = numpy.random.RandomState(6789)

    
    size_0 = (options['dim_token'], options['lstm_layer_0_n'])
    size_1 = (options['lstm_layer_0_n'],  options['lstm_layer_0_n'])

    myparams[_p(prefix, 'Wf')] = numpy.concatenate([init_my_param(size_0),
                                                  init_my_param(size_0),
                                                  init_my_param(size_0),
                                                  init_my_param(size_0)], axis=1)

    myparams[_p(prefix, 'Uf')] = numpy.concatenate([init_my_param(size_1),
                                                  init_my_param(size_1),
                                                  init_my_param(size_1),
                                                  init_my_param(size_1)], axis=1)

    myparams[_p(prefix, 'bf')] = numpy.zeros((options['lstm_layer_0_n'] * 4,)).astype(config.floatX)

    #
    myparams[_p(prefix, 'Wb')] = numpy.concatenate([init_my_param(size_0),
                                                  init_my_param(size_0),
                                                  init_my_param(size_0),
                                                  init_my_param(size_0)], axis=1)

    myparams[_p(prefix, 'Ub')] = numpy.concatenate([init_my_param(size_1),
                                                  init_my_param(size_1),
                                                  init_my_param(size_1),
                                                  init_my_param(size_1)], axis=1)

    myparams[_p(prefix, 'bb')] = numpy.zeros((options['lstm_layer_0_n'] * 4,)).astype(config.floatX)



    myparams[_p(prefix, 'V')] = rng.normal(scale=0.01, size=(options['lstm_layer_0_n'] * 1,)).astype(config.floatX)
    ################################################################################################


    return myparams


def param_init_lstm_1(options, myparams, prefix='lstm_layer_1'):
    """
    Init the LSTM_1 parameter:

    :see: init_params
    """

    rng = numpy.random.RandomState(6789)

    
    size_0 = (options['dim_token'], options['lstm_layer_1_n'])
    size_1 = (options['lstm_layer_1_n'],  options['lstm_layer_1_n'])

    myparams[_p(prefix, 'Wf')] = numpy.concatenate([init_my_param(size_0),
                                                  init_my_param(size_0),
                                                  init_my_param(size_0),
                                                  init_my_param(size_0)], axis=1)

    myparams[_p(prefix, 'Uf')] = numpy.concatenate([init_my_param(size_1),
                                                  init_my_param(size_1),
                                                  init_my_param(size_1),
                                                  init_my_param(size_1)], axis=1)

    myparams[_p(prefix, 'bf')] = numpy.zeros((options['lstm_layer_1_n'] * 4,)).astype(config.floatX)

    #
    myparams[_p(prefix, 'Wb')] = numpy.concatenate([init_my_param(size_0),
                                                  init_my_param(size_0),
                                                  init_my_param(size_0),
                                                  init_my_param(size_0)], axis=1)

    myparams[_p(prefix, 'Ub')] = numpy.concatenate([init_my_param(size_1),
                                                  init_my_param(size_1),
                                                  init_my_param(size_1),
                                                  init_my_param(size_1)], axis=1)

    myparams[_p(prefix, 'bb')] = numpy.zeros((options['lstm_layer_1_n'] * 4,)).astype(config.floatX)


    myparams[_p(prefix, 'V')] = rng.normal(scale=0.01, size=(options['lstm_layer_1_n'] * 1,)).astype(config.floatX)

    ################################################################################################

    return myparams



def param_init_attention_layer_2D_0(options, myparams, prefix='att_layer_0'):
    '''
    input : 2D
    output: 1D
    '''

    rng = numpy.random.RandomState(7498)

    myparams[_p(prefix, 'V')] = rng.normal(scale=0.01, size=(options['lstm_layer_0_n'] * 1,)).astype(config.floatX)

    return myparams



def param_init_attention_layer_2D_1(options, myparams, prefix='att_layer_1'):
    '''
    input : 2D
    output: 1D
    '''

    rng = numpy.random.RandomState(3445)

    myparams[_p(prefix, 'V')] = rng.normal(scale=0.01, size=(options['lstm_layer_1_n'] * 1,)).astype(config.floatX)

    return myparams



def lstm_layer_0(tparams, input_state, mask, options, prefix='lstm_layer_0'):

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step_f(m_, x_, h_, c_):
        preact = tensor.dot(x_, tparams[_p(prefix, 'Wf')]) + tparams[_p(prefix, 'bf')] + \
                 tensor.dot(h_, tparams[_p(prefix, 'Uf')])

        i = tensor.nnet.sigmoid(_slice(preact, 0, options[_p(prefix, 'n')]))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options[_p(prefix, 'n')]))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options[_p(prefix, 'n')]))
        c = tensor.tanh(_slice(preact, 3, options[_p(prefix, 'n')]))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    def _step_b(m_, x_, h_, c_):
        preact = tensor.dot(x_, tparams[_p(prefix, 'Wb')]) + tparams[_p(prefix, 'bb')] + \
                 tensor.dot(h_, tparams[_p(prefix, 'Ub')])

        i = tensor.nnet.sigmoid(_slice(preact, 0, options[_p(prefix, 'n')]))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options[_p(prefix, 'n')]))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options[_p(prefix, 'n')]))
        c = tensor.tanh(_slice(preact, 3, options[_p(prefix, 'n')]))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c


    dim_proj = options[_p(prefix, 'n')]
    ##############################################################################################
    rval_f, updates_f = theano.scan(_step_f,
                                sequences=[mask, input_state],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), input_state.shape[1], dim_proj),
                                              tensor.alloc(numpy_floatX(0.), input_state.shape[1], dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=input_state.shape[0])

    rval_b, updates_b = theano.scan(_step_b,
                                sequences=[mask, input_state],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), input_state.shape[1], dim_proj),
                                              tensor.alloc(numpy_floatX(0.), input_state.shape[1], dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=input_state.shape[0],
                                go_backwards=True)


    proj_0 = rval_f[0] + rval_b[0][::-1]
    #proj_0 = rval_f[0]

    # Attention
    y_0 = (tensor.tanh(proj_0) * mask[:, :, None]) * tparams[_p(prefix, 'V')]
    y_0 = y_0.sum(axis=2).transpose()
    alpha = tensor.nnet.softmax(y_0).transpose()
    proj_0 = proj_0 * alpha[:, :, None]#(proj_0 * mask[:, :, None])

    proj_0 = proj_0.sum(axis=0)#(proj_0 * mask[:, :, None])
    ##############################################################################################

    # max pooling
    #proj_0 = rval_f[0]
    #proj_0 = proj_0 * mask[:, :, None]
    #proj_0 = tensor.max(proj_0, axis=0)

    proj_0 = tensor.tanh(proj_0)
    
     
    return proj_0


def lstm_layer_1(tparams, input_state, mask, options, prefix='lstm_layer_1'):

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step_f(m_, x_, h_, c_):
        preact = tensor.dot(x_, tparams[_p(prefix, 'Wf')]) + tparams[_p(prefix, 'bf')] + \
                 tensor.dot(h_, tparams[_p(prefix, 'Uf')])

        i = tensor.nnet.sigmoid(_slice(preact, 0, options[_p(prefix, 'n')]))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options[_p(prefix, 'n')]))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options[_p(prefix, 'n')]))
        c = tensor.tanh(_slice(preact, 3, options[_p(prefix, 'n')]))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    def _step_b(m_, x_, h_, c_):
        preact = tensor.dot(x_, tparams[_p(prefix, 'Wb')]) + tparams[_p(prefix, 'bb')] + \
                 tensor.dot(h_, tparams[_p(prefix, 'Ub')])

        i = tensor.nnet.sigmoid(_slice(preact, 0, options[_p(prefix, 'n')]))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options[_p(prefix, 'n')]))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options[_p(prefix, 'n')]))
        c = tensor.tanh(_slice(preact, 3, options[_p(prefix, 'n')]))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c


    dim_proj = options[_p(prefix, 'n')]
    ##############################################################################################
    rval_f, updates_f = theano.scan(_step_f,
                                sequences=[mask, input_state],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), input_state.shape[1], dim_proj),
                                              tensor.alloc(numpy_floatX(0.), input_state.shape[1], dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=input_state.shape[0])

    rval_b, updates_b = theano.scan(_step_b,
                                sequences=[mask, input_state],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), input_state.shape[1], dim_proj),
                                              tensor.alloc(numpy_floatX(0.), input_state.shape[1], dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=input_state.shape[0],
                                go_backwards=True)


    proj_0 = rval_f[0] + rval_b[0][::-1]
    #proj_0 = rval_f[0]

    # Attention
    y_0 = (tensor.tanh(proj_0) * mask[:, :, None]) * tparams[_p(prefix, 'V')]
    y_0 = y_0.sum(axis=2).transpose()
    alpha = tensor.nnet.softmax(y_0).transpose()
    proj_0 = proj_0 * alpha[:, :, None]#(proj_0 * mask[:, :, None])

    proj_0 = proj_0.sum(axis=0)#(proj_0 * mask[:, :, None])
    ##############################################################################################

    #proj_0 = rval_f[0]
    #proj_0 = proj_0 * mask[:, :, None]
    #proj_0 = tensor.max(proj_0, axis=0)

    proj_0 = tensor.tanh(proj_0)
    
     
    return proj_0




def attention_layer_2D_0(tparams, input_state, options, prefix='att_layer_0'):

    y_0 = tensor.tanh(input_state) * tparams[_p(prefix, 'V')]
    y_0 = y_0.sum(axis=1)
    alpha = tensor.nnet.softmax(y_0).flatten()

    proj_0 = input_state * alpha[:, None]
    p_0 = proj_0.sum(axis=0)
    p_0 = tensor.tanh(p_0)

    #p_0 = p_0.flatten()

    return p_0


def attention_layer_2D_1(tparams, input_state, options, prefix='att_layer_1'):

    y_0 = tensor.tanh(input_state) * tparams[_p(prefix, 'V')]
    y_0 = y_0.sum(axis=1)
    alpha = tensor.nnet.softmax(y_0).flatten()

    proj_0 = input_state * alpha[:, None]
    p_0 = proj_0.sum(axis=0)
    p_0 = tensor.tanh(p_0)

    #p_0 = p_0.flatten()

    return p_0



def sgd(lr, tparams, grads, x, masks, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function(x + masks + [y], 
                                    cost, updates=gsup, name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, masks, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(x + masks + [y], 
                                    cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, masks, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(x + masks + [y], 
                                    cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update




def momentum(lr, tparams, grads, x, masks, y, cost):
    
    rho = 0.9

    pre_step = [theano.shared(p.get_value() * 0., name='%s_pre_step' % k, 
                broadcastable = p.broadcastable)
               for k, p in tparams.items()]

    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function(x + masks + [y], 
                                    cost, 
                                    updates=gsup,
                                    name='momentum_sgd_f_grad_shared')


    step = [rho * ps - lr * g for ps, g in zip(pre_step, gshared)]

    sup = [(ps, s) for ps, s in zip(pre_step, step)]

    pup = [(p, p + s) for p, s in zip(tparams.values(), step)]


    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup+sup,
                               name='sgd_f_update')

    return f_grad_shared, f_update



def nesterov_momentum(lr, tparams, grads, x, masks, y, cost):
    
    rho = 0.9

    pre_step = [theano.shared(p.get_value() * 0., name='%s_pre_step' % k, 
                broadcastable = p.broadcastable)
               for k, p in tparams.items()]

    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function(x + masks + [y], 
                                    cost, 
                                    updates=gsup,
                                    name='sgd_f_grad_shared')


    step = [rho * ps - lr * g for ps, g in zip(pre_step, gshared)]

    sup = [(ps, s) for ps, s in zip(pre_step, step)]

    pup = [(p, p + rho * rho * s - (1+rho) * lr * g) for p, s, g in zip(tparams.values(), step, gshared)]


    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup+sup,
                               name='sgd_f_update')

    return f_grad_shared, f_update






def build_model(tparams, options):

    trng = RandomStreams(SEED)

    # Discriminator
    x0 = tensor.matrix('x0', dtype='int32') #
    x1 = tensor.matrix('x1', dtype='int32') #

    mask0 = tensor.matrix('mask0', dtype=config.floatX)
    mask1 = tensor.matrix('mask1', dtype=config.floatX)

    y0 = tensor.vector('y0', dtype='int32')

    
    #dropout_ratio = tensor.scalar(name='dropout_ratio')
    #dropout_decay_ratio = tensor.scalar(name='dropout_decay_ratio')   

    #####################################
    # 
    p_0 = lstm_layer_0(tparams, input_state=tparams['Wemb'][x0], mask=mask0, options=options)
    p_1 = lstm_layer_1(tparams, input_state=tparams['Wemb'][x1], mask=mask1, options=options)
    

    #p_0 = tensor.max(p_0, axis=0)
    #p_1 = tensor.max(p_1, axis=0)
    p_0 = attention_layer_2D_0(tparams, input_state=p_0,  options=options)
    p_1 = attention_layer_2D_1(tparams, input_state=p_1,  options=options)
    

    proj_0 = tensor.concatenate((p_0, p_1), axis=0)
   
    #proj_0 = proj_0 * dropout_mask_1D(proj_0, 1, dropout_ratio, trng) * dropout_decay_ratio
    
    pred_0 = tensor.nnet.softmax(tensor.dot(proj_0, tparams['Ws'])+ tparams['bs']) 
    pred_0 = pred_0.flatten()


    f_pred_prob = theano.function(inputs=[x0, x1, mask0, mask1], 
                                    outputs=pred_0.max(axis=0), 
                                    name='f_pred_prob')

    f_pred = theano.function(inputs=[x0, x1,  mask0, mask1],
                               outputs=pred_0.argmax(axis=0), 
                               name='f_pred')

    off = 1e-8

    cost = -tensor.mean(tensor.log(pred_0[y0[1]] + off))

    #####################################

    adv_p_0 = lstm_layer_0(tparams, input_state=tparams['p_Wemb'][x0], mask=mask0, options=options)
    adv_p_1 = lstm_layer_1(tparams, input_state=tparams['p_Wemb'][x1], mask=mask1, options=options)

    adv_p_0 = attention_layer_2D_0(tparams, input_state=adv_p_0,  options=options)
    adv_p_1 = attention_layer_2D_1(tparams, input_state=adv_p_1,  options=options)

    adv_proj_0 = tensor.concatenate((adv_p_0, adv_p_1), axis=0)

    adv_pred_0 = tensor.nnet.softmax(tensor.dot(adv_proj_0, tparams['Ws'])+ tparams['bs']) 
    adv_pred_0 = adv_pred_0.flatten()

    f_adv_pred_prob = theano.function(inputs=[x0, x1, mask0, mask1], 
                                    outputs=adv_pred_0.max(axis=0), 
                                    name='f_adv_pred_prob')

    f_adv_pred = theano.function(inputs=[x0, x1,  mask0, mask1],
                               outputs=adv_pred_0.argmax(axis=0), 
                               name='f_adv_pred')

    adv_cost = -tensor.mean(tensor.log(adv_pred_0[0] + off))

    cost = (cost + adv_cost) * 0.5

    #####################################

    return [x0,x1], [mask0, mask1], y0, \
           f_pred_prob, f_pred, \
           f_adv_pred_prob, f_adv_pred, \
           cost, adv_cost



def output_pred_labels(options, f_pred, f_pred_prob, prepare_data, data, iterator, verbose, path):
    f = open(path,'w')
    for _, test_index in iterator:
        x0, mask0, _ = prepare_data(data[0][test_index])
        x1, mask1, _ = prepare_data(data[1][test_index])
        
        pred_labels = f_pred(x0, x1,  mask0, mask1)
        pred_maxProbs = f_pred_prob(x0, x1,  mask0,  mask1)

        f.write(str(pred_labels)+' '+str(pred_maxProbs)+'\n')

    f.close()



def train_nn(

    # Hyper-Parameters

    dim_token = 100,  # word embeding dimension
    
    lstm_layer_0_n = 50,
    lstm_layer_1_n = 50,
  
    ydim0 = 5,
    ydim1 = 6,

    #win_size = 3,

    

    #n_cueTypes = 4,
    n_vocb_words = 31848,  # Vocabulary size
    #n_locDiffs = 111,  # Location difference size

    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=100,  # The maximum number of epoch to run
    #dispFreq=10,  # Display to stdout the training progress every N updates
    #decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.01,  # Learning rate for sgd (not used for adadelta and rmsprop)
    dropout_p = 1.0,
    adv_epsilon = 0.001,
    
    optimizer = momentum,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).


    #maxlen=1000,  # Sequence longer then this get ignored
    batch_size=10,  # The batch size during training.
    #inter_cost_margin = 0.001,


    # Parameter for extra option
    #noise_std=0.,
    #use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    #reload_model=None,  # Path to a saved model we want to start from.
    #test_size=-1
):

    

    # Model options
    model_options = locals().copy()
    print('-------------------------------------------------------------')
    print("model options", model_options)
    print('-------------------------------------------------------------')

    #load_data, prepare_data = get_dataset(dataset)

    print('Loading data ... ... ...')
    train, valid, test = data.load_data(path='../mydata.pkl.gz',n_words=n_vocb_words)
   
    
    print('Building model ... ... ...')

    params = init_params(model_options, Wemb_value=data.read_gz_file("../../matrix.pkl.gz"))
    
    tparams = init_tparams(params)

    
    (x,
     masks,
     y,
     f_pred_prob,
     f_pred,
     f_adv_pred_prob,
     f_adv_pred,
     cost,
     adv_cost) = build_model(tparams, model_options)


    #f_cost = theano.function([x[0], x[1], masks[0], masks[1], y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    #f_grad = theano.function([x[0], x[1], masks[0], masks[1], y], grads, name='f_grad')

    adv_grads = tensor.grad(cost, wrt=tparams['Wemb'])
    f_adv_grad = theano.function([x[0], x[1], masks[0], masks[1], y], adv_grads, name='f_adv_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, masks, y, cost)



    print('training ... ... ...')

    kf_valid = my_get_minibatches_idx(len(valid[0]))
    kf_test  = my_get_minibatches_idx(len(test[0]))

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    #history_errs = []
    best_p = None
    bad_counter = 0
    stop_counter = 0

    #if validFreq == -1:
        #validFreq = len(train[0]) // batch_size
    #if saveFreq == -1:
        #saveFreq = len(train[0]) // batch_size

    last_ave_of_train_costs = numpy.inf
    costs_list = []
  
    uidx = 0  # the number of update done
    estop = False  # early stop
    #start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = my_get_minibatches_idx(len(train[0]), shuffle=True)


            #training_sum_costs = 0

            #ave_of_g_costs_sum = 0
            #ave_of_d_costs_sum = 0

            for train_batch_idx, train_index in kf:
                #uidx += 1
                #use_noise.set_value(1.)

                                
                # Select the random examples for this minibatch
                x_0 = train[0][train_index]
                x_1 = train[1][train_index]
                y_0 = train[2][train_index]
            
                x_0, mask_0, _ = data.prepare_data(x_0)
                x_1, mask_1, _ = data.prepare_data(x_1)
                
                y_0 = numpy.asarray(y_0, dtype='int32')

                
                cost = f_grad_shared(x_0, x_1, mask_0, mask_1, y_0)
                costs_list.append(cost)
                f_update(lrate)

                cur_adv_grad = f_adv_grad(x_0, x_1, mask_0, mask_1, y_0)
                tparams['p_Wemb'] = adv_epsilon * cur_adv_grad / (tensor.sqrt(cur_adv_grad**2+1e-4))


                if train_batch_idx % 100 == 0 or train_batch_idx == len(kf) - 1:
                    print("---Now %d/%d training bacthes @ epoch = %d" % (train_batch_idx, len(kf), eidx))

                                

            cur_ave_of_train_costs = sum(costs_list) / len(costs_list)
            print("cur_ave_of_train_costs = ",cur_ave_of_train_costs,"@ epoch = ", eidx)

            if numpy.isnan(cur_ave_of_train_costs) or numpy.isinf(cur_ave_of_train_costs):
                print('bad cost detected: ', cur_ave_of_train_costs)
                print('End of Program')
                break

            print('outputing predicted labels of test set ... ... ...')
            output_pred_labels(model_options,
                               f_pred, f_pred_prob,
                               data.prepare_data, test, kf_test, 
                               verbose=False, path="test_pred_labels.txt")

            if cur_ave_of_train_costs >= last_ave_of_train_costs * 0.9:
                stop_counter += 1

            last_ave_of_train_costs = cur_ave_of_train_costs

            print('counter for early stopping : %d/%d' % (stop_counter, patience))
            print('learning rate in this epoch = ', lrate)
            print('--------------------------------------------------')

            del costs_list[:]

            
            if stop_counter >= patience:
                print('Early Stop!')
                estop = True
                break

            if estop:
                break


    except KeyboardInterrupt:
        print("Training interupted")



if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_nn()
