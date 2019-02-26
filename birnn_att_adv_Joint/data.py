from __future__ import print_function
from six.moves import xrange
import six.moves.cPickle as pickle

import gzip
import os

import numpy
import theano

import StringIO


def read_data_xy(readfilename):
    f = open(readfilename,"r")#.txt file

    x0 = []#list of list of list
    x1 = []#list of list of list
    y0 = []#list of list

    xs0 = []#list of list of list
    xs1 = []#list of list of list
    ys0 = []#list of list

    allLines = f.readlines()
    f.close()

    i = 0
    while i < len(allLines):
        y0.append(map(int, allLines[i].split(' ')))
        i += 1

        x0_oneSample = []
        x1_oneSample = []

        i_sents_0 = int(allLines[i])
        i += 1

        for m in range(i_sents_0):
            x0_oneSample.append(map(int, allLines[i].split(' ')))
            i += 1
        
        i_sents_1 = int(allLines[i])
        i += 1

        for n in range(i_sents_1):
            x1_oneSample.append(map(int, allLines[i].split(' ')))
            i += 1

        x0.append(x0_oneSample)
        x1.append(x1_oneSample)

        ####
        xs0_oneDoc = []
        xs1_oneDoc = []
        ys0_oneDoc = []

        i_events = int(allLines[i])
        i += 1

        for m0 in range(i_events):
            ys0_oneDoc.append(map(int, allLines[i].split(' ')))
            i += 1

            xs0_oneDoc.append(map(int, allLines[i].split(' ')))
            i += 1

            xs1_oneDoc.append(map(int, allLines[i].split(' ')))
            i += 1

        xs0.append(xs0_oneDoc)
        xs1.append(xs1_oneDoc)
        ys0.append(ys0_oneDoc)


    return x0, x1, y0, xs0, xs1, ys0


def produce_data(readfilenames,savefilename):
    '''.txt files'''
    tr_x0, tr_x1, tr_y0, tr_xs0, tr_xs1, tr_ys0  = read_data_xy(readfilename[0])
    va_x0, va_x1, va_y0, va_xs0, va_xs1, va_ys0  = read_data_xy(readfilename[1])
    te_x0, te_x1, te_y0, te_xs0, te_xs1, te_ys0  = read_data_xy(readfilename[2])


    data = ((tr_x0, tr_x1, tr_y0, tr_xs0, tr_xs1, tr_ys0),
            (va_x0, va_x1, va_y0, va_xs0, va_xs1, va_ys0),
            (te_x0, te_x1, te_y0, te_xs0, te_xs1, te_ys0))#tuple

    f = open(savefilename,'wb')
    pickle.dump(data,f)
    f.close()  


    

def prepare_data(seqs, addIdxNum=0, maxlen=None, win_size=1):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    '''if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None'''

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    '''
    n_samples : numbers of sentences
    '''

    x = numpy.zeros((maxlen, n_samples)).astype('int32')
    x_mask = numpy.zeros(((maxlen - addIdxNum) / win_size, n_samples)).astype(theano.config.floatX)

    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:((lengths[idx] - addIdxNum) / win_size), idx] = 1.

    #labels = numpy.asarray(labels).astype('int32')

    return x, x_mask, maxlen - addIdxNum


def load_data(path, n_words, valid_portion=0.0, maxlen=None,
              sort_by_len=False):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset 
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

   

    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()

    '''if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            #if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y'''


    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    '''test_set_x0, test_set_x1, test_set_y0 = test_set
    valid_set_x0, valid_set_x1,  valid_set_y0 = valid_set
    train_set_x0, train_set_x1,  train_set_y0 = train_set'''

    '''train_set_x0 = remove_unk(train_set_x0)
    train_set_x1 = remove_unk(train_set_x1)
    
    valid_set_x0 = remove_unk(valid_set_x0)
    valid_set_x1 = remove_unk(valid_set_x1)

    test_set_x0 = remove_unk(test_set_x0)
    test_set_x1 = remove_unk(test_set_x1)'''

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    '''if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x0 = [test_set_x0[i] for i in sorted_index]
        test_set_x1 = [test_set_x1[i] for i in sorted_index]
        test_set_y0 = [test_set_y0[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x0 = [valid_set_x0[i] for i in sorted_index]
        valid_set_x1 = [valid_set_x1[i] for i in sorted_index]
        valid_set_y0 = [valid_set_y0[i] for i in sorted_index]


        sorted_index = len_argsort(train_set_x)
        train_set_x0 = [train_set_x0[i] for i in sorted_index]
        train_set_x1 = [train_set_x1[i] for i in sorted_index]
        train_set_y0 = [train_set_y0[i] for i in sorted_index]

    train = (train_set_x0, train_set_x1, train_set_y0)
    valid = (valid_set_x0, valid_set_x1, valid_set_y0)
    test  = (test_set_x0, test_set_x1,  test_set_y0)'''

    return train_set, valid_set, test_set

def read_embedding_file_to_get_matrix(filename, savefilename):
    file_obj = open(filename,"r")
    embeddings = []
    
    for tmp_line in file_obj:
         one_embedding = numpy.loadtxt(StringIO.StringIO(tmp_line))#matrix
         embeddings.append(one_embedding)

    matrix = numpy.asarray(embeddings)

    file_obj.close()

    f = open(savefilename,'wb')
    pickle.dump(matrix,f)
    f.close()

    return matrix

def read_gz_file(filename):
    f = gzip.open(filename,'rb')
    data = pickle.load(f)
    f.close()

    return data


if __name__ == '__main__':
    
    #data = read_data_xy("../valid_idx.txt")
    #print(len(data[0]))

    ##############################################################

    readfilename = ["../train_idx.txt",
                    "../valid_idx.txt",
                    "../test_idx.txt"]
    savefilename = '../mydata.pkl'
    produce_data(readfilename,savefilename)


    m_arr = read_embedding_file_to_get_matrix("../word_embed.txt",
                                              "../../matrix.pkl")
    print(m_arr.shape)
