#!/usr/bin/env python3
# coding: utf-8

import time
startup_time = time.time()

import sys
import random
import math

from argparse import ArgumentParser

from primitiv import Device, Graph, Optimizer
from primitiv import Model, Parameter, Node
from primitiv import devices as D
from primitiv import operators as F
from primitiv import initializers as I
from primitiv import optimizers as O

TRAIN_FILE = "../data/text/ptb.train.txt"
TEST_FILE  = "../data/text/ptb.test.txt"
MAX_EPOCH = 30

class LSTM(Model):
    def __init__(self):
        self.pwxh = Parameter()
        self.pwhh = Parameter()
        self.pbh = Parameter()
        self.add_all_parameters()

    def init(self, in_size, out_size):
        self.pwxh.init([4 * out_size, in_size], I.XavierUniform())
        self.pwhh.init([4 * out_size, out_size], I.XavierUniform())
        self.pbh.init([4 * out_size], I.Constant(0))

    def reset(self, init_c = Node(), init_h = Node()):
        out_size = self.pwhh.shape()[1]
        self.wxh = F.parameter(self.pwxh)
        self.whh = F.parameter(self.pwhh)
        self.bh = F.parameter(self.pbh)
        self.c = init_c if init_c.valid() else F.zeros([out_size])
        self.h = init_h if init_h.valid() else F.zeros([out_size])

    def forward(self, x):
        out_size = self.pwhh.shape()[1]
        u = self.wxh @ x + self.whh @ self.h + self.bh
        i = F.sigmoid(F.slice(u, 0, 0, out_size))
        f = F.sigmoid(F.slice(u, 0, out_size, 2 * out_size))
        o = F.sigmoid(F.slice(u, 0, 2 * out_size, 3 * out_size))
        j = F.tanh(F.slice(u, 0, 3 * out_size, 4 * out_size))
        self.c = i * j + f * self.c
        self.h = o * F.tanh(self.c)
        return self.h


class RNNLM(Model):
    def __init__(self):
        self.plookup = Parameter()
        self.lstm = LSTM()
        self.pwhy = Parameter()
        self.pby = Parameter()

        self.add_all_parameters()
        self.add_all_submodels()

    def init(self, vocab_size, embed_size, hidden_size):
        self.plookup.init([embed_size, vocab_size], I.XavierUniform())
        self.lstm.init(embed_size, hidden_size)
        self.pwhy.init([vocab_size, hidden_size], I.XavierUniform())
        self.pby.init([vocab_size], I.Constant(0))

    def forward(self, word):
        x = F.pick(self.lookup, word, 1)
        h = F.sigmoid(self.lstm.forward(x))
        return self.why @ h + self.by

    def loss(self, inputs):
        self.lookup = F.parameter(self.plookup)
        self.lstm.reset()
        self.why = F.parameter(self.pwhy)
        self.by = F.parameter(self.pby)

        losses = []
        for i in range(len(inputs)-1):
            output = self.forward(inputs[i])
            losses.append(F.softmax_cross_entropy(output, inputs[i+1], 0))
        return F.batch.mean(F.sum(losses))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', metavar="INT", type=int, default=-1,
                        help="GPU device ID (default: %(default)d)")
    parser.add_argument('embed', type=int, help="embedding layer size")
    parser.add_argument('hidden', type=int, help="hidden layer size")
    parser.add_argument("minibatch", type=int, help="minibatch size")
    return parser.parse_args()

def make_vocab(filename):
    vocab = {}
    with open(filename, "r") as ifs:
        for line in ifs:
            line = line.strip() + " <s>"
            for word in line.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
    return vocab

def load_corpus(filename, vocab):
    corpus = []
    with open(filename, "r") as ifs:
        for line in ifs:
            line = line.strip() + " <s>"
            sentence = [vocab[word] for word in line.split()]
            corpus.append(sentence)
    return corpus

def count_labels(corpus):
    ret = 0
    for sent in corpus:
        ret += len(sent) - 1
    return ret

def make_batch(corpus, sent_ids, eos_id):
    batch_size = len(sent_ids)
    max_len = 0
    for sid in sent_ids:
        max_len = max(max_len, len(corpus[sid]))
    batch = [[eos_id] * batch_size for i in range(max_len)]
    for i in range(batch_size):
        sent = corpus[sent_ids[i]]
        for j in range(len(sent)):
            batch[j][i] = sent[j]
    return batch

if __name__ == '__main__':
    args = parse_args()

    vocab = make_vocab(TRAIN_FILE)
    vocab_size = len(vocab)
    eos_id = vocab["<s>"]

    train_corpus = load_corpus(TRAIN_FILE, vocab)
    test_corpus = load_corpus(TEST_FILE, vocab)
    num_train_sents = len(train_corpus)
    num_test_sents = len(test_corpus)
    num_train_labels = count_labels(train_corpus)
    num_test_labels = count_labels(test_corpus)

    if args.gpu >= 0:
        dev = D.CUDA(args.gpu)
    else:
        dev = D.Naive()
    Device.set_default(dev)

    g = Graph()
    Graph.set_default(g)

    rnnlm = RNNLM()
    rnnlm.init(vocab_size, args.embed, args.hidden)

    optimizer = O.Adam()
    optimizer.add_model(rnnlm)

    train_ids = list(range(num_train_sents))
    test_ids = list(range(num_test_sents))

    print("startup time=%r" % (time.time() - startup_time))

    for epoch in range(MAX_EPOCH):
        train_time = time.time()

        # train_loss = 0
        random.shuffle(train_ids)
        for ofs in range(0, num_train_sents, args.minibatch):
            batch_ids = train_ids[ofs : min(ofs+args.minibatch, num_train_sents)]
            batch = make_batch(train_corpus, batch_ids, eos_id)

            g.clear()
            loss = rnnlm.loss(batch)
            # train_loss += loss.to_float() * len(batch_ids)
            optimizer.reset_gradients()
            loss.backward()
            optimizer.update()
        # train_ppl = math.exp(train_loss / num_train_labels)
        train_time = time.time() - train_time

        test_loss = 0
        for ofs in range(0, num_test_sents, args.minibatch):
            batch_ids = test_ids[ofs : min(ofs+args.minibatch, num_test_sents)]
            batch = make_batch(test_corpus, batch_ids, eos_id)

            g.clear()
            loss = rnnlm.loss(batch)
            test_loss += loss.to_float() * len(batch_ids)
        test_ppl = math.exp(test_loss / num_test_labels)

        print("epoch=%d, time=%.4f, ppl=%.4f, word_per_sec=%.4f" % (
            epoch+1, train_time, test_ppl, num_train_labels / train_time))
