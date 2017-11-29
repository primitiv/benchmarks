#!/usr/bin/env python
# coding: utf-8

# Sequece to Sequence Learning with Neural Networks
# Sutskever et al., NIPS'14

import time
startup_time = time.time()

import sys
import random
import math

from argparse import ArgumentParser
from collections import defaultdict

from primitiv import Device, Graph, Optimizer
from primitiv import Model, Parameter, Node

from primitiv import devices as D
from primitiv import optimizers as O
from primitiv import operators as F
from primitiv import initializers as I

SRC_TRAIN = "../data/small_parallel_enja/train.en"
TRG_TRAIN = "../data/small_parallel_enja/train.ja"
SRC_TEST = "../data/small_parallel_enja/dev.en"
TRG_TEST = "../data/small_parallel_enja/dev.ja"
MAX_EPOCH = 30


class LSTM(Model):
    def __init__(self):
        self.pwxh = Parameter()
        self.pwhh = Parameter()
        self.pbh = Parameter()
        self.add_all_parameters()

    def init(self, in_size, out_size):
        self.pwxh.init([4 * out_size, in_size], I.Uniform(-0.08, 0.08))
        self.pwhh.init([4 * out_size, out_size], I.Uniform(-0.08, 0.08))
        self.pbh.init([4 * out_size], I.Uniform(-0.08, 0.08))

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

class EncoderDecoder(Model):
    def __init__(self):
        self.psrc_lookup = Parameter()
        self.ptrg_lookup = Parameter()
        self.pw = Parameter()

        self.src_lstm = LSTM()
        self.trg_lstm = LSTM()

        self.add_all_parameters()
        self.add_all_submodels()

    def init(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size):
        self.psrc_lookup.init([embed_size, src_vocab_size], I.XavierUniform())
        self.ptrg_lookup.init([embed_size, trg_vocab_size], I.XavierUniform())
        self.pw.init([trg_vocab_size, hidden_size], I.XavierUniform())

        self.src_lstm.init(embed_size, hidden_size)
        self.trg_lstm.init(embed_size, hidden_size)

    def encode(self, src):
        src_lookup = F.parameter(self.psrc_lookup)
        self.src_lstm.reset()
        for word in src:
            x = F.pick(src_lookup, word, 1)
            self.src_lstm.forward(x)

        self.trg_lookup = F.parameter(self.ptrg_lookup)
        self.w = F.parameter(self.pw)
        self.trg_lstm.reset()

    def decode_step(self, trg_word):
        x = F.pick(self.trg_lookup, trg_word, 1)
        h = self.trg_lstm.forward(x)
        return self.w @ h

    def loss(self, trg):
        losses = []
        for i in range(len(trg) - 1):
            y = self.decode_step(trg[i])
            losses.append(F.softmax_cross_entropy(y, trg[i+1], 0))
        return F.batch.mean(F.sum(losses))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', metavar="INT", type=int, default=-1,
                        help="GPU device ID (default: %(default)d)")
    parser.add_argument('src_vocab', type=int, help="source vocabulary size")
    parser.add_argument('trg_vocab', type=int, help="target vocabulary size")
    parser.add_argument('embed', type=int, help="embedding layer size")
    parser.add_argument('hidden', type=int, help="hidden layer size")
    parser.add_argument("minibatch", type=int, help="minibatch size")
    return parser.parse_args()


def make_vocab(path, size):
    if (size < 3):
        print("Vocab size should be >= 3.", file=sys.stderr)
        sys.exit(1)
    ifs = open(path, "r")

    freq = defaultdict(lambda : 0)
    for line in ifs:
        for word in line.split():
            freq[word] += 1

    vocab = {}
    vocab["<unk>"] = 0
    vocab["<bos>"] = 1
    vocab["<eos>"] = 2
    for i, (k, v) in zip(range(3, size), sorted(freq.items(), key=lambda x: -x[1])):
        vocab[k] = i
    return vocab


def count_labels(corpus):
    ret = 0
    for sent in corpus:
        ret += len(sent) - 1
    return ret


def make_batch(corpus, sent_ids, vocab):
    batch_size = len(sent_ids)
    eos_id = vocab["<eos>"]
    max_len = 0
    for sid in sent_ids:
        max_len = max(max_len, len(corpus[sid]))
    batch = [[eos_id] * batch_size for i in range(max_len)]
    for i in range(batch_size):
        sent = corpus[sent_ids[i]]
        for j in range(len(sent)):
            batch[j][i] = sent[j]
    return batch


def load_corpus(path, vocab):
    unk_id = vocab["<unk>"]
    ret = []
    for line in open(path):
        line = "<bos> " + line + " <eos>"
        sent = [vocab.get(word, unk_id) for word in line.split()]
        ret.append(sent)
    return ret


if __name__ == "__main__":
    args = parse_args()
    batchsize = args.minibatch

    if args.gpu >= 0:
        dev = D.CUDA(args.gpu)
    else:
        dev = D.Naive()
    Device.set_default(dev)

    g = Graph()
    Graph.set_default(g)

    encdec = EncoderDecoder()
    encdec.init(args.src_vocab, args.trg_vocab, args.embed, args.hidden)

    optimizer = O.SGD(0.7)
    optimizer.add_model(encdec)

    src_vocab = make_vocab(SRC_TRAIN, args.src_vocab)
    trg_vocab = make_vocab(TRG_TRAIN, args.trg_vocab)

    src_train = load_corpus(SRC_TRAIN, src_vocab)
    trg_train = load_corpus(TRG_TRAIN, trg_vocab)
    src_test = load_corpus(SRC_TEST, src_vocab)
    trg_test = load_corpus(TRG_TEST, trg_vocab)

    num_train_sents = len(trg_train)
    num_test_sents = len(trg_test)
    num_train_labels = count_labels(trg_train)
    num_test_labels = count_labels(trg_test)
    train_ids = list(range(num_train_sents))
    test_ids = list(range(num_test_sents))

    print("startup time=%r" % (time.time() - startup_time))

    for epoch in range(MAX_EPOCH):
        train_time = time.time()

        if epoch > 5:
            new_scale = 0.5 * optimizer.get_learning_rate_scaling()
            optimizer.set_learning_rate_scaling(new_scale)

        random.shuffle(train_ids)
        for ofs in range(0, num_train_sents, batchsize):
            if epoch > 5 and 0 <= (ofs - num_train_sents/2) < batchsize:
                new_scale = 0.5*optimizer.get_learning_rate_scaling()
                optimizer.set_learning_rate_scaling(new_scale)

            batch_ids = train_ids[ofs : min(ofs + batchsize, num_train_sents)]
            src_batch = make_batch(src_train, batch_ids, src_vocab)
            trg_batch = make_batch(trg_train, batch_ids, trg_vocab)

            g.clear()
            encdec.encode(src_batch)
            loss = encdec.loss(trg_batch)

            optimizer.reset_gradients()
            loss.backward()
            optimizer.update()
        train_time = time.time() - train_time

        test_loss = 0
        for ofs in range(0, num_test_sents, batchsize):
            batch_ids = test_ids[ofs : min(ofs + batchsize, num_test_sents)]
            src_batch = make_batch(src_test, batch_ids, src_vocab)
            trg_batch = make_batch(trg_test, batch_ids, trg_vocab)

            g.clear()
            encdec.encode(src_batch)
            loss = encdec.loss(trg_batch)
            test_loss += loss.to_float() * len(batch_ids)

        test_ppl = math.exp(test_loss / num_test_labels)

        print("epoch=%d, time=%.4f, ppl=%.4f, word_per_sec=%.4f" %(
            epoch + 1, train_time, test_ppl, num_train_labels / train_time))
