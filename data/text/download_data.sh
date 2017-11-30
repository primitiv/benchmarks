#!/usr/bin/env bash

wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar xf simple-examples.tgz
mv simple-examples/data/ptb.train.txt ./
mv simple-examples/data/ptb.test.txt ./
rm -rf simple-examples{,.tgz}
