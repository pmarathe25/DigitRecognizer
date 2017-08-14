#!/bin/bash
./bin/createMinibatches ./data/raw/train-images.idx3-ubyte ./data/raw/train-labels.idx1-ubyte ./data/training/ 100
./bin/createMinibatches ./data/raw/t10k-images.idx3-ubyte ./data/raw/t10k-labels.idx1-ubyte ./data/testing/ 10
