# FactorG-cluster: a distributed matrix factorizer with stochastic gradient descent on GPU clusters

This repository is a CUDA + MPI implementation of matrix factorization with
stochastic gradient descent.

## Requirements

These are the base requirements to build and use FactorG-cluster: 

  * POSIX-standard shell
  * GNU-compatible Make
  * G++ compiler with C++11 support
  * CUDA toolkit 7.5
  * MPI compiler and runtime

## Quick start

```sh
make
cd bin
mpirun -np num-nodes mf -t train-file [-e test-file] [-c config-file]
```

