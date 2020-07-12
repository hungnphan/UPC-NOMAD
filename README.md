

# A implementation of NOMAD with UPC++

[![](https://travis-ci.org/joemccann/dillinger.svg?branch=master)]()

This repository presents an implementation of Nonlocking, stOchastic Multi-machine algorithm for Asynchronous and Decentralized matrix completion (NOMAD) in C++ with UPC++. The primary ideas are extracted from this paper
> Yun, Hyokun, Hsiang-Fu Yu, Cho-Jui Hsieh, S. V. N. Vishwanathan and Inderjit S. Dhillon. “NOMAD: Nonlocking, stOchastic Multi-machine algorithm for Asynchronous and Decentralized matrix completion.” Proc. VLDB Endow. 7 (2014): 975-986.
> [https://arxiv.org/abs/1312.0193](https://arxiv.org/abs/1312.0193)


## UPC++ Installation 
[![](https://bitbucket-assetroot.s3.amazonaws.com/c/photos/2015/May/07/1791043611-5-upcxx-logo_avatar.png)](https://bitbucket.org/berkeleylab/upcxx/wiki/Home)

UPC++ is a parallel programming library for developing C++ applications with the Partitioned Global Address Space (PGAS) model. UPC++ has three main objectives:
+ Provide an object-oriented PGAS programming model in the context of the popular C++ language
+ Expose useful asynchronous parallel programming idioms unavailable in traditional SPMD models, such as remote function invocation and continuation-based operation completion, to support complex scientific applications
+ Offer an easy on-ramp to PGAS programming through interoperability with other existing parallel programming systems (e.g., MPI, OpenMP, CUDA)

You can setup UPC++ as following the instruction at [here](https://bitbucket.org/berkeleylab/upcxx/wiki/INSTALL)

## Spare Matrix Generation
You can generate a random sparse matrix of integers with an assumption that there is at least one non-zero value in each column and each row:

```sh
$ g++ -o gen_sparse_mat data/generate_sparse_matrix.cpp
$ ./gen_sparse_mat no_row no_colums
```

Example: The below command will generate a sparse matrix of integers with 100 rows and 700 columns
```sh
$ g++ -o gen_sparse_mat data/generate_sparse_matrix.cpp
$ ./gen_sparse_mat 100 700
```

## NOMAD Executation
You can optionally modify the source code and build the source with UPC++ as simple commands as follow:
```sh
$ upcxx -O -o UPC-NOMAD *.cpp
```

To run this solution, you must specify the number of processes `NUM_PROC`, the input file for sparse matrix `INPUT_FILE` and the number of epochs you need to run `NUM_EPOCHS`
```sh
$ upcxx-run -n [NUM_PROC] UPC-NOMAD [NUM_EPOCHS] [INPUT_FILE]
```

For example: If we want to execute this implementation with `5` processes, the input matrix is store in `matrix.txt`, and the epoch of running is `5000`, then the command should be:
```sh
$ upcxx-run -n 5 UPC-NOMAD 5000 matrix.txt
```

The result will be stored in a output text file named: `out_[INPUT_FILE]`

### Notice
There are some slight difference in this implementation as compared to the original idea in the paper:
+ We change the update function (9) and (10) into    
  ![equ](https://latex.codecogs.com/gif.latex?w_{it}&space;\gets&space;w_{it}-s_t&space;[(w_{it}h_{jt}-A_{itjt})&space;h_{jt}+\lambda\dot&space;\|\|w_{it}\|\|])
  ![equ](https://latex.codecogs.com/gif.latex?h_{jt}&space;\gets&space;h_{jt}-s_t&space;[(w_{it}h_{jt}-A_{itjt})&space;w_{it}+\lambda\dot&space;\|\|h_{jt}\|\|])
+ Instead of transfer a pair of ![equ](https://latex.codecogs.com/gif.latex?(j,h_j)), we store all matrix ![equ](https://latex.codecogs.com/gif.latex?H) in the global memory and we only transfer the index of corresponding rows ![equ](https://latex.codecogs.com/gif.latex?j) of H 




### Todos

 - Plug-in `mmap` file reading in C++ for big file reading
 - Visualize the procedure of resources transfer and allocation


License
----
> **Free for you, Easy to use**
