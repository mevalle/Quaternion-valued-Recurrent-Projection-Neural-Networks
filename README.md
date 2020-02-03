# Quaternion-valued Recurrent Projection Neural Networks on Unit Quaternions

Quaternion-valued recurrent projection neural networks (QRPNNs) are obtained by combining the non-local projection learning with the quaternion-valued recurrent correlation neural network (QRCNNs).
QRPNNs overcome the cross-talk problem of QRCNNs and are appropriate to implement associative memories. 
Furthermore, computational experiments reveal that QRPNNs exhibit greater storage capacity and noise tolerance than their corresponding QRCNNs.

## Getting Started

This repository contain the Julia source-codes of QRPNNs on unit quaternions, as described in the paper "Quaternion-Valued Recurrent Projection NeuralNetworks on Unit Quaternions" by Marcos Eduardo Valle and Rodolfo Lobo (see https://arxiv.org/abs/2001.11846). The Jupyter-notebook of the computational experimens are also available in this repository.

## Usage:

First of all, call the QRPNN module using: 
```
include("QRPNN.jl")
```
  
Define a real-valued excitation function f. Examples of excitation functions are provided in the QRPNN.jl including:
```
QRPNN.identity, QRPNN.exponential, QRPNN.potential, and QRPNN.high_order
```

Given a quaternion-valued matrix U=[u1,...,up] of size NxP whose columns correspond to the fundamental memories, train the QRPNN using the command: 
```
V = QRPNN.train(f,f_params,U)
```
where f_params are the parameters of f.

Given an input quaternion-valued vector x of length N, the output is given by 
```
y = QRPNN.main(f,f_parameters, U, V, x, it_max, verbose)
```
where it_max (default is it_max = 1000) is the maximum number of iterations and verbose (default is true) informs if the maximum number of iterations has been reached.

For example, the exponential quaternion-valued recurrent projection neural network with parameter alpha = 10 adn the default maximum number of iteration is called using the command:
```
y = QRPNN.main(QRPNN.exponential, 10, U, V, x)
```
Remark: Quaternion-valued Recurrent Correlation Neural Network (see https://doi.org/10.1109/IJCNN.2018.8489714) is obtained by considering V = U (without training).

## Authors

* **Marcos Eduardo Valle and Rodoldo Lobo** - *University of Campinas*



