# **SparseRecurrentNetwork** #

SparseRecurrentNetwork is an experimental framework for developing and testing deep, recurrent neural networks for sequence prediction. It is based on an experimental architecture that uses among others

* sparse autoencoders
* a custom cell unit containing feedforward, recurrent and feedback connections with a custom update logic
* gradient descent with momentum and adaptive gradient updates
* dropout and inhibition
* audio and text input preprocessers

It is written in Python to provide easy, simple testing-developing cycles (parallelization and cluster deployment is currently WIP).

## **Architecture** ##

### Network architecture ###
![alt tag](https://cloud.githubusercontent.com/assets/2136696/10036936/9a4d3ef4-61d8-11e5-8023-e5629d6c158b.jpg)

### Forward pass computation ###
![alt tag](https://cloud.githubusercontent.com/assets/2136696/10036939/9f38c7a8-61d8-11e5-92d8-b01e7d1e8b74.jpg)

### Backpropagation ###
![alt tag](https://cloud.githubusercontent.com/assets/2136696/10036940/9f6fc2bc-61d8-11e5-9884-9baf89d6a422.jpg)