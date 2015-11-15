# **SparseRecurrentNetwork** #
SparseRecurrentNetwork is an experimental framework for developing and testing deep, recurrent neural networks for sequence prediction. It relies on Tensor Factorization based optimization:
### Unsupervised Feature Tensor Creation ###
![recog1 001](https://cloud.githubusercontent.com/assets/2136696/11167923/40e2d264-8bab-11e5-95f0-234675d96e98.png)

### Optimizing by Tensor Factorization ###
![recog2 001](https://cloud.githubusercontent.com/assets/2136696/11167924/40e36d28-8bab-11e5-9d5a-bb59c02bf088.png)

The experimental architecture uses further among others

* sparse autoencoders
* a custom cell unit containing feedforward, recurrent and feedback connections with a custom update logic
* gradient descent with momentum and adaptive gradient updates
* dropout and inhibition
* audio and text input preprocessers

It is written in Python using Google's TensorFlow to provide easy, simple developing-testing cycles (parallelization and cluster deployment is currently WIP as well as development in Scala/Spark). It is part of [ReCog Technologies](http://recog-technologies.com)

## **Simplified Architecture Visualizations** ##

### Network architecture ###
![alt tag](https://cloud.githubusercontent.com/assets/2136696/10036936/9a4d3ef4-61d8-11e5-8023-e5629d6c158b.jpg)

### Forward pass computation ###
![alt tag](https://cloud.githubusercontent.com/assets/2136696/10036939/9f38c7a8-61d8-11e5-92d8-b01e7d1e8b74.jpg)

### Backpropagation ###
![alt tag](https://cloud.githubusercontent.com/assets/2136696/10036940/9f6fc2bc-61d8-11e5-9884-9baf89d6a422.jpg)
