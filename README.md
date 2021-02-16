# Deployed Machine Learning

[![Test](https://github.com/JFL110/deployed-ml/workflows/Test/badge.svg)](https://github.com/JFL110/deployed-ml/actions?query=workflow%3ATest)
[![Test & Deploy](https://github.com/JFL110/deployed-ml/workflows/Test%20&%20Deploy/badge.svg)](https://github.com/JFL110/deployed-ml/actions?query=workflow%3A%22Test+%26+Deploy%22)

Front end hosted [here](https://github.com/JFL110/jamesleach.dev) and available at [jamesleach.dev/ml-digit](https://www.jamesleach.dev/ml-digit).

A quick demo of a neural network implemented using Java's [Deeplearning4j](https://deeplearning4j.org/). The simple feed forward network is trained on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) of 60,000 hand-drawn images of the digits zero to nine. The network is trained using a Spring Boot command line application and the serialized network is uploaded to AWS S3. A separate Spring Boot REST application reads the serialized network and uses it to classify images input as pixel arrays. The REST application is packaged as a Docker image and deploy to an AWS Elastic Container Service cluster. 
