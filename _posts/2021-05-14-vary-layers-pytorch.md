---
toc: true
layout: post
description: Code example of showing right way parameterize number of layers to be added in Neural Network
categories: [markdown]
title: Adding Variable Number of Layers in Neural Network
---

Recently, I was implementing a library related to Graph Networks in PyTorch framework. There I encountered a requirement where the neural network model would have number of layers
required as input. The user would specify a random number and the neural network model would need to add those many layers to the neural network.
