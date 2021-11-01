---
toc: true
layout: post
description: Explaining Cross-Entropy loss and Focal Loss
categoires: ["Machine Learning"]
title: Understanding Cross-Entropy Loss and Focal Loss
---

In this blogpost we will understand cross-entropy loss and its various different names. Later in the post, we will learn about Focal Loss, a successor of Cross-Entropy(CE) loss that performs better than CE in highly imbalanced dataset setting. We will also implement Focal Loss in PyTorch.

Cross-Entropy loss has its different names due to its different variations used in different settings but its core concept (or understanding) remains same across all the different settings. Cross-Entropy Loss is used in a supervised setting and before diving deep into CE, first let's revise widely known and important concepts:

**Multi-Class Classification** <br>
One-of-many classification. Each data point can belong to ONE of *C* classes. The target (ground truth) vector *t* will be a one-hot vector with a positive class and *C*-1 negative classes. All the *C* classes are mutually exclusives and no two classes can be positive class. The deep learning model will have *C* output neurons depicting probability of each of the *C* class to be positive class and it is gathered in a vector *s* (scores). This task is treated as a single classification problem of samples in one of *C* classes.

**Multi-Label Classification**<br>
Each data point can belong to more than one class from *C* classes. The deep learning model will have *C* output neurons. Unlike in multi-class classification, here classes are *not* mutually exclusive. The target vector *t* can have more than a positive class, so it will be a vector of 0s and 1s with *C* dimensionality where 0 is *negative* and 1 is *positive* class. One intutive way to understand multi-label classification is to treat multi-label classification as *C* different binary and independent classification problem where each output neuron decides if a sample belongs to a class or not.

