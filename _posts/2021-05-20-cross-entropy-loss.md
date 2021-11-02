---
toc: true
layout: post
description: Explaining Cross-Entropy loss and Focal Loss
categories: [Machine Learning]
title: Understanding Cross-Entropy Loss and Focal Loss
---

In this blogpost we will understand cross-entropy loss and its various different names. Later in the post, we will learn about Focal Loss, a successor of Cross-Entropy(CE) loss that performs better than CE in highly imbalanced dataset setting. We will also implement Focal Loss in PyTorch.

Cross-Entropy loss has its different names due to its different variations used in different settings but its core concept (or understanding) remains same across all the different settings. Cross-Entropy Loss is used in a supervised setting and before diving deep into CE, first let's revise widely known and important concepts:

#### Classifications
**Multi-Class Classification** <br>
One-of-many classification. Each data point can belong to ONE of *C* classes. The target (ground truth) vector *t* will be a one-hot vector with a positive class and *C*-1 negative classes. All the *C* classes are mutually exclusives and no two classes can be positive class. The deep learning model will have *C* output neurons depicting probability of each of the *C* class to be positive class and it is gathered in a vector *s* (scores). This task is treated as a single classification problem of samples in one of *C* classes.

**Multi-Label Classification**<br>
Each data point can belong to more than one class from *C* classes. The deep learning model will have *C* output neurons. Unlike in multi-class classification, here classes are *not* mutually exclusive. The target vector *t* can have more than a positive class, so it will be a vector of 0s and 1s with *C* dimensionality where 0 is *negative* and 1 is *positive* class. One intutive way to understand multi-label classification is to treat multi-label classification as *C* different binary and independent classification problem where each output neuron decides if a sample belongs to a class or not.

#### Output Activation Functions
These functions are transformations applied to vectors coming out from the deep learning models before the loss computation. The outputs after transformations represents probabilities of belonging to either one or more classes based on multi-class or multi-label setting. 

**Sigmoid**<br>
It squashes a vector in the range (0,1). It is applied independently to each element of vector *s*.
![]({{ site.baseurl }}/images/sigmoid.png "Sigmoid Activation Function")

$$f(s_i) = \frac{1}{1 + e^{-s_{i}}}$$

**Softmax**<br>
It squashes a vector in the range (0, 1) and all the resulting elements add up to 1. It is applied to the output vector *s*. The Softmax activation cannot be applied independently to each element of vector *s*, since it depends on all elements of *s*. For a given class *s_i*, the Softmax function can be computed as:

$$f(s)_i = \frac{e^{(s_i)}}{\sum_{j}^C e^{s_j}}$$


## Losses
### Cross Entropy Loss
The cross-entropy loss is defined as:

$$CE = -\sum_i^C t_i log(s_i )$$

where $$t_i$$ and $$s_i$$ are the goundtruth and output score for each class *i* in *C*.

In multi-class setting, target vector *t* is one-hot encoded vector with only one positive class (i.e.$$t_i = 1$$) and rest are negative class (i.e. $$t_i = 0$$). Due to this, we can notice that losses for negative classes are always zero. Hence, it does not make much sense to calculate loss for every class. Whenever our target (ground truth) vector is one-hot vector, we can ignore other labels and utilize only on the hot class for computing cross-entropy loss. So, Cross-Entropy loss becomes:

$$ CE = -log(s_i)$$ 

{% include info.html text="The above form of cross-entropy is called as Categorical Cross-Entropy loss. In multi-class classification, this form is often used for simplicity." %}

The *Categorical Cross-Entropy* loss is computed as follows:
![]({{ site.baseurl }}/images/softmax_loss.png "Categorixcal Cross-Entropy also known as Softmax Loss Function")





