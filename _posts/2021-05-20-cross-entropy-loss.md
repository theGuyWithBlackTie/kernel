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

Taking a very rudimentary example, consider the target(groundtruth) vector *t* and output score vector *s* as below:
```
Target Vector: [0.6     0.3     0.1]
Score Vector:  [0.2     0.3     0.5]
```
Then *CE* will be computed as follows:
```
CE = -(0.6)log(0.2) - 0.3log(0.3) - 0.1log(0.5) = 0.606
```
{% include alert.html text="In supervised machine learning settings, elements of target vectors are either 1 or 0. The above example shows how CE is computed and how it is also applicable to find loss between the distributions." %}

#### Categorical Cross-Entropy Loss
In multi-class setting, target vector *t* is one-hot encoded vector with only one positive class (i.e.$$t_i = 1$$) and rest are negative class (i.e. $$t_i = 0$$). Due to this, we can notice that losses for negative classes are always zero. Hence, it does not make much sense to calculate loss for every class. Whenever our target (ground truth) vector is one-hot vector, we can ignore other labels and utilize only on the hot class for computing cross-entropy loss. So, Cross-Entropy loss becomes:

$$ CE = -log(s_i)$$ 

{% include info.html text="The above form of cross-entropy is called as Categorical Cross-Entropy loss. In multi-class classification, this form is often used for simplicity." %}

The ***Categorical Cross-Entropy*** loss is computed as follows:
![]({{ site.baseurl }}/images/softmax_loss.png)

$$ f(s)_i = \frac{e^{s_i}}{\sum_{j}^{C}e^{s_j}} \Rightarrow CE = -\sum_i^C t_i log(f(s)_i) \Rightarrow CE = -log(f(s)_i)$$

As, SoftMax activation function is used, many deep learning frameworks and papers often called it as SoftMax Loss as well.

#### Binary Cross-Entropy Loss
Based on another classification setting, another variant of Cross-Entropy loss exists called as ***Binary Cross-Entropy Loss***(BCE) that is employed during binary classification (*C* = 2). Binary classification is multi-class classification with only 2 classes. To dumb it down further, if one class is a *negative class* automatically the other class becomes *positive class*. In this classification, the output is not a vector *s* but just a single value. Let's understand it further.

The target(ground truth) vector for a random sample contains only one element with value of either 1 or 0. Here, 1 and 0 represents two different classes (*C* = 2). The output score value ranges from 0 to 1. If this value is closer to 1 then class 1 is being predicted and if it is closer to 0, class 0 is being predicted.

    $$BCE = -\sum_{i=1}^{C=2}t_ilog(f(s)_i) = -t_1log(f(s_1)) - (1-t_1)log(1-f(s_1))$$


$$s_1$$ and $$t_1$$ are the score and groundtruth label for the class $$C_i$$ in $$C$$. $$s_2 = 1 -s_1$$ and $$t_2 = 1 - t_1$$ are the score and groundtruth label for the class $$C_2$$. If $$t_1 = 0$$ then $$-t_1log(f(s_1))$$ would become $$0$$ and $$(1-t_1)log(1-f(s_1))$$ would become active. Similarly, if $$t_1 = 1$$ then $$-t_1log(f(s_1))$$ would become active and $$(1-t_1)log(1-f(s_1))$$ would become $$0$$. The loss can be expressed as:

$$CE = 
    \begin{cases}
    -log(f(s_1)) & if & t_1 = 1 \\
    -log(1-f(s_1)) & if & t_1 = 0
    \end{cases}
$$

To get the output score value between [0,1], sigmoid activation function is used.
{% include info.html text="Due to the using of Sigmoid Activation function it is also called as Sigmoid-Cross Entropy Loss." %}

![]({{ site.baseurl }}/images/sigmoid_loss.png)

$$
f(s_1) =\frac{1}{1+e^{-s_1}} \Rightarrow CE = -t_1log(f(s_1)) - (1-t_1)log(1-f(s_1))
$$

#### Cross-Entropy in Multi-Label Classification
