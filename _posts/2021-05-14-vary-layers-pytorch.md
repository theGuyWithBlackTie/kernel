---
toc: true
layout: post
description: Code example of showing right way parameterize number of layers to be added in Neural Network
categories: [PyTorch]
title: Adding Variable Number of Layers in Neural Network
---

Consider following code block that defines a fixed 2-layer neural network. Imagine a scenario, where the network has huge number of layers, and typing out each layer manually is just not feasible. An even more notable scenario is when the number of layers of network are not fixed and it depends on some other conigurations. This article deals with these scenarios and lays out solution.

```python
class Net(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(Net, self).__init__()
    linear_layer_one = nn.Linear(input_dim, output_dim)
    linear_layer_two = nn.Linear(input_dim, output_dim)
    
   def forward(self, input):
    output = linear_layer_two( linear_layer_one(input) )
    return output
```

Recently, I was implementing a library related to Graph Networks in PyTorch framework. There I encountered the second scenario where the number of layers in the neural network were not fixed and it network required number of layers to be added as an input. The user of the library would specify how many layers is required as the input and that many layers would be added in the neural network.

As a daily user of Python, my first solution was to use a `list` data structure with a `for` loop to add n-number of layers, like below code block.

```python
class Net(nn.Module):
  def __init__(self, input_dim, output_dim, nos_linear_layer):
    super(Net, self).__init__()
    self.nn_layers = []
    for i in range(0,nos_linear_layer):
      linear_layer = nn.Linear(input_dim, output_dim)
      self.nn_layers.append(linear_layer)
    
   def forward(self, input):
    outputs = None
    for i,layer in enumerate(self.nn_layers):
      outputs = layer(input)
    
    outputs = torch.nn.functional.Softmax(outputs, 1)
    return outputs
```

Above code would look correct and would be expected to run without any issue. But, the main issue is that the linear layers stored in Python `list` would not be trained. On calling `model.parameters()`, PyTorch would simply ignore the parameters of linear layers stored in the Python `list`.

The correct way is to use PyTorch's list `nn.ModuleList`.

```python
class Net(nn.Module):
  def __init__(self, input_dim, output_dim, nos_linear_layer):
    super(Net, self).__init__()
    self.nn_layers = nn.ModuleList()
    for i in range(0,nos_linear_layer):
      linear_layer = nn.Linear(input_dim, output_dim)
      self.nn_layers.append(linear_layer)
    
   def forward(self, input):
    outputs = None
    for i,layer in enumerate(self.nn_layers):
      outputs = layer(input)
    
    outputs = torch.nn.functional.Softmax(outputs, 1)
    return outputs
```
