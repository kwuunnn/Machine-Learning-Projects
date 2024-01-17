# CS2109S PyTorch Command Glossary v3

## Preface

This document serves as a reference point to all things PyTorch. It is designed specifically to avoid unnecessary searching online, providing you with the key elements of the `torch` library for Problem Sets 6 and 7. In fact, you can use it for your projects, other modules, and wherever else you decide to take PyTorch!

> For more information, visit the [official PyTorch
> documentation](https://pytorch.org/docs/stable/index.html).

### Table of Contents

- Installation
- Usage
- Tensors
    - Basics
    - Randomness
    - Operations
- Working with Gradients
    - Partial Differentiation
- Common `torch` operations
- The `torch.nn` Layers API
- The `torch.nn.Sequential` API <sup>1</sup>
- Training PyTorch Networks
    - Optimisers
    - Losses
- Data Augmentations with `torchvision.transforms` <sup>2</sup>
- Closing Words

> <sup>1,2</sup> For Problem Set 7

---

### Installation

To install PyTorch, use `pip` in your terminal. You can either download it globally across your system or inside a virtual environment (recommended but not necessary).

```bash
$ pip install torch torchvision
```

> The additional `torchvision` library allows us to make use of popular datasets, hence the additional package
> installation.

### Usage

To use PyTorch, import the library and its submodules:

```python
import torch
import torch.nn as nn
```

- `torch` is the base library
- `torch.nn` allows you to build neural network layers, create loss functions and optimisers, and more

## Tensors

The PyTorch `Tensor` is akin to `NumPy`'s `numpy.ndarray` object – essentially, a n-dimensional matrix. There are a few different ways to create tensors:

### Tensor Basics

```python
a = torch.tensor(...)  # creating a tensor

# data types:
torch.Tensor(...)  # any kind of value
torch.FloatTensor(...)  # float values only
torch.LongTensor(...)  # integer values only
```

You can replace `...` with any value of any numerical data type:

- integer
- float
- n-dim (nested) array of integers/floats

Let's stick to using `torch.tensor(...)` to create tensors in this module. Let's avoid using `Tensor`, `FloatTensor`, and `LongTensor` as they impose restrictions on what values they can hold.

> __Bonus:__ if your tensor has a single element (i.e., a `1x1` tensor), you can extract its value using the `.item()` method.
>
> For instance, if `a = torch.tensor(123)`, `a.item()` will return `123`.

### Randomness

Most often, you are required to inject randomness to your experiments. Similar to `numpy`, you can generate Tensors of any arbitrary size/dimensionality with random values. Here are some ways to generate random tensors:

- `torch.rand(size)`: draws digits from Uniform distribution `x ~ U(0, 1)`
- `torch.randn(size)`: draws digits from Normal distribution `x ~ N(0, 1)`
- `torch.randint(low, high, size)`: generates tensors with random integers

```python
a = torch.rand(10, 10)  # a 10x10 matrix 
b = torch.rand(10)  # vector with 10 elements
c = torch.rand(10, 1)  # vector with 10 elements with an extra (insignificant) dimension
d = torch.rand(28, 28, 28)  # a "cube" tensor with 28 elements

e = torch.randn(10, 5)  # a 10x5 matrix

f = torch.randint(0, 100, (5, 5))  # a 5x5 matrix of integers in [0, 100)
```

> All of these random tensors are, by default, `torch.Tensor` object data type. Each element of these tensors are also of the same `torch.Tensor` type.
>
> So, large tensors are made of smaller tensor units. It's the fundamental "building block" of PyTorch (like the "cell" in animal!).

### Operations

As with `np.array`, you can perform familiar tensor operations such as addition, subtraction, multiplication, division, and exponentiation.

```python
a = torch.tensor(50)
b = torch.tensor(75)
p = torch.tensor(2)

c = a + b
print(c)  # torch.Tensor(75)

d = b - a
print(d)  # torch.Tensor(25)

e = b * a
print(e)  # torch.Tensor(3750)

f = b / a
print(f)  # torch.Tensor(1.5000)

g = a ** p
print(g)  # torch.Tensor(2500)
```

In fact, when working with PyTorch tensors, you can perform operations with non-tensors as well:

```python
a = torch.tensor(50)

b = a + 4  # torch.Tensor(54)
c = a - 4  # torch.Tensor(46)
d = a * 2  # torch.Tensor(100)
d = a / 2  # torch.Tensor(25.0)
e = a ** 2  # torch.Tensor(2500)
```

---

## Working with Gradients

Efficiently computing gradients is what PyTorch is known for. When creating tensors, we use the `requires_grad` parameter to tell PyTorch we hope to perform gradient computation with this variable in the future. This allows PyTorch to store gradient information inside the tensor for later access. By default, this parameter is `False` because it's relatively more space-heavy to store gradients inside the tensor object.

```python
a = torch.tensor(10.0, requires_grad=True)  # set the param to True, default is False
```

### Partial Differentiation

In Machine Learning, gradient computation involves taking partial derivatives of one variable with respect to another. To achieve this, we use the `backward()` method of tensors (provided that they have `requires_grad=True`).

```python
a = torch.tensor(5.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)

c = (2 * a) + b ** 2  # torch.Tensor(14.0)
c.backward()
```

The variable on which `backward()` is called is the target variable (`c` in this case). All other tensors involved in the computation have their gradient values automatically computed. So, in this case, partial derivatives `dc/da` and `dc/db` are computed automatically and stored within `a` and `b` respectively.

> Fun fact: this is why we call the package `autograd`, which alludes to automatic computation of gradients!!!

Once we call `backward()`, all that's left to do is access the gradient values for each variable of interest. This is done via the `grad` attribute of a tensor:

```python
"""
Partial derivatives:

c = 2a + b^2
dc/da = 2
dc/db = 2b
"""

dc_da = a.grad  # 2.0
dc_db = b.grad  # 4.0
```

---

## Common `torch` Operations

Most, if not all, of these operations are differentiable by nature. This means you can use them within your *computation graph* and compute gradients.

| __Operation__                 | __Remarks__                                                                                                          |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `torch.sum(input)`            | Returns the sum of all elements in the input tensor.                                                                 |
| `torch.pow(base, exp)`        | Returns the exponentiation of the base tensor to the exponent.                                                       |
| `torch.mean(input)`           | Returns the mean of all elements in the input tensor.                                                                |
| `torch.square(input)`         | Returns the square of elements in the input tensor.                                                                  |
| `torch.no_grad()`             | Pauses all gradient computation and tracking inside the `with torch.no_grad()` block.                                |
| `torch.matmul(input, other)`  | Returns the matrix product of tensors `input` and `other`. Same effect as `A @ B`.                                   |
| `torch.reshape(input, shape)` | Returns the reshaped input matrix if dimensions commute. Same as `torch.view(input, shape)`.                         |
| `torch.softmax(input, dim)`   | Computes the Softmax of an input along a specified dimension/axis.                                                   |
| `torch.max(input, dim)`       | Returns the maximum element in the input tensor along a specific dimension/axis.                                     |
| `torch.min(input, dim)`       | Returns the minimum element in the input tensor along a specific dimension/axis.                                     |
| `torch.manual_seed(seed)`     | Sets the random number generator seed to the one specified. Good for reproducibility of runs.                        |
| `torch.zeros(size)`           | Returns a tensor of zeros corresponding to the specific size.                                                        |
| `torch.ones(size)`            | Returns a tensor of ones corresponding to the specific size.                                                         |
| `torch.squeeze(input, dim)`   | Returns the tensor by removing a dimension `dim` from it. Eg: (1, 32, 32) -> dim=0 -> (32, 32)                       |
| `torch.unsqueeze(input, dim)` | Returns the tensor by adding an extra dimension at `dim`. Eg: (32, 32) -> dim=0 -> (1, 32, 32)                       |
| `torch.clip(input, min, max)` | Returns the tensor with all values in range `[min, max]`. All out-of-bounds values are made `max`/`min` accordingly. |

> For `torch.matmul(...)`, you can also use `@` between the matrices of interest as long as they _commute_. Suppose `A`
> is a 3x4 tensor and `B` is a 4x5 tensor. `C = A @ B` will be a 3x5 tensor.
>
> PyTorch can also matrix multiply tensors of higher dimensions (3D, 4D, ...) but we will not be getting into that topic
> just yet.
>
> In fact, all these operations can be called on `Tensor` objects themselves like `x.squeeze(0)` for example.

---

## The `torch.nn` Layers API

The speciality of PyTorch lies in its pythonic way of building neural networks. It provides a nice interface to quickly prototype models and train/test them using a compact, low-overhead, neatly-written train-test loop.

### `nn.Module`

The `nn.Module` interface provides the necessary methods to facilitate the construction of neural networks, both simple and complex. The `__init__` and `forward` method are the most important: they house the individual layers of the network, and compute the forward pass for a given input tensor respectively.

> __IMPORTANT:__ By convention, ALL layers are initialised in the `__init__` method. These same layers are then
> referenced and used via `self` in the `forward` method.

Here's a snippet of a neural network using PyTorch:

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()  # don't forget to inherit from the parent class
        self.l1 = ...
        self.l2 = ...
        self.l3 = ...
        self.l4 = ...

    def forward(self, x):
        """
        By default, the only input this function can take in is `x`, the input tensor.
        Don't add any other parameters into this function to keep things simple.
        """
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        out = self.l4(x)

        return out
```

Here are two layers you'll use most during your time in CS2109S. It's best to familiarise yourself with them!

| Layer                   | Usage                                                                | Remarks                                                                                                  |
|-------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Fully-connected / Dense | `nn.Linear(in_features, out_features, bias=True)`                    | Inputs are vectors of size `in_features`. Performs `Y=Wx+b` and outputs a vector of size `out_features`. |
| Convolution<sup>1</sup> | `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)` | Inputs are images/tensors with `in_channels` number of channels of arbitrary height and width.           |
| ReLU                    | `nn.ReLU()`                                                          | Performs the Rectified Linear Units (ReLU) activation on the input tensor.                               |
| Leaky ReLU              | `nn.LeakyReLU(negative_slope=0.01)`                                  | Performs the Leaky ReLU activation on the input tensor with the specified negative slope.                |
| Sigmoid                 | `nn.Sigmoid()`                                                       | Performs the Sigmoid activation on the input tensor with the specified negative slope.                   |
| Max Pooling             | `nn.MaxPool2d(pool_size)`                                            | Performs the Maximum Pooling operation on the input tensor with the specified pooling size.              |
| Dropout                 | `nn.Dropout(p)`                                                      | Performs Dropout on the layer _prior to being called_ with the specified dropping probability.           |

> <sup>1</sup> Only in Problem Set 7.

### Optimisers and Losses

Most important for any gradient-based computation program is the optimiser and objective (i.e., loss) function. Writing your own optimiser or loss is a tedious process and is error-prone if you are not sure how to write efficient PyTorch code. To alleviate this, PyTorch allows you invoke popular optimisers and losses with a single line of code.

Additionally, here's a list of popular loss functions:

| **Loss**               | **Usage**               |
|------------------------|-------------------------|
| Cross Entropy          | `nn.CrossEntropyLoss()` |
| Binary Cross Entropy   | `nn.BCELoss()`          |
| Mean Squared Error     | `nn.MSELoss()`          |
| Mean Absolute Error    | `nn.L1Loss()`           |
| Negative Log Liklihood | `nn.NLLLoss()`          |

After computing the output of the forward pass using your model, you can do:

```python
loss_fn = nn.XYZLoss()  # some arbitary loss from the above table

output = ...  # some tensor
target = ...  # some tensor
loss = loss_fn(output, target)

"""
As mentioned above, to backpropagate the loss wrt the parameters, you can simply call
`loss.backward()` and it will compute the partial derivates (i.e., the gradients) and
store them inside the `.grad` attribute of each and every parameter tensor!

Pretty cool, huh? ;)
"""
```

Here's a list of popular optimisers:

| **Optimiser**                     | **Usage**                                  |
|-----------------------------------|--------------------------------------------|
| Stochastic Gradient Descent (SGD) | `torch.optim.SGD(parameters, lr)`          |
| Adaptive Momentum (Adam)          | `torch.optim.Adam(parameters, lr=0.001)`   |
| Adaptive Gradient (Adagrad)       | `torch.optim.Adagrad(parameters, lr=0.01)` |

Here, `parameters` refers to the network parameters and `lr` is the learning rate. Different optimisers have different default learning rates while some require the user to input that in (for example, SGD needs you to specify the `lr` while Adam has a learning rate of `0.001`). In your Problem Sets, if the learning rate is __NOT__ specified, it means we expect you to use the default; you don't have to tune these numbers yourself.

Suppose we have a network `net = Net(...)` that's build using the `nn.Module` interface. To access the parameters of the model, we simply call the `net.parameters()` method; it will return a list of all the weights and biases (i.e., parameters) of the model. We pass these parameters into the optimiser, along with any other arguments (like learning rate, for instance).

```python
net = Net(...)
optimiser = torch.optim.SGD(net.parameters(), lr=0.001)
```

__IMPORTANT NOTES:__

- Before you perform a forward pass, we must ensure that the optimiser doesn't have the previous iteration's gradients stored inside it. To flush them, we must reset them to zero. To do so, add the line `optimiser.zero_grad()` before your forward pass through the network using input `x`.

- Additionally, after a backward pass via Backpropagation, we must perform an update step to the parameters within the network. In Gradient Descent, for example, this update step is `w = w - lr * dLdw`. To do so, simply call `optimiser.step()` after the `loss.backward()` line.

Simply put, your forward and backard pass should look like this:

```python
for x, y in dataset:
    optimizer.zero_grad()  # flush the prev gradients
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()  # perform backpropagation
    optimizer.step()  # update parameters
```

## The `torch.nn.Sequential` API

In PS7, we will be working with the `nn.Sequential` API. So far, you've been creating model layers one by one and calling them things like `self.linear1` or `self.relu`. However, for larger networks, the process is tedious and cumbersome.

This is why, we use the `nn.Sequential` API that allows you to add in the layer objects __by name__ without instantiating them with a variable name. Here are some examples of the Sequential API in action:

```python
densenet = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(1)  # softmax dimension
)

x = torch.rand(15, 784)  # a batch of 15 MNIST images
y = densenet(x)  # here we simply run the sequential densenet on the `x` tensor
print(y.shape)  # a batch of 15 predictions
```

```python
convnet = nn.Sequential(
    nn.Conv2d(1, 32, (3, 3)),
    nn.ReLU(),
    nn.Conv2d(32, 64, (3, 3)),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(36864, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(1)  # softmax dimension
)

x = torch.rand(15, 1, 28, 28)  # a batch of 15 MNIST images
y = convnet(x)  # here we simply run the sequential convnet on the `x` tensor
print(y.shape)  # a batch of 15 predictions
```

__Note:__ Do __NOT__ pass your layers as a array in `nn.Sequential`'s arguments:

```python
net = nn.Sequential(xyz, abc, mno)      # correct
net = nn.Sequential([xyz, abc, mno])    # error
```

## Data Augmentations with `torchvision.transforms`

In PS7, we'll be dealing with Computer Vision, which requires us to apply transformation on bitmap (tensor) images. Here are a bunch of augmentations you can pick from! Feel free to check out the [`torchvision.transforms` documentation](https://pytorch.org/vision/stable/transforms.html) for more!

| **Augmentation**            | **Remarks**                                                                                              |
|-----------------------------|----------------------------------------------------------------------------------------------------------|
| `ToTensor()`                | Converts a numpy array or JPEG image to `torch.Tensor` format; compulsory for transforms                 |
| `Normalize([mean], [std])`  | Normalises incoming tensors; for x-D images, you should specify the `mean` and `std` as x-sized arrays   |
| `Grayscale`                 | Converts a coloured RGB image to grayscale                                                               |
| `RandomHorizontalFlip(p)`   | Horizontally flips an image with the specified probability                                               |
| `RandomVerticalFlip(p)`     | Vertically flips an image with the specified probability                                                 |
| `RandomRotation(degrees)`   | Rotates an image with a randomly chosen degree from specified [d0, d1, ..., dn] array; do not use radian |
| `GaussianBlur(kernel_size)` | Applies a gaussian blur on an image using the specified kernel size                                      |
| `RandomGrayscale(p)`        | Randomly Converts a coloured RGB image to grayscale with the specified probability                       |

---

There's definitely more than meets the eye when it comes to PyTorch. This library is the primary workhorse around the world, so it's beneficial to learn it from the ground up. Problem Sets 6 and 7 simply offer a taste on how life with PyTorch is like – it's way better than initialising individual biases and manually writing out the equations for a lot of Machine Learning applications.

Of course, as with any library, if you're interested in diving under its hood, feel free to look at its documentation where you can find fun tutorials and exercises to jog your mind.

> For more information, visit the [official PyTorch
> documentation](https://pytorch.org/docs/stable/index.html).

"Happy (Machine) Learning!!!" ~ CS2109S Teaching Team