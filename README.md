# Convolutional Neural Network From Scratch

This Repository Contains The Explanation and The Implementation Of Convolutional Neural Network Using Keras and Pytorch .

In This Repository you'll see :

* Introduction to CNN .

* Convolutional Neural Network vs Multilayer Perceptron .

* Convolutional Neural Network Layers .
    
    * Kernels or Filters .
    
    * Convolutional layer .
    
    * Activation Layer .
    
    * Pooling Layer .
    
    * Fully Connected Layer .
 
* Different Layers in **Keras** and **pyTorch** . 
* Most Common Architectures of CNN and their Implementation .

* References . 

---
## Introduction :

<img align="right" width="40%"  src="resources/cnn.jpg">

The Convolutional Neural Network, known as CNN (Convolutional Neural Network), is one of the deep learning algorithms that is the development of the Multilayer Perceptron (MLP) designed to process data in the form of a Matrix (image, sound ...).




Convolutional Neural Networks are used in many fields, but we will just be interested in the application of CNNs to Images.

**The question now is, what is an Image?**

Image is Just a Matrix of Pixels .

<div align="center" >
<img src="resources/pixels.jpg" width="400" height="200">
</div>


**Coding Modes of an Image:**

<div align="center" >
<img src="resources/Codage.jpg" width="400" height="200">
</div>

---
## Convolutional Neural Network vs Multilayer Perceptron :

Imagine with me that we've an Image classification problem to solve , and we've only one choice which is Multilayer Perceptron (Neural Network ) , and The images they have 240 height and 240 width and we're Using RGB.

do you know that we need to build a Neural Network with 240 * 240 * 3 = 172 800 Input which is a very big Neural Network , and it will be very hard for as to train it .

**Can we find a solution that reduces the size of the images and preserves the Characteristics ?**

This is Exactly What CNN Can Do .

<div align="center" >
<img src="resources/cnn-vs-mlp.png" width="400" height="200">
</div>

In General :

CNN = Convolutional Layers + Activation Layers + Pooling Layers + Fully Connected Layers .

<div align="center" >
<img src="resources/cnnlayers.png" width="400" height="200">
</div>

---
## Convolutional Neural Network Layers :

### Kernels or Filters in The Convolutional layer  :

In the convolutional neural network, the Kernel is nothing more than a filter used to extract features from images. The kernel is a matrix that moves over the input data, performs the dot product with the input data subregion, and obtains the output as a dot product matrix. The kernel moves on the input data by the stride value.

<div align="center" >
<img src="resources/kernels.png" width="350" height="250">
</div>

There is a lot Kernels , each one is responsible for extracting a specific Feature.

<div align="center" >
<img src="resources/kernels-examples.png" width="350" height="250">
</div>

### Convolutional Layers :

The Convolution Layer Extract The Characteristics of The Image By Performing this operation To The Input Image :

<div align="center" >
<img src="resources/conv.gif" width="300" height="150">
</div>

The Convolutional Layer produce an Output Image with this Formula :

<div align="center" >
<img src="resources/ConvOutputFormula.png" width="300" height="150">
</div>

The Convolutional Layer needs Two Parameters to work :

* Padding :  the amount of pixels added to an image when it is being processed by the kernel of a CNN.
* Stride : Stride is the number of pixels shifts over the input matrix .  


**Example 1 :** Stride = 1 , Padding = 0 :

<div align="center" >
<img src="resources/conv-ex1.png" width="350" height="200">
</div>

if we Applied our Formula (In The Picture above) we'll get The Same Result .

```
output width = (input_width - kernel_width + 2 * padding) / stride_width + 1

output height = (input_height - kernel_height + 2 * padding) / stride_height + 1

input Image : 6*6
Kernel Size : 2*2

output width = (6 - 2 + 2 * 0) / 1 + 1 = 5
output height = (6 - 2 + 2 * 0) / 1 + 1 = 5
```

**Example 2 :** Stride = 2 , Padding = 0 :

<div align="center" >
<img src="resources/conv-ex2.png" width="350" height="200">
</div>

```
input Image : 6*6
Kernel Size : 2*2

output width = (6 - 2 + 2 * 0) / 2 + 1 = 3
output height = (6 - 2 + 2 * 0) / 2 + 1 = 3
```

**Example 3 :** Stride = 2 , Padding = 1 :

<div align="center" >
<img src="resources/conv-ex3.png" width="350" height="200">
</div>

```
input Image : 6*6
Kernel Size : 2*2

output width = (6 - 2 + 2 * 1) / 2 + 1 = 4
output height = (6 - 2 + 2 * 1) / 2 + 1 = 4
```

In All The Examples Above we was talking about Convolution 2D , now let See The general Case which is Convolution 3D :

```
Input Image : W1×H1×D1 .
Number of filters : K (With Size F*F).
the stride  : S .
Padding : P .
Output : 
W2 = (W1−F+2P)/S+1 .
           H2 = (H1−F+2P)/S+1 .
           D2 = K .

```

<div align="center" >
<img src="resources/conv-3d.gif" width="350" height="250">
</div>

---
## Activation Function in The Convolutional layer :

The activation function used in CNN networks is RELU and it is defined as follows:

```
RELU (z) = max (0, z)
```

<div align="center" >
<img src="resources/ReluGraph.png" width="300" height="200">
</div>

---
## Pooling Layer :

The Pooling Layer Reduce The Size of The Image , there is two type of Pooling :

* Max Pooling .
* AVG Pooling .

<div align="center" >
<img src="resources/MaxAvg.png" width="400" height="280">
</div>

The Output Of The Pooling Layer Can be calculated Using This Formula :

<div align="center" >
<img src="resources/Pooling-Formula.png" width="300" height="100">
</div>

#### Max Pooling :

<div align="center" >
<img src="resources/MaxPool.png" width="400" height="250">
</div>

#### AVG Pooling :

<div align="center" >
<img src="resources/AvgPool.png" width="400" height="250">
</div>

---
## Fully Connected Layer :

fully connected layer it can be seen as one layer of a simple Neural Network .

<div align="center" >
<img src="resources/fully-connected-layer.PNG" width="300" height="150">
</div>

---
## Different Layers in **Keras** and **pyTorch** : 

### **Keras** :

<img align="right" width="30%"  src="resources/keras.png">

Keras is an open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library. 




* Convolution Layer :

```python
tf.keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```


* Activation Layer :

```python
tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)
```

* Pooling Layer :

    * Max-Pooling :
   
    ```python
    tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs
    )
    ```
    
    * Avg-Pooling :
    
    ```python
    tf.keras.layers.AveragePooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs
    )
    ```
* Dropout Layer :

```python
tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
```

* Dense Layer or Fully Connected Layer :

```python
tf.keras.layers.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```




### **pyTorch** :

<img align="right" width="30%" src="resources/pyTorch.png">

PyTorch is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab. It is free and open-source software released under the Modified BSD license.


* Convolution Layer :

```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
```


* Activation Layer :

```python
torch.nn.ReLU(inplace=False)
```

* Pooling Layer :

    * Max-Pooling :
   
    ```python
    torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    ```
    
    * Avg-Pooling :
    
    ```python
    torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
    ```

* Dropout Layer :

```python
torch.nn.Dropout(p=0.5, inplace=False)
```

* Dense Layer or Fully Connected Layer :

```python
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```

---
## Most Common Architectures of CNN and their 

---
**References :**

* [Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network).

* [Illustrated: 10 CNN Architectures](https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d).

* [Multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron).