# LeNet-5 :

**Paper :** [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) .

**Authors :** Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner .

**Published in:** Proceedings of the IEEE 1998 . 

<div align="center" >
<img src="../resources/LeNet5.png" width="400" height="250">
</div>

let Apply our Formulas and see how an input Image with 32 * 32 * 1 size become a Vector with 100 length .

```
Input Image = 32 * 32 * 1 .

Convolution Layer 1 : K = 6 , S = 1 , P = 0 , kernel_size = 5*5.

    Output Width  = (input_width - kernel_width + 2 * padding) / stride + 1 . 
                  = (32 - 5 + 2 * 0) / 1 + 1 = 28 .

    Output Height = (input_height - kernel_height + 2 * padding) / stride + 1 . 
                  = (32 - 5 + 2 * 0) / 1 + 1 = 28 .
    
    Output Depth  = Number of kernels .
                  = 6 .

Pooling Layer 1 : S = 2 , P = 0  , filter_size = 2 * 2.

    Output Width  = (input_width - filter_width) / stride + 1 . 
                  = (28 - 2) / 2 + 1 = 14

    Output Height = (input_height - filter_height) / stride + 1 . 
                  = (28 - 2) / 2 + 1 = 14 

    Output Depth  = The Same Depth .
                  = 6 .

Convolution Layer 2 : K = 16 , S = 1 , P = 0 , kernel_size = 5*5.

    Output Width  = (input_width - kernel_width + 2 * padding) / stride + 1 . 
                  = (14 - 5 + 2 * 0) / 1 + 1 = 10 .

    Output Height = (input_height - kernel_height + 2 * padding) / stride + 1 . 
                  = (14 - 5 + 2 * 0) / 1 + 1 = 10 .

    Output Depth  = 16 .

Pooling Layer 2 : S = 2 , P = 0  , filter_size = 2 * 2.

    Output Width  = (input_width - filter_width) / stride + 1 . 
                  = (10 - 2) / 2 + 1 = 5

    Output Height = (input_height - filter_height) / stride + 1 . 
                  = (10 - 2) / 2 + 1 = 5 

    Output Depth  = 16 .

Flatten Layer : S = 1 , P = 0 , K = 120 , kernel_size = 5*5 .

    Output Width  = (input_width - filter_width) / stride + 1 . 
                  = (5 - 5) / 1 + 1 = 1

    Output Height = (input_height - filter_height) / stride + 1 . 
                  = (5 - 5) / 1 + 1 = 1 

    Output Depth  = 120 .

    Output Vector = Output Width * Output Height * Output Depth .
                  = 1 * 1 * 120 .
                  = 120 .


```

**Model Architecture :** 

<div align="center" >
<img src="resources/leNet5Architecture.PNG" width="200" height="400">
</div>

**keras :**

```python
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , AveragePooling2D , Dense , Dropout , Flatten
from tensorflow.keras.optimizers import Adam

# Disable TensorFlow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def load_dataSet():
    (x_train , y_train) , (x_test , y_test) = mnist.load_data()
    
    #Concatenate The Data
    x = np.concatenate((x_train , x_test))
    y = np.concatenate((y_train , y_test))    
    
    #Transform The Images from 28*28 to 32*32
    x = np.pad(x , ((0,0) , (2,2) , (2,2)))
    
    #Reshape The Data
    x = x.reshape((x.shape[0] , 32 , 32 , 1))
    
    #One Hot Encodig
    y = to_categorical(y)
    
    #Normalize The Data
    
    x = x.astype('float32')
    
    x /= 255.0
    
    return x , y

def LeNet5():
    
    model = Sequential()
    
    model.add(Conv2D(filters=6 , kernel_size=(5,5) , strides=(1,1) , activation="tanh" , input_shape=(32 , 32 , 1)))
              
    model.add(AveragePooling2D((2,2)))
    
    model.add(Conv2D(filters=16 , kernel_size=(5,5) , strides=(1,1) , activation="tanh" ))   
                  
    model.add(AveragePooling2D((2,2)))
              
    model.add(Flatten())
    
    model.add(Dense(units=120 , activation="tanh"))
    
    model.add(Dropout(0.2))
    
    model.add(Dense(units=84 , activation="tanh"))
    
    model.add(Dropout(0.2))
    
    model.add(Dense(units=10 , activation="softmax"))
    
    adam = Adam(learning_rate=0.01)
    
    model.compile(optimizer=adam , loss = 'binary_crossentropy'  , metrics=['accuracy'])
    
    return model


def displayAccuracy(history):
    # display The accuracy of our Model
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def displayLoss(history):
    # dsiplay the loss of our Model
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    
    # Load The DataSet
    x , y = load_dataSet()
    
    #Define The Model
    model = LeNet5()
    
    # Train The Model
    history = model.fit(x , y ,validation_split=0.33, epochs=10, batch_size=100)
    
    #display The Model Accuracy
    displayAccuracy(history)
    
    #display The Model Loss
    displayLoss(history)
    
    #display The Model Architecture
    print("The Architecture of The Model is : ")
    print(model.summary())
    
```

**pyTorch :**

```python
from torch.nn import Conv2d , AvgPool2d , Linear , Flatten , Module
import torch.nn.functional as F

class LeNet5(Module):
    
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1   = Conv2d(in_channels = 1, out_channels = 6, kernel_size = (5,5) , stride = 1)
        self.conv2   = Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5,5) , stride = 1)
        self.conv3   = Conv2d(in_channels = 16, out_channels = 120, kernel_size = (5,5) , stride = 1)
        self.avgpool = AvgPool2d(kernel_size=2 , stride=2)
        self.fc1     = Linear(in_features = 120, out_features = 84)
        self.fc2     = Linear(in_features = 84, out_features = 10)
        
    def forward(self , x):
        x = self.conv1(x)
        x = F.tanh(x)
        
        x = self.avgpool(x)
        
        x = self.conv2(x)
        x = F.tanh(x)
        
        x = self.avgpool(x)
        
        x = self.conv3(x)
        x = F.tanh(x)
        
        x = x.view(x.shape[0] , -1)
        
        x = self.fc1(x)
        x = F.tanh(x)
        
        x = self.fc2(x)
        x = F.softmax(x , dim=1)
        
        return x
```