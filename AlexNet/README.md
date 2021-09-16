### AlexNet :

**Paper :** [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) .

**Authors :** Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton. University of Toronto, Canada.

**Published in :** NeurIPS 2012 .

**Model Architecture :** 

<div align="center" >
<img src="../resources/AlexNetArchitecture.PNG" width="200" height="500">
</div>

**keras :**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPool2D , Dense , Dropout , Flatten
from tensorflow.keras.optimizers import Adam

def AlexNet() :
    
    model = Sequential()
    
    model.add(Conv2D(filters=96 ,kernel_size=(11,11) , strides=(4,4), padding='valid' , input_shape=(224 , 224 , 3) ,activation='relu'))
    
    model.add(MaxPool2D(pool_size=(3,3) , strides=2))
    
    model.add(Conv2D(filters=256 ,kernel_size=(5,5) , strides=(1,1) , padding='valid',activation='relu'))
    
    model.add(MaxPool2D(pool_size=(3,3), strides=2))
    
    model.add(Conv2D(filters=384 , kernel_size=(3,3), strides=(1,1) ,padding='valid' , activation='relu'))
    
    model.add(Conv2D(filters=384 , kernel_size=(3,3) , strides=(1,1) ,padding='valid', activation='relu'))
    
    model.add(Conv2D(filters=256 , kernel_size=(3,3) , strides=(1,1) ,padding='valid', activation='relu'))
    
    model.add(MaxPool2D(pool_size=(3,3), strides=2))
    
    model.add(Dense(units=4096, activation='relu'))
    
    model.add(Dropout(rate=0.5))
    
    model.add(Dense(units=4096, activation='relu'))
    
    model.add(Dropout(rate=0.5))
    
    model.add(Dense(units=1000 , activation='softmax'))

    return model
```

**pyTorch :**

```python
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(Module):
  def __init__(self):
    super(AlexNet , self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3 , out_channels=96 , kernel_size=(11,11) , padding=(0,0),stride=(4,4))
    self.conv2 = nn.Conv2d(in_channels=96 , out_channels=256 , kernel_size=(5,5) ,padding=(2,2) ,stride=(1,1))
    self.conv3 = nn.Conv2d(in_channels=256 , out_channels=384 , kernel_size=(3,3) ,padding=(1,1),stride=(1,1))
    self.conv4 = nn.Conv2d(in_channels=384 , out_channels=384 , kernel_size=(3,3) ,padding=(1,1),stride=(1,1))
    self.conv5 = nn.Conv2d(in_channels=384 , out_channels=256 , kernel_size=(3,3) ,padding=(1,1),stride=(1,1))

    self.maxPool = MaxPool2d(kernel_size=(3,3) , stride=(2,2))

    self.fc1 = nn.Linear(in_features=6400 ,out_features= 4096)
    self.fc2 = nn.Linear(in_features=4096,out_features= 4096)
    self.fc3 = nn.Linear(in_features=4096,out_features= 1000)

  def forward(self ,x):
    x = self.conv1(x)
    x = F.relu(x)
    
    x = self.maxPool(x)
    
    x = self.conv2(x)
    x = F.relu(x)
    
    x = self.maxPool(x)
    
    x = self.conv3(x)
    x = F.relu(x)
    
    x = self.conv4(x)
    x = F.relu(x)
    
    x = self.conv5(x)
    x = F.relu(x)
    
    x = self.maxPool(x)
    
    x = x.reshape(x.shape[0] , -1)
    
    x = self.fc1(x)
    x = F.relu(x)
    x = nn.Dropout(p = 0.5)(x)

    x = self.fc2(x)
    x = F.relu(x)
    x = nn.Dropout(p = 0.5)(x)

    x = self.fc3(x)
    x = F.softmax(x)

    return x
```