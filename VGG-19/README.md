### VGG-19 :

**Paper :** [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)

**Authors :** Karen Simonyan, Andrew Zisserman Visual Geometry Group, Department of Engineering Science, University of Oxford . 

**Published in :** 2014 .

**Model Architecture :** 

<div align="center" >
<img src="../resources/VGG19.PNG" width="100" height="400">
</div>

**Keras :**

```python
from keras.models import Model
from keras.layers import Conv2D , MaxPool2D , Input , Flatten , Dense , Dropout  


def VGG19():
    input_layer = Input(shape=(224 , 224 , 3))
    
    #Block 1
    x = Conv2D(filters = 64, kernel_size = (3,3), padding='same' , activation='relu') (input_layer)
    x = Conv2D(filters = 64, kernel_size = (3,3), padding='same' , activation='relu') (x) 
    x = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same') (x)
    
    #Block 2
    x = Conv2D(filters = 128, kernel_size = (3,3), padding='same' , activation='relu') (x)
    x = Conv2D(filters = 128, kernel_size = (3,3), padding='same' , activation='relu') (x) 
    x = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same') (x)
    
    #Block 3
    x = Conv2D(filters = 256, kernel_size = (3,3), padding='same' , activation='relu') (x)
    x = Conv2D(filters = 256, kernel_size = (3,3), padding='same' , activation='relu') (x)
    x = Conv2D(filters = 256, kernel_size = (3,3), padding='same' , activation='relu') (x)
    x = Conv2D(filters = 256, kernel_size = (3,3), padding='same' , activation='relu') (x)
    x = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same') (x)
    
    #Block 4
    x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
    x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
    x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
    x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
    x = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same') (x)
    
    #Block 5
    x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
    x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
    x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
    x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
    x = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same') (x)
    
    #Block 6
    x = Flatten()(x)
    x = Dense(units = 4096 , activation='relu') (x)
    x = Dropout(rate = 0.2)(x)
    x = Dense(units = 4096 , activation='relu') (x)
    x = Dropout(rate = 0.2)(x)
    x = Dense(units = 1000 , activation='softmax') (x)
    
    model = Model(inputs = input_layer , outputs = x , name = 'VGG-19')
    return model
```

**pyTorch :**

```python
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class VGG19(nn.Module):
  def __init__(self):
    super(VGG19 , self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3 , out_channels=64 , kernel_size=(3,3), stride=(1,1) , padding=(1,1))
    self.conv2 = nn.Conv2d(in_channels=64 , out_channels=128 , kernel_size=(3,3), stride=(1,1) , padding=(1,1))

    self.conv3 = nn.Conv2d(in_channels=128 , out_channels=128 , kernel_size=(3,3), stride=(1,1) , padding=(1,1))
    self.conv4 = nn.Conv2d(in_channels=128 , out_channels=256 , kernel_size=(3,3), stride=(1,1) , padding=(1,1))
    self.conv5 = nn.Conv2d(in_channels=256 , out_channels=256 , kernel_size=(3,3), stride=(1,1) , padding=(1,1))

    self.conv6 = nn.Conv2d(in_channels=256 , out_channels=512 , kernel_size=(3,3), stride=(1,1) , padding=(1,1))
    self.conv7 = nn.Conv2d(in_channels=512 , out_channels=512 , kernel_size=(3,3), stride=(1,1) , padding=(1,1))

    self.maxPool = nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2))

    self.fc1 = nn.Linear(in_features=25088 , out_features=4096)
    self.fc2 = nn.Linear(in_features=4096 , out_features=4096)
    self.fc3 = nn.Linear(in_features=4096 , out_features=1000)

  def forward(self,x):
    # 2 Conv Layers with 64 kernels of size 3*3  
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    
    #Max Pooling Layer with kernel size 2*2 and stride 2
    x = self.maxPool(x)
    
    # 2 Conv Layers with 128 kernels of size 3*3
    x = self.conv3(x)
    x = F.relu(x)
    x = self.conv4(x)
    x = F.relu(x)
    
    #Max Pooling Layer with kernel size 2*2 and stride 2
    x = self.maxPool(x)
    
    # 2 Conv Layers with 256 kernels of size 3*3
    x = self.conv5(x)
    x = F.relu(x)
    x = self.conv5(x)
    x = F.relu(x)
    x = self.conv5(x)
    x = F.relu(x)    
    x = self.conv6(x)
    x = F.relu(x)
    
    #Max Pooling Layer with kernel size 2*2 and stride 2
    x = self.maxPool(x)
    
    # 4 Conv Layers with 512 kernels of size 3*3
    x = self.conv7(x)
    x = F.relu(x)
    x = self.conv7(x)
    x = F.relu(x)
    x = self.conv7(x)
    x = F.relu(x)
    x = self.conv7(x)
    x = F.relu(x)
    
    #Max Pooling Layer with kernel size 2*2 and stride 2
    x = self.maxPool(x)
    
    # 4 Conv Layers with 512 kernels of size 3*3
    x = self.conv7(x)
    x = F.relu(x)
    x = self.conv7(x)
    x = F.relu(x)
    x = self.conv7(x)
    x = F.relu(x)
    x = self.conv7(x)
    x = F.relu(x)
    
    #Max Pooling Layer with kernel size 2*2 and stride 2
    x = self.maxPool(x)

    x = x.reshape(x.shape[0] , -1)

    #Fully Connected Layer With 4096 Units  
    x = self.fc1(x)
    x = F.relu(x)

    #Fully Connected Layer With 4096 Units
    x = self.fc2(x)
    x = F.relu(x)

    #Fully Connected Layer With 1000 Units
    x = self.fc3(x)
    x = F.softmax(x)

    return x
```