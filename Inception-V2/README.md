### Inception-ResNet-V2 :

**Paper :** [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf).

Authors : Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi , Google .

**Published in :** Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence .

**keras :**

```python
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Conv2D , MaxPool2D , Input , GlobalAveragePooling2D ,AveragePooling2D, Dense , Dropout ,Activation, Flatten , BatchNormalization

def conv_Block(prev_layer , nbr_kernels , filter_size , stride = (1,1) , padding = 'valid'):
  x = Conv2D(filters = nbr_kernels , kernel_size = filter_size , strides = stride , padding=padding) (prev_layer)
  x = BatchNormalization(axis = 3) (x)
  x = Activation(activation='relu') (x)
  return x

def StemBlock(prev_layer):
  x = conv_Block(prev_layer = prev_layer , nbr_kernels = 32 ,filter_size = 3 ,stride = 2)
  x = conv_Block(prev_layer = x , nbr_kernels = 32 ,filter_size = 3 )
  x = conv_Block(prev_layer = x , nbr_kernels = 32 ,filter_size = 3 , padding='same')
  x = MaxPool2D(pool_size = 3 , strides=2) (x)
  x = conv_Block(prev_layer = x , nbr_kernels = 80 ,filter_size = 1)
  x = conv_Block(prev_layer = x , nbr_kernels = 192 ,filter_size = 3)
  x = MaxPool2D(pool_size = (3,3) , strides=2) (x)
  return x

def InceptionBlock(prev_layer):
  branch1 = conv_Block(prev_layer = prev_layer , nbr_kernels = 64 , filter_size = 1)
  branch1 = conv_Block(prev_layer = branch1 , nbr_kernels = 64 , filter_size = 3 , padding='same')
  branch1 = conv_Block(prev_layer = branch1 , nbr_kernels = 96 , filter_size = 3 , padding='same')

  branch2 = conv_Block(prev_layer = prev_layer , nbr_kernels = 48 , filter_size = 1)
  branch2 = conv_Block(prev_layer = branch2 , nbr_kernels = 64 , filter_size = 5 , padding='same')

  branch3 = conv_Block(prev_layer = prev_layer , nbr_kernels = 96 , filter_size = 1)

  branch4 = MaxPool2D(pool_size = 3 , strides=1 , padding='same') (prev_layer)
  branch4 = conv_Block(prev_layer = branch4 , nbr_kernels = 64 , filter_size = 1)

  out = concatenate([branch1 , branch2 , branch3 , branch4] , axis=3)
  return out


def InceptionResNet_A(prev_layer):
  branch1 = conv_Block(prev_layer = prev_layer , nbr_kernels = 32 , filter_size=1 )
  branch1 = conv_Block(prev_layer = branch1 , nbr_kernels = 48 , filter_size=3 , padding='same')
  branch1 = conv_Block(prev_layer = branch1 , nbr_kernels = 64 , filter_size=3 , padding='same')

  branch2 = conv_Block(prev_layer = prev_layer , nbr_kernels = 32 , filter_size=1 )
  branch2 = conv_Block(prev_layer = branch2 , nbr_kernels = 32 , filter_size=3 , padding='same' )

  branch3 = conv_Block(prev_layer = prev_layer , nbr_kernels = 32 , filter_size=1 )
  
  out = concatenate([branch1 , branch2 , branch3] , axis = 3)

  out = conv_Block(prev_layer = out , nbr_kernels = 320 , filter_size=1 )

  out += prev_layer
  return out

def InceptionResNet_B(prev_layer):
  branch1 = conv_Block(prev_layer = prev_layer , nbr_kernels = 128 , filter_size=1 )
  branch1 = conv_Block(prev_layer = branch1 , nbr_kernels = 160 , filter_size=(1,7) , padding='same' )
  branch1 = conv_Block(prev_layer = branch1 , nbr_kernels = 192 , filter_size=(7,1) , padding='same' )

  branch2 = conv_Block(prev_layer = prev_layer , nbr_kernels = 192 , filter_size=1 )

  out = concatenate([branch1 , branch2] , axis = 3)

  out = conv_Block(prev_layer = out , nbr_kernels = 1088 , filter_size=1 )

  out += prev_layer
  return out

def InceptionResNet_C(prev_layer):
  branch1 = conv_Block(prev_layer = prev_layer , nbr_kernels = 192 , filter_size=1 )
  branch1 = conv_Block(prev_layer = branch1 , nbr_kernels = 224 , filter_size=(1,3) , padding = 'same' )
  branch1 = conv_Block(prev_layer = branch1 , nbr_kernels = 256 , filter_size=(3,1) , padding = 'same' )

  branch2 = conv_Block(prev_layer = prev_layer , nbr_kernels = 192 , filter_size=1 )

  out = concatenate([branch1 , branch2] , axis = 3)

  out = conv_Block(prev_layer = out , nbr_kernels = 2080 , filter_size=1 )

  out += prev_layer

  return out


def Reduction_A(prev_layer):
  branch1 = conv_Block(prev_layer = prev_layer , nbr_kernels = 256 , filter_size=1 )
  branch1 = conv_Block(prev_layer = branch1 , nbr_kernels = 256 , filter_size=3 , padding='same')
  branch1 = conv_Block(prev_layer = branch1 , nbr_kernels = 384 , filter_size=3 , stride=2 )

  branch2 = conv_Block(prev_layer = prev_layer , nbr_kernels = 384 , filter_size=3 , stride=2 )

  branch3 = MaxPool2D(pool_size = 3 , strides=2) (prev_layer)
  
  out = concatenate([branch1 , branch2 , branch3] , axis = 3)
  return out

def Reduction_B(prev_layer):
  branch1 = conv_Block(prev_layer = prev_layer , nbr_kernels = 256 , filter_size=1 )
  branch1 = conv_Block(prev_layer = branch1 , nbr_kernels = 288 , filter_size=3 , padding = 'same' )
  branch1 = conv_Block(prev_layer = branch1 , nbr_kernels = 320 , filter_size=3 , stride = 2 )

  branch2 = conv_Block(prev_layer = prev_layer , nbr_kernels = 256 , filter_size=1 )
  branch2 = conv_Block(prev_layer = branch2 , nbr_kernels = 384 , filter_size=3 , stride = 2 )

  branch3 = conv_Block(prev_layer = prev_layer , nbr_kernels = 256 , filter_size=1 )
  branch3 = conv_Block(prev_layer = branch3 , nbr_kernels = 288 , filter_size=3 , stride = 2)

  branch4 = MaxPool2D(pool_size = 3 , strides = 2)(prev_layer)

  out = concatenate([branch1 , branch2 , branch3 , branch4] , axis = 3)
  return out

def InceptionV2():
  input_layer = Input(shape=(299 , 299 , 3))

  out = StemBlock(prev_layer=input_layer)
  
  out = InceptionBlock(prev_layer=out)
  
  out = InceptionResNet_A(prev_layer=out)
  out = InceptionResNet_A(prev_layer=out)
  out = InceptionResNet_A(prev_layer=out)
  out = InceptionResNet_A(prev_layer=out)
  out = InceptionResNet_A(prev_layer=out)
  out = InceptionResNet_A(prev_layer=out)
  out = InceptionResNet_A(prev_layer=out)
  out = InceptionResNet_A(prev_layer=out)
  out = InceptionResNet_A(prev_layer=out)
  out = InceptionResNet_A(prev_layer=out)
  
  out = Reduction_A(prev_layer=out)
  
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  out = InceptionResNet_B(prev_layer=out)
  
  out = Reduction_B(prev_layer=out)
  
  out = InceptionResNet_C(prev_layer=out)
  out = InceptionResNet_C(prev_layer=out)
  out = InceptionResNet_C(prev_layer=out)
  out = InceptionResNet_C(prev_layer=out)
  out = InceptionResNet_C(prev_layer=out)
  out = InceptionResNet_C(prev_layer=out)
  out = InceptionResNet_C(prev_layer=out)
  out = InceptionResNet_C(prev_layer=out)
  out = InceptionResNet_C(prev_layer=out)
  out = InceptionResNet_C(prev_layer=out)
  
  out = conv_Block(prev_layer= out , nbr_kernels=1536 , filter_size=1)
  
  out = conv_Block(prev_layer= out , nbr_kernels=1536 , filter_size=8)
  
  out = Flatten()(out)
  
  out = Dense(units = 1536 , activation='relu') (out)
  out = Dense(units = 1000 ,activation='softmax') (out)

  model = Model(inputs = input_layer , outputs = out , name = 'Inception-V2')
  return model
```

**pyTorch :**

```python
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

class conv_Block(nn.Module):
  def __init__(self, in_channels , out_channels , kernel_size , stride , padding):
    super(conv_Block , self).__init__()
    self.conv = nn.Conv2d(in_channels , out_channels , kernel_size , stride , padding)
    self.batchNormalization = nn.BatchNorm2d(out_channels)
    self.activation = nn.ReLU()

  def forward(self,x):
    out = self.conv(x)
    out = self.batchNormalization(out)
    out = self.activation(out)
    return out

class StemBlock(nn.Module):
  def __init__(self,in_channels):
    super(StemBlock , self).__init__()

    self.stem = nn.Sequential(
        conv_Block(in_channels , 32 , 3 , 2 , 0),
        conv_Block(32 , 32 , 3 , 1 , 0),
        conv_Block(32 , 32 , 3 , 1 , 1),
        nn.MaxPool2d(kernel_size=3 , stride=2 , padding=0),
        conv_Block(32 , 80 , 1 , 1 , 0),
        conv_Block(80 , 192 , 3 , 1 , 0),
        nn.MaxPool2d(kernel_size=3 , stride=2 , padding=0)
    )

  def forward(self , x):

    out = self.stem(x)

    return out


class InceptionBlock(nn.Module):
  def __init__(self ,in_channels):
    super(InceptionBlock , self).__init__()

    self.branch1 = nn.Sequential(
        conv_Block(in_channels , 64 , 1 , 1 , 0),
        conv_Block(64 , 64 , 3 , 1 , 1),
        conv_Block(64 , 96 , 3 , 1 , 1)
    )

    self.branch2 = nn.Sequential(
        conv_Block(in_channels , 48 , 1 , 1 , 0),
        conv_Block(48 , 64 , 5 , 1 , 2)
    )

    self.branch3 = conv_Block(in_channels , 96 , 1 , 1  ,0)

    self.branch4 = nn.Sequential(
        nn.AvgPool2d(kernel_size=3 , stride=1 , padding=1),
        conv_Block(in_channels , 64 , 1 , 1 , 0)
    )

  def forward(self , x):

   branch1 = self.branch1(x)
   branch2 = self.branch2(x)
   branch3 = self.branch3(x)
   branch4 = self.branch4(x)   

   out = torch.cat([branch1 , branch2 , branch3 , branch4] , 1)

   return out


class Inception_ResNet_A(nn.Module):
  def __init__(self , in_channels):
    super(Inception_ResNet_A , self).__init__()

    self.branch1 = nn.Sequential(
        conv_Block(in_channels , 32 , 1 , 1 , 0),
        conv_Block(32 , 48 , 3 , 1 , 0),
        conv_Block(48 , 64 , 3 , 1 , 2)
    )

    self.branch2 = nn.Sequential(
        conv_Block(in_channels , 32 , 1 , 1 , 0),
        conv_Block(32 , 32 , 3 , 1 , 1)
    )

    self.branch3 = conv_Block(in_channels , 32 , 1 , 1 , 0)

    self.conv = conv_Block(128 , 320 , 1 , 1 , 0)

  def forward(self , x):

    branch1 = self.branch1(x)
    branch2 = self.branch2(x)
    branch3 = self.branch3(x)

    out = torch.cat([branch1 ,branch2 , branch3 ] , 1)

    out = self.conv(out)

    out += x 

    return out  

class Inception_ResNet_B(nn.Module):
  def __init__(self , in_channels):
    super(Inception_ResNet_B , self).__init__()

    self.branch1 = nn.Sequential(
        conv_Block(in_channels , 128 , 1 , 1 , 0),
        conv_Block(128 ,160 , (1,7) , 1 , (0,3) ),
        conv_Block(160 , 192 , (7,1) , 1 , (3 , 0))
    )

    self.branch2 = conv_Block(in_channels , 192 , 1 , 1 , 0)

    self.conv = conv_Block(384 ,1088 , 1 , 1 , 0 )

  def forward(self , x):

    branch1 = self.branch1(x)
    branch2 = self.branch2(x)

    out = torch.cat([branch1 , branch2] , 1)

    out = self.conv(out)
    out += x

    return out

class Inception_ResNet_C(nn.Module):
  def __init__(self , in_channels):
    super(Inception_ResNet_C , self).__init__()

    self.branch1 = nn.Sequential(
        conv_Block(in_channels , 192 , 1 , 1 , 0),
        conv_Block(192 , 224 , (1,3) , 1 , (0,1)),
        conv_Block(224 , 256 , (3,1) , 1 , (1,0))
    )

    self.branch2 = conv_Block(in_channels , 192 , 1 , 1 , 0)

    self.conv = conv_Block(448 , 2080 , 1 , 1 , 0)

  def forward(self , x):

   branch1 = self.branch1(x) 
   branch2 = self.branch2(x)
   
   out = torch.cat([branch1 ,branch2] , 1)

   out = self.conv(out)

   out += x
   return out


class Reduction_Block_A(nn.Module):
  def __init__(self , in_channels):
    super(Reduction_Block_A , self).__init__()

    self.branch1 = nn.Sequential(
        conv_Block(in_channels , 256 , 1 , 1 , 0),
        conv_Block(256 , 256 , 3 , 1 , 1),
        conv_Block(256 , 384 , 3 , 2 , 0)
    )

    self.branch2 = conv_Block(in_channels , 384 , 3 , 2 , 0)

    self.branch3 = nn.MaxPool2d(kernel_size=3 , stride=2 , padding=0)

  def forward(self , x):

    branch1 = self.branch1(x)
    branch2 = self.branch2(x)
    branch3 = self.branch3(x)  

    out = torch.cat([branch1 , branch2 , branch3] , 1)
    return out

class Reduction_Block_B(nn.Module):
  def __init__(self , in_channels):
    super(Reduction_Block_B , self).__init__()

    self.branch1 = nn.Sequential(
        conv_Block(in_channels , 256 , 1 , 1 , 0),
        conv_Block(256 , 288 , 3 , 1 , 1),
        conv_Block(288 , 320 , 3 , 2 , 0)
    )

    self.branch2 = nn.Sequential(
        conv_Block(in_channels , 256 , 1 , 1 , 0),
        conv_Block(256 , 384 , 3 , 2 , 0)
    )

    self.branch3 = nn.Sequential(
        conv_Block(in_channels , 256 , 1 , 1 , 0),
        conv_Block(256 , 288 , 3 , 2  , 0)
    )

    self.branch4 = nn.MaxPool2d(kernel_size = 3 , stride = 2 , padding = 0)
  
  def forward(self , x):

    branch1 = self.branch1(x)
    branch2 = self.branch2(x)
    branch3 = self.branch3(x)
    branch4 = self.branch4(x)

    out = torch.cat([branch1 , branch2 , branch3 , branch4] , 1)
    
    return out

class InceptionV2(nn.Module):
  def __init__(self):
    super(InceptionV2 , self).__init__()

    self.stem = StemBlock(3)

    self.inceptionBlock = InceptionBlock(192)

    self.inceptionResNet_A = Inception_ResNet_A(320)

    self.reduction_A = Reduction_Block_A(320)

    self.inceptionResNet_B = Inception_ResNet_B(1088)

    self.reduction_B = Reduction_Block_B(1088)

    self.inceptionResNet_C = Inception_ResNet_C(2080)

    self.conv = conv_Block(2080 , 1536 , 1 , 1 , 0)
    self.globalAvgPooling = conv_Block(1536 , 1536 , 8 , 1 , 0)

    self.fc1 = nn.Linear(1536 , 1536)
    self.fc2 = nn.Linear(1536 , 1000)
  
  def forward(self , x):

    out = self.stem(x)
    
    out = self.inceptionBlock(out)
    
    out = self.inceptionResNet_A(out)
    out = self.inceptionResNet_A(out)
    out = self.inceptionResNet_A(out)
    out = self.inceptionResNet_A(out)
    out = self.inceptionResNet_A(out)
    out = self.inceptionResNet_A(out)
    out = self.inceptionResNet_A(out)
    out = self.inceptionResNet_A(out)
    out = self.inceptionResNet_A(out)
    out = self.inceptionResNet_A(out)
    
    out = self.reduction_A(out)
    
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    out = self.inceptionResNet_B(out)
    
    out = self.reduction_B(out)
    
    out = self.inceptionResNet_C(out)
    out = self.inceptionResNet_C(out)
    out = self.inceptionResNet_C(out)
    out = self.inceptionResNet_C(out)
    out = self.inceptionResNet_C(out)
    out = self.inceptionResNet_C(out)
    out = self.inceptionResNet_C(out)
    out = self.inceptionResNet_C(out)
    out = self.inceptionResNet_C(out)
    out = self.inceptionResNet_C(out)
    
    out = self.conv(out)
    
    out = self.globalAvgPooling(out)
    
    out = torch.flatten(out , start_dim=1, end_dim=-1) # or you can use This out = out.reshape(out.shape[0] , -1)

    out = self.fc1(out)
    out = nn.ReLU()(out)

    out = self.fc2(out)
    out = nn.Softmax(out)

    return out
```