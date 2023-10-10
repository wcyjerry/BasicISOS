# How to use your custom model
This is an tutorial on how to use your own model. 
### 1. define your own model 
Create your own model under basic/networks, eg. I create an custom_model.py here
There is only one more thing you need to do different from normal codes is:
```
from basicseg.utils.registry import NET_REGISTRY
```
and then add a line of decorator code before your network definition like:
```
@NET_REGISTRY.register() 
class My_Model(nn.Module)
```
### 2. Modified your train option yaml
As showed in custom_exp.yaml,
modified type under net to My_Model(the name should be the same as your class name),
I have defined three params in My_Model: in_c, out_c, base_dim
You can easily parse different params to create different model to do ablation experiments.
Like change base_dim from 32 to 64 to create a bigger model.

### 3. train with new option
python train.py --opt ./custom_configs/custom_exp.yaml

中文教程
### 1. 定义你的网络模型
在basic/networks下面创建你自己的网络模型， 例如：在这里我创建了一个custom_model.py 
跟通常的网络定义代码比，只有一个额外的步骤需要做：
在你的网络模型前用装饰器装饰一下你的网络，当然你得在开头加上
```
fro basicseg.utils.registry import NET_REGISTRY
@NET_REGISTRY.register()
class My_Model(nn.Moduler)
```
### 2. 更改训练yaml文件
如同在custom_exp.yaml下显示的， 
要使用刚才我定义的模型，更改net下面的type为My_Model(需要跟你的网络类名字完全一样)，
我在网络初始化的时候定义了3个参数:in_c, out_c, base_dim
你可以很容易地通过传递不同的参数来创建不同的网络模型，这更有助于做消融实验
比如，在下面的 base_dim 从32改到64，我就可以创建一个更大的模型

### 3. 使用刚才的配置训练
python trian.py --opt ./custom_configs/custom_exp.yaml