# Pytorch Model Zoo

## Outline

This part contains model configs as follows:

* DenseNet - DenseNet121, DenseNet169, DenseNet201, DenseNet161, densenet_cifar
* DPN (Dual Path Network) - DPN26, DPN92
* GoogleNet - GoogleNet
* MobileNet - MobileNet
* MobileNetV2 - MobileNetV2
* PNASNet - PNASNetA, PNASNetB
* PreActResNet - PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
* ResNet - ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
* ResNext - ResNeXt29_2x64d, ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d
* SENet - SENet18
* ShuffleNet - ShuffleNetG2, ShuffleNetG3
* ShuffleNetV2 - ShuffleNetV2
* VGG - VGG11, VGG13, VGG16, VGG19

## Usage

Simply select 

```
from models import model_factory
model = model_factory(model_name)
```