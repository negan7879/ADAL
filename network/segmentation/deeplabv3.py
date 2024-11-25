from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.models.utils import load_state_dict_from_url
from .mobilenet import MobileNetV2
from ._deeplab import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCN, FCNHead
import torch.nn as nn
from torchvision.models import resnet,densenet121,densenet
from .wresnet import wrn_16_2,wrn_16_1
import timm



def _segm_denseNet(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    # backbone = resnet.__dict__[backbone_name](
    #     pretrained=pretrained_backbone,
    #     replace_stride_with_dilation=[False, True, True])

    # densenet121
    backbone = densenet.__dict__[backbone_name](pretrained=pretrained_backbone)
    # backbone = densenet121(pretrained=pretrained_backbone)

    # return_layers = {'layer4': 'out'}
    return_layers = {'features': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model_map = {
        'deeplab': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    if backbone_name == "densenet121":
        inplanes = 1024   #  1024  2208
    elif backbone_name == "densenet161":
        inplanes = 2208
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, None)
    return model


def _segm_mobilenet(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    backbone = MobileNetV2(pretrained=pretrained_backbone)

    return_layers = {'features': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model_map = {
        'deeplab': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }

    inplanes = 320
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, None)
    return model


def _segm_wide_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    # backbone = resnet.__dict__[backbone_name](
    #     pretrained=pretrained_backbone,
    #     replace_stride_with_dilation=[False, True, True])

    # backbone = densenet121(pretrained=pretrained_backbone)
    # backbone = wrn_16_2(num_classes)  # 128
    backbone = wrn_16_1(num_classes) # 64

    return_layers = {'block3': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model_map = {
        'deeplab': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = 64
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, None)
    return model


def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    # backbone = densenet121(pretrained=pretrained_backbone)

    return_layers = {'layer4': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model_map = {
        'deeplab': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, None)
    return model

def deeplabv3_mobilenet(progress=True,num_classes=21, aux_loss=None, dropout_p=0.0, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _segm_mobilenet("deeplab", "mobilenet_v2", num_classes, aux_loss, **kwargs)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout_p
    return model

def deeplabv3_resnet34(progress=True, num_classes=21, dropout_p=0.0, aux_loss=None, **kwargs):
    model = _segm_resnet("deeplab", backbone_name='resnet34', num_classes=num_classes, aux=aux_loss, **kwargs)
    # model = timm.create_model('resnet34', pretrained=True)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout_p
    return model



def deeplabv3_denset121(progress=True, num_classes=21, dropout_p=0.0, aux_loss=None, **kwargs):
    model = _segm_denseNet("deeplab", backbone_name='densenet121', num_classes=num_classes, aux=aux_loss, **kwargs)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout_p
    return model


def deeplabv3_denset161(progress=True, num_classes=21, dropout_p=0.0, aux_loss=None, **kwargs):
    model = _segm_denseNet("deeplab", backbone_name='densenet161', num_classes=num_classes, aux=aux_loss, **kwargs)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout_p
    return model



def deeplabv3_resnet50(progress=True, num_classes=21, dropout_p=0.0, aux_loss=None, **kwargs):
    model = _segm_resnet("deeplab", backbone_name='resnet50', num_classes=num_classes, aux=aux_loss, **kwargs)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout_p
    return model

def deeplabv3_resnet101(progress=True, num_classes=21, dropout_p=0.0, aux_loss=None, **kwargs):
    model = _segm_resnet("deeplab", backbone_name='resnet101', num_classes=num_classes, aux=aux_loss, **kwargs)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout_p
    return model


def deeplabv3_resnet152(progress=True, num_classes=21, dropout_p=0.0, aux_loss=None, **kwargs):
    model = _segm_resnet("deeplab", backbone_name='resnet152', num_classes=num_classes, aux=aux_loss, **kwargs)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout_p
    return model


def deeplabv3_wide_resnet(progress=True, num_classes=21, dropout_p=0.0, aux_loss=None, **kwargs):
    model = _segm_wide_resnet("deeplab", backbone_name='resnet152', num_classes=num_classes, aux=aux_loss, **kwargs)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout_p
    return model