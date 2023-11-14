import torchvision
from .am_gcn import AM_GCN

model_dict = {'AM_GCN': AM_GCN}

def get_model(num_classes, args):
    res152 = torchvision.models.resnet101(pretrained=True)
    model = model_dict[args.model_name](res152, num_classes)
    return model