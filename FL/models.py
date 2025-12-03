import torch
from torch import nn
import torch.nn.functional as F
import torchvision as tv

class CNNMNIST(nn.Module):
    def __init__(self):
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def setup_model(model_architecture, num_classes = None, tokenizer = None, embedding_dim = None):

        available_models = {
            "CNNMNIST": CNNMNIST,
            "ResNet18" : tv.models.resnet18,
            "VGG16" : tv.models.vgg16,
            "DN121": tv.models.densenet121,
        }
        print('--> Creating {} model.....'.format(model_architecture))
        # variables in pre-trained ImageNet models are model-specific.
        if "ResNet18" in model_architecture:
            model = available_models[model_architecture]()
            n_features = model.fc.in_features
            model.fc = nn.Linear(n_features, num_classes)
        elif "VGG16" in model_architecture:
            model = available_models[model_architecture]()
            n_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(n_features, num_classes)
        else:
            model = available_models[model_architecture]()

        if model is None:
            print("Incorrect model architecture specified or architecture not available.")
            raise ValueError(model_architecture)
        print('--> Model has been created!')
        return model
    
def create_model(model_name, num_classes, tokenizer=None, embedding_dim=100):

    if model_name == "CNNMNIST":
        return CNNMNIST()

    elif model_name == "ResNet18":
        model = tv.models.resnet18(pretrained=False)  # 不加载预训练权重
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_name == "VGG16":
        model = tv.models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(512, num_classes)  # VGG16 classifier[6] 是最后一层
        return model

    elif model_name == "DN121":  # DenseNet121
        model = tv.models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unknown model: {model_name}")