import torch.nn as nn
import torch
import numpy as np
from transformers import ResNetForImageClassification, Mask2FormerForUniversalSegmentation
import albumentations as A
from torchvision import transforms as T
import cv2
import matplotlib.pyplot as plt

RESNET_MODEL_NAME = "./base_models/microsoft_resnet50"
M2F_MODEL_NAME = "./base_models/facebook-m2f_swin_large"

normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRANSFORM_MYCNN = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    normalize
    ])

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Classifier, self).__init__()
        
        #
        model = ResNetForImageClassification.from_pretrained(RESNET_MODEL_NAME)
        model.requires_grad_(False)

        # ResNet backbone
        self.backbone = model.resnet

        #
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(2048, 512),
            nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        encoder_output = self.backbone(x).pooler_output
        out = self.fc(encoder_output)
        return out

class Mask2FormerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(Mask2FormerClassifier, self).__init__()
        
        #
        m2f = Mask2FormerForUniversalSegmentation.from_pretrained(M2F_MODEL_NAME)
        m2f.requires_grad_(False)
        self.bb_features = 1536

        # M2F backbone
        self.embeddings = m2f.model.pixel_level_module.encoder.embeddings
        self.encoder = m2f.model.pixel_level_module.encoder.encoder
        self.layernorm = nn.LayerNorm(self.bb_features)
        self.pooler = nn.AdaptiveAvgPool1d(1)

        #
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(self.bb_features, 512),
            nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(256, num_classes)
    )

    def forward(self, x):
        embedding_output, input_dimensions = self.embeddings(x)
        encoder_outputs = self.encoder(embedding_output, input_dimensions)
        
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)

        pooled_output = self.pooler(sequence_output.transpose(1, 2))

        out = self.fc(pooled_output)

        return out
    
class MyCNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MyCNNClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            # 224x224x3
            nn.Conv2d(3, 128, kernel_size=4, stride=4),
            # 56x56x128
            nn.SiLU(),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=4, stride=4),
            # 12x12x128
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            # 5x5x256
            nn.SiLU(),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=3, stride=2),
            # 2x2x512
        )

        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
    
    def forward(self, x):
        x1 = self.conv_layers(x)
        x2 = self.linear_layers(x1)
        return x2