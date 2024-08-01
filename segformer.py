# segformer.py

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class SegformerModel(nn.Module):
    def __init__(self, num_classes):
        super(SegformerModel, self).__init__()
        # Load pre-trained SegFormer model
        self.model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
        # Modify classifier head to match the number of classes
        self.model.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Forward pass
        return self.model(x).logits
