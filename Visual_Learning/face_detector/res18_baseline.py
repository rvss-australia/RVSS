import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torchvision import models
from torchvision.models.resnet import model_urls


class Res18Baseline(nn.Module):
    def __init__(self, args):
        self.args = args
        super(Res18Baseline, self).__init__()
        # Load pre-trained lyrn_backend
        res18 = models.resnet18(pretrained=False)
        # with torch.no_grad():
        self.res18_backbone = nn.Sequential(*list(
            res18.children())[:-6])
        self.conv2_x = nn.Sequential(*list(
            res18.children())[-6:-5])
        self.conv3_x = nn.Sequential(*list(
            res18.children())[-5:-4])
        self.conv4_x = nn.Sequential(*list(
            res18.children())[-4:-3])
        self.conv5_x = nn.Sequential(*list(
            res18.children())[-3:-2])

        # Reduces the number of channels of resnet output
        self.top_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            nn.ReLU())
        
        # backgound is considered as one additional class
        #   with label '0' by default
        self.segmentation_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.args.n_classes + 1, kernel_size=1)
        )
        
        # Define loss function here
        self.criterion = self.get_criterion()
        
    def forward(self, img):
        # Encoder
        c1 = self.res18_backbone(img)
        c2 = self.conv2_x(c1) # feature map spaticl dim 48 x 64
        c3 = self.conv3_x(c2) # feature map spaticl dim 24 x 32
        c4 = self.conv4_x(c3) # feature map spaticl dim 12 x 16
        c5 = self.conv5_x(c4) # feature map spaticl dim 6 x 8
        # Decoder
        out = self.top_conv(c5)
        out = nn.UpsamplingBilinear2d(scale_factor=16)(out)
        out = self.segmentation_conv(out)
        return out

    # Define loss function here:
    def get_criterion(self):
        return CrossEntropyLoss()
    
    # Define optimiser here
    def get_optimiser(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr,
                                weight_decay=self.args.weight_decay)

    # Learning rate is reduced to 'lr*gamma' every 'step_size' of epochs
    def get_lr_scheduler(self, optimiser):
        """
        Returns:
            This function by default returns None
        """
        return lr_scheduler.StepLR(
            optimiser, gamma=self.args.scheduler_gamma,
            step_size=self.args.scheduler_step)
        
    # A training step, do not need modification
    def step(self, batch):
        image, label = batch
        pred = self.forward(image)
        loss = self.criterion(pred, label)
        return loss