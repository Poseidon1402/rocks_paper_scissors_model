import torch.nn as nn

NUM_CLASSES = 3

class CNNModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # 3 x 128 x 128 -> conv blocks -> flatten -> FC
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 3 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 128 -> 64
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 64 -> 32
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    # 64 -> 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 32 -> 16
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),    # 128 -> 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2), # 16 -> 8
        )

        # After 3 poolings : 3 x 128 x 128 -> 256 x 8 x 8
        self.flatten_dim = 256 * 8 * 8

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes) # logits for each class
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(x.size(0), -1)   # flatten
        x = self.fc(x)
        return x