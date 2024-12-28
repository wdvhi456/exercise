import torch
import torch.nn as nn
class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3*3, stride=1,
                                 padding=0)  # 由572*572*3变成了570*570*64
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)  # 由570*570*64变成了568*568*64
        self.relu1_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.relu1_1(x1)
        x2 = self.conv1_2(x1)
        x2 = self.relu1_2(x2)
        return x1





input_data = torch.randn([1,3, 572,572])
unet=Unet()
output_data=unet(input_data)
print(output_data.shape)
