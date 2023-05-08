class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, num_channels):
        super(UpsampleBlock, self).__init__()

        self.conv = nn.Conv2d(num_channels, 4*num_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out

class SRResNet(nn.Module):
    def __init__(self, upscale_factor=2, num_channels=64, num_residual_blocks=16):
        super(SRResNet, self).__init__()

        # First convolutional layer
        #nn.Conv2d(number of input channels,number of output channels,kernel size,padding)
        #applied symmetric padding, the size of output is the same as that of input
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=9, padding=4)

        # Residual blocks
        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualBlock(num_channels))

        self.res_blocks = nn.Sequential(*res_blocks)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        # Upscaling layers
        upsample_blocks = []
        for _ in range(int(upscale_factor/2)):
            upsample_blocks.append(UpsampleBlock(num_channels))

        self.upsample = nn.Sequential(*upsample_blocks)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(num_channels, 3, kernel_size=9, padding=4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.res_blocks(out)
        out = self.conv2(out)
        out = self.upsample(out)
        out = self.conv3(out)
        return out

