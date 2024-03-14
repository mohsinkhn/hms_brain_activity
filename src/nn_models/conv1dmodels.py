import timm
import torch
from torch import nn


class ConvBnSilu(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding="same"
    ):
        super(ConvBnSilu, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c1 = ConvBnSilu(in_channels, out_channels, k // 4 + 1, 1)
        self.c2 = ConvBnSilu(in_channels, out_channels, k // 2 + 1, 1)
        self.c3 = ConvBnSilu(in_channels, out_channels, k, 1)
        self.c4 = ConvBnSilu(in_channels, out_channels, k * 2 + 1, 1)
        self.c5 = ConvBnSilu(in_channels, out_channels, k * 4 + 1, 1)
        self.c6 = ConvBnSilu(out_channels * 5 + in_channels, out_channels, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(x)
        c3 = self.c3(x)
        c4 = self.c4(x)
        c5 = self.c5(x)
        x = torch.cat([c1, c2, c3, c4, c5, x], dim=1)
        x = self.c6(x)
        return x


class Conv1DEncoder(nn.Module):
    # Input shape: (batch_size, in_channels, 8192)
    # Output shape: (batch_size, in_channels, 256, 256)
    def __init__(
        self,
        in_channels,
        features=[8, 16, 32, 64, 128, 256, 512],
        kernel_sizes=[3, 3, 5, 5, 5, 5, 5],
        dropout=0.1,
    ):
        super(Conv1DEncoder, self).__init__()
        blocks = []
        in_channels = 1
        for feat, kernel_size in zip(features, kernel_sizes):
            blocks.append(
                nn.Sequential(
                    nn.Dropout1d(dropout),
                    ConvBnSilu(in_channels, feat, kernel_size, 1, 0),
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                )
            )
            in_channels = feat
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Conv1DInceptionEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        features=[8, 16, 32, 64, 128, 256, 512],
        kernel_sizes=[3, 3, 5, 5, 5, 5, 5],
        dropout=0.1,
    ):
        super(Conv1DInceptionEncoder, self).__init__()
        blocks = []
        in_channels = 1
        for feat, kernel_size in zip(features, kernel_sizes):
            blocks.append(
                nn.Sequential(
                    nn.Dropout1d(dropout),
                    InceptionBlock(in_channels, feat, kernel_size),
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                )
            )
            in_channels = feat
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class DeepConv1DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        features,
        kernel_sizes,
        conv2d_features=32,
        dropout=0.1,
    ):
        super(DeepConv1DModel, self).__init__()
        self.conv1d_encoder = Conv1DEncoder(
            in_channels, features=features, kernel_sizes=kernel_sizes, dropout=dropout
        )
        self.conv2d = nn.Conv2d(
            in_channels=8,
            out_channels=conv2d_features,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.bnorm = nn.BatchNorm2d(conv2d_features)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(conv2d_features * features[-1] * 2, out_channels)

    def forward(self, x):
        xc = []
        for i in range(x.shape[2]):
            xc.append(self.conv1d_encoder(x[:, :, i].unsqueeze(1)))
        xc = torch.stack(xc, dim=1)
        x = self.conv2d(xc)
        x = self.bnorm(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1, x.size(-1))
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class InceptionConv1DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        features,
        kernel_sizes,
        conv2d_features=32,
        rnn_dim=128,
        dropout=0.1,
    ):
        super(InceptionConv1DModel, self).__init__()
        self.conv1d_encoder = Conv1DInceptionEncoder(
            in_channels, features=features, kernel_sizes=kernel_sizes, dropout=dropout
        )
        self.ekg_encoder = Conv1DInceptionEncoder(
            in_channels,
            features=[8] * (len(features) - 1) + [features[-1]],
            kernel_sizes=[5] * len(kernel_sizes),
            dropout=dropout,
        )
        self.conv2d = timm.create_model(
            "efficientnet_b1",
            pretrained=True,
            num_classes=0,
            in_chans=in_channels,
            global_pool="",
        )
        self.in_channels = in_channels
        self.bnorm = nn.BatchNorm2d(conv2d_features)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Linear(1280 * 2, out_channels)

    def forward(self, x):
        xc = []
        x1 = x[:, :, : self.in_channels]
        for i in range(x1.shape[2]):
            xc.append(self.conv1d_encoder(x1[:, :, i].unsqueeze(1)))
        x2 = x[:, :, self.in_channels].unsqueeze(1)
        # x2 = self.ekg_encoder(x2)
        # xc.append(x2)
        x = torch.stack(xc, dim=1)
        x = self.conv2d(x)
        # x = self.avg_pool(x)
        x = torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)
        x = x.view(x.size(0), -1)
        # x = self.bnorm(x)
        # x = self.relu(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1, x.size(-1))
        # x1 = self.avg_pool(x)
        # x2 = self.max_pool(x)
        # x = torch.cat([x1, x2], dim=1)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
