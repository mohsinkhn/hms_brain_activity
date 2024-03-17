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
        self.bn = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c1 = ConvBnSilu(in_channels, out_channels, k, 1)
        self.c2 = ConvBnSilu(in_channels, out_channels, k + 2, 1)
        self.c3 = ConvBnSilu(in_channels, out_channels, k + 4, 1)
        self.c4 = ConvBnSilu(in_channels, out_channels, k + 6, 1)
        # self.c5 = ConvBnSilu(in_channels, out_channels, k + 8, 1)
        self.c6 = ConvBnSilu(out_channels * 4 + in_channels, out_channels, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(x)
        c3 = self.c3(x)
        c4 = self.c4(x)
        # c5 = self.c5(x)
        x = torch.cat([c1, c2, c3, c4, x], dim=1)
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
        model_name="efficientnet_b1",
        use_stem_rnn=False,
        use_feature_rnn=False,
        dropout=0.1,
    ):
        super(InceptionConv1DModel, self).__init__()
        self.conv1d_encoder = Conv1DInceptionEncoder(
            in_channels, features=features, kernel_sizes=kernel_sizes, dropout=dropout
        )
        self.conv2d = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            in_chans=in_channels,
            global_pool="",
        )
        self.in_channels = in_channels
        self.use_stem_rnn = use_stem_rnn
        self.stem_rnn = nn.GRU(features[-1], features[-1], 1, batch_first=True)
        self.use_feature_rnn = use_feature_rnn
        self.feature_rnn = nn.RNN(
            self.conv2d.num_features, self.conv2d.num_features, 1, batch_first=True
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Linear(self.conv2d.num_features * 2, out_channels)

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.relu(x)
        xc = []
        b, l, c = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(b * c, 1, l)
        x = self.conv1d_encoder(x)
        if self.use_stem_rnn:
            x = x.permute(0, 2, 1).contiguous()
            x, _ = self.stem_rnn(x)
            x = x.permute(0, 2, 1).contiguous()
        x = x.view(b, c, x.shape[1], x.shape[2])
        x = self.conv2d(x)
        if self.use_feature_rnn:
            x = x.mean(dim=2)
            x = x.permute(0, 2, 1).contiguous()
            x, _ = self.feature_rnn(x)
            x = x.permute(0, 2, 1).contiguous()
            x = x.unsqueeze(2)
        x = torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
