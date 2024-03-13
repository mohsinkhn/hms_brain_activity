import timm
import torch
from torch import nn


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups=1,
        dilation=1,
        padding=0,
        dropout=0.1,
    ):
        super(Conv1DBlock, self).__init__()
        self.drop = nn.Dropout1d(dropout)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
            dilation=dilation,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.SELU()

    def forward(self, x):
        x = self.drop(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthwiseConv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation=1,
        padding=0,
        dropout=0.1,
    ):
        super(DepthwiseConv1DBlock, self).__init__()
        self.drop = nn.Dropout1d(dropout)
        self.depth_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            dilation=dilation,
            padding=padding,
        )
        self.point_conv = nn.Conv1d(
            in_channels, out_channels, 1, 1, groups=1, dilation=1, padding=0
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.SELU()

    def forward(self, x):
        x = self.drop(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Refer to interveted residual block
# https://github.com/pytorch/vision/blob/1aef87d01eec2c0989458387fa04baebcc86ea7b/torchvision/models/mobilenet.py#L45
class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        norm_layer=None,
        dropout=0.1,
    ):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv1d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            nn.SELU(inplace=True),
            nn.Dropout1d(dropout),
        )


class InvertedResidual(nn.Module):
    def __init__(
        self, inp, oup, stride, kernel_size, expand_ratio, norm_layer=None, dropout=0.1
    ):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvBNReLU(
                    inp,
                    hidden_dim,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    dropout=dropout,
                )
            )
        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    dropout=dropout,
                ),
                # pw-linear
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class CNNStemBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout=0.1):
        super(CNNStemBlock, self).__init__()
        self.conv1 = Conv1DBlock(
            in_channels, out_channels, kernel_size, stride, dropout
        )
        self.conv2 = Conv1DBlock(out_channels, out_channels, kernel_size, 1, dropout)
        self.conv3 = Conv1DBlock(out_channels, out_channels, kernel_size, 1, dropout)
        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.pool(x)
        return x


class InvertResidualGRU(nn.Module):
    def __init__(
        self,
        in_channels,
        num_outputs,
        stem_kernel=3,
        kernel_sizes=[5, 3, 3, 3, 3],
        filters=[16, 32, 64, 128, 256],
        strides=[2, 1, 2, 1, 1],
        expand_ratio=2,
        latent_steps=128,
        dropout=0.1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        res_layers = []
        self.stem = CNNStemBlock(in_channels, filters[0], stem_kernel, 2, dropout)
        in_channels = filters[0]
        for kernel_size, fil, stride in zip(kernel_sizes, filters, strides):
            res_layers.append(
                InvertedResidual(
                    in_channels, fil, stride, kernel_size, expand_ratio, dropout=dropout
                )
            )
            res_layers.append(
                InvertedResidual(
                    fil, fil, 1, kernel_size, expand_ratio, dropout=dropout
                )
            )
            in_channels = fil
        self.features = nn.Sequential(*res_layers)
        self.gru = nn.GRU(
            filters[-1], filters[-1], 1, batch_first=True, bidirectional=True
        )
        self.drop = nn.Dropout1d(dropout)
        self.avgpool1 = nn.AdaptiveAvgPool1d(latent_steps)
        self.maxpool1 = nn.AdaptiveMaxPool1d(latent_steps)
        self.avgpool2 = nn.AdaptiveAvgPool1d(1)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(filters[-1] * 2 * 2, num_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool1(x)
        x = x.permute(0, 2, 1)
        # x = self.drop(x)
        # Counting ??
        x, _ = self.gru(x)
        x = x.permute(0, 2, 1)
        x1 = self.avgpool2(x).squeeze()
        x2 = self.maxpool2(x).squeeze()
        x = torch.cat([x1, x2], dim=-1)
        x = self.classifier(x)
        return x


class ConvRNNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dropout=0.1
    ):
        super(ConvRNNBlock, self).__init__()
        self.conv1 = Conv1DBlock(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            out_channels,
            out_channels,
            1,
            batch_first=True,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        x = self.drop(x)
        x, _ = self.rnn(x)
        return x


class ConvRNNModel(nn.Module):
    """Multiple convrnnblocks stacked in residual fashion followed by a linear layer"""

    def __init__(
        self,
        in_channels,
        latent_channels,
        num_outputs,
        kernel_size,
        stride,
        padding,
        dropout=0.1,
        num_blocks=1,
    ):
        super(ConvRNNModel, self).__init__()
        self.fc1 = nn.Linear(in_channels, latent_channels)
        self.conv1 = Conv1DBlock(in_channels, latent_channels, kernel_size, 2, 0)
        self.pad = nn.ZeroPad1d(((kernel_size - 1), 0))
        self.conv2 = Conv1DBlock(latent_channels, latent_channels, kernel_size, 2, 0)
        self.conv3 = Conv1DBlock(latent_channels, latent_channels, kernel_size, 1, 0)
        self.conv4 = Conv1DBlock(latent_channels, latent_channels, kernel_size, 1, 0)
        self.drop = nn.Dropout(dropout)
        self.rnn1 = nn.LSTM(
            latent_channels,
            latent_channels,
            1,
            batch_first=True,
            bidirectional=True,
        )
        self.downsample1 = nn.AdaptiveAvgPool1d(2000)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(latent_channels * 2 * 2, num_outputs)

    def forward(self, x):
        # x = self.fc1(x)
        x = x.permute(0, 2, 1)
        # x = self.downsample1(x)
        x = self.pad(x)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.pad(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.pad(x)
        x = self.conv3(x)
        x = self.drop(x)
        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn1(x)
        b = x.shape[0]
        x1 = self.max_pool(x.permute(0, 2, 1)).squeeze()
        x2 = self.avg_pool(x.permute(0, 2, 1)).squeeze()
        x = torch.cat([x1, x2], dim=-1)
        x = self.fc(x)
        return x


# It seems features need to be captured from 2 samples to 200 samples
# We can provide different sub-samples of the data
# Many be feautures extracted from CNN can also be normalized
# Does averaging make sense? depthwise separable convolutions make more sense here
# Yolo like multi scale feature extraction?


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups,
        dropout=0.1,
        dilations=[1, 2, 4, 8],
    ):
        super(MultiScaleBlock, self).__init__()
        layers = []
        for dilation in dilations:
            layers.append(
                DepthwiseConv1DBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    dropout=dropout,
                    dilation=dilation,
                    padding=(kernel_size - 1) // 2 * dilation,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.point_conv = nn.Conv1d(
            out_channels * len(dilations), out_channels, 1, 1, groups=1, dilation=1
        )

    def forward(self, x):
        out = []
        for layer in self.layers:
            out.append(layer(x))
        out = torch.cat(out, dim=1)
        out = self.point_conv(out)
        return out


class MultiScaleModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        strides=[2, 1, 1, 1],
        feature_dims=[16, 32, 64, 128],
        dilations=[1, 2, 4, 8],
        latent_len=6,
    ):
        super(MultiScaleModel, self).__init__()
        self.block1 = MultiScaleBlock(
            in_channels,
            feature_dims[0],
            kernel_size,
            strides[0],
            1,
            dilations=dilations,
        )
        self.max_pool = nn.MaxPool1d(2, 2)
        self.block2 = MultiScaleBlock(
            feature_dims[0],
            feature_dims[1],
            kernel_size,
            strides[1],
            1,
            dilations=dilations,
        )
        self.block3 = MultiScaleBlock(
            feature_dims[1],
            feature_dims[2],
            kernel_size,
            strides[2],
            1,
            dilations=dilations,
        )
        self.block4 = MultiScaleBlock(
            feature_dims[2],
            feature_dims[3],
            kernel_size,
            strides[3],
            1,
            dilations=dilations,
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(latent_len)
        self.classifier = nn.Linear(feature_dims[-1] * latent_len, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.block1(x)
        x = self.max_pool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride, dropout=0.1):
        super(InceptionBlock, self).__init__()
        self.layers = []
        for kernel_size in kernel_sizes:
            layer = Conv1DBlock(
                in_channels, out_channels, kernel_size, stride, dropout=dropout
            )
            self.layers.append(layer)

    def forward(self, x):
        out = []
        for layer in self.layers:
            out.append(layer(x))
        out = torch.cat(out, dim=1)
        return out


class Conv2DNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, backbone, stem_filters=[3, 5, 7, 11, 15, 19]
    ):
        super(Conv2DNet, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True)
        self.inception_stem = InceptionBlock(
            in_channels, 8, kernel_sizes=[3 + 2 * i for i in range(16)], stride=2
        )
        self.conv_stem = nn.Conv1d(in_channels, 32, 5, 2, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(stem_filters[-1], out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
