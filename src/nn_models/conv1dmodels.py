import timm
import torch
from torch import nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ConvBnSilu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        use_bnorm=True,
        bnorm="batch",
        inp_raw_channels=16,
    ):
        super(ConvBnSilu, self).__init__()
        self.use_bnorm = use_bnorm
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.bnorm = bnorm
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bnorm:
            x = self.bn(x)
        x = self.silu(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, k, use_bnorm, bnorm="batch", *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.c1 = ConvBnSilu(
            in_channels, out_channels, k, 1, use_bnorm=use_bnorm, bnorm=bnorm
        )
        self.c2 = ConvBnSilu(
            in_channels, out_channels, k + 2, 1, use_bnorm=use_bnorm, bnorm=bnorm
        )
        self.c3 = ConvBnSilu(
            in_channels, out_channels, k + 4, 1, use_bnorm=use_bnorm, bnorm=bnorm
        )
        self.c4 = ConvBnSilu(
            in_channels, out_channels, k + 8, 1, use_bnorm=use_bnorm, bnorm=bnorm
        )
        self.c5 = ConvBnSilu(
            in_channels, out_channels, k + 16, 1, use_bnorm=use_bnorm, bnorm=bnorm
        )
        self.c6 = ConvBnSilu(
            out_channels * 5 + in_channels,
            out_channels,
            1,
            use_bnorm=use_bnorm,
            bnorm=bnorm,
        )

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(x)
        c3 = self.c3(x)
        c4 = self.c4(x)
        c5 = self.c5(x)
        x = torch.cat([c1, c2, c3, c4, c5, x], dim=1)
        x = self.c6(x)
        return x


class InceptionResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        feat,
        kernel_size,
        dropout,
        use_bnorm=True,
        bnorm="batch",
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.drop = nn.Dropout1d(dropout)
        self.conv1 = ConvBnSilu(
            in_channels, feat, 1, 1, use_bnorm=use_bnorm, bnorm=bnorm
        )
        self.inception = InceptionBlock(
            in_channels, feat, kernel_size, use_bnorm=use_bnorm, bnorm=bnorm
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.drop(x)
        x = self.inception(x)
        x = self.pool(x)
        return x


class Conv1DInceptionEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        features=[8, 16, 32, 64, 128, 256, 512],
        kernel_sizes=[3, 3, 5, 5, 5, 5, 5],
        dropout=0.1,
        use_bnorm=True,
        bnorm="batch",
    ):
        super(Conv1DInceptionEncoder, self).__init__()
        blocks = []
        for feat, kernel_size in zip(features, kernel_sizes):
            blocks.append(
                InceptionResBlock(
                    in_channels,
                    feat,
                    kernel_size,
                    dropout,
                    use_bnorm=use_bnorm,
                    bnorm=bnorm,
                )
            )
            in_channels = feat
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class InceptionConv1DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        features,
        kernel_sizes,
        model_name="efficientnet_b1",
        pretrained=True,
        use_stem_rnn=False,
        use_feature_rnn=False,
        dropout=0.1,
        use_bnorm=True,
        bnorm="batch",
        conv2d_stride=2,
        old=False,
    ):
        super(InceptionConv1DModel, self).__init__()
        self.conv1d_encoder = Conv1DInceptionEncoder(
            1,
            features=features,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            use_bnorm=use_bnorm,
            bnorm=bnorm,
        )
        self.conv2d = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels,
            global_pool="",
        )
        # self.conv2d.conv_stem.stride = (conv2d_stride, conv2d_stride)
        self.in_channels = in_channels
        self.use_stem_rnn = use_stem_rnn
        # self.stem_rnn = nn.GRU(features[-1], features[-1], 1, batch_first=True)
        self.use_feature_rnn = use_feature_rnn
        # self.feature_rnn = nn.RNN(
        #     self.conv2d.num_features, self.conv2d.num_features, 1, batch_first=True
        # )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Linear(self.conv2d.num_features * 2, out_channels)
        self.old = old

    def forward_features(self, x):
        b, l, c = x.shape
        x = x.permute(0, 2, 1).contiguous().view(b * c, 1, l)
        x = self.conv1d_encoder(x)
        x = x.view(b, c, x.shape[1], x.shape[2])
        x = self.conv2d(x)
        if self.old:
            x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        else:
            x = torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)
        # x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


class InceptionStackedModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        features,
        kernel_sizes,
        model_name="efficientnet_b1",
        pretrained=True,
        use_stem_rnn=False,
        use_feature_rnn=False,
        dropout=0.1,
        use_bnorm=True,
        bnorm="batch",
        conv2d_stride=2,
    ):
        super(InceptionStackedModel, self).__init__()
        # self.conv1d_encoder1 = Conv1DInceptionEncoder(
        #     1,
        #     features=[features[0]],
        #     kernel_sizes=[kernel_sizes[0]],
        #     dropout=dropout,
        #     use_bnorm=use_bnorm,
        #     bnorm=bnorm,
        # )
        self.conv1d_encoder = Conv1DInceptionEncoder(
            1,
            features=features,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            use_bnorm=use_bnorm,
            bnorm=bnorm,
        )
        self.conv2d = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels,
            global_pool="",
            img_size=(128, 304),
        )
        # self.conv2d.conv_stem.stride = (conv2d_stride, conv2d_stride)
        self.in_channels = in_channels
        self.use_stem_rnn = use_stem_rnn
        self.stem_rnn = nn.GRU(
            features[-1], 48, 1, batch_first=True, bidirectional=True
        )
        self.use_feature_rnn = use_feature_rnn
        # self.feature_rnn = nn.RNN(
        #     self.conv2d.num_features, self.conv2d.num_features, 1, batch_first=True
        # )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Linear(self.conv2d.num_features * 2, out_channels)

    def forward_features(self, x):
        b, l, c = x.shape
        x = x.permute(0, 2, 1).contiguous().view(b * c, 1, l)
        x = self.conv1d_encoder(x)
        x = x.view(b, c, x.shape[1], x.shape[2])
        x = x[:, :, :, 4:-4]
        x = self.conv2d(x)
        x = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x
