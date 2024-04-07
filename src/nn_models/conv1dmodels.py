import numpy as np
import timm
import torch
from torch import nn
import torch.nn.functional as F


class BlurPool1D(nn.Module):
    def __init__(self, channels, pad_type="reflect", filt_size=3, stride=2, pad_off=0):
        super(BlurPool1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if self.filt_size == 1:
            a = np.array(
                [
                    1.0,
                ]
            )
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer("filt", filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride]
        else:
            return F.conv1d(
                self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1]
            )


def get_pad_layer_1d(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad1d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad1d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad1d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer


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
        if use_bnorm:
            if bnorm == "batch":
                self.bn = nn.BatchNorm1d(out_channels)
            elif bnorm == "group":
                self.bn = nn.GroupNorm(4, out_channels)
            elif bnorm == "instance":
                self.bn = nn.InstanceNorm1d(out_channels)
            elif bnorm == "layer":
                self.bn = nn.LayerNorm(out_channels)
            elif bnorm == "none":
                self.bn = None
            else:
                raise ValueError("Batchnorm type not recognized")
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
        # attention block
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
        # self.pool = BlurPool1D(feat, filt_size=3, stride=2, pad_off=0)

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

    def forward_features(self, x, spec_x=None):
        b, l, c = x.shape
        x = x.permute(0, 2, 1).contiguous().view(b * c, 1, l)
        x = self.conv1d_encoder(x)
        x = x.view(b, c, x.shape[1], x.shape[2])  # b, 16, 128, 312 -
        x = self.conv2d(x)
        if self.old:
            x = torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)
        else:
            x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)

        # x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x, spec_x=None):
        x = self.forward_features(x, spec_x=None)
        x = self.classifier(x)
        return x

    def freeze_batch_norm(self):
        layers = [mod for mod in self.conv2d.children()]
        for layer in layers:
            if isinstance(layer, nn.BatchNorm2d):
                # print(layer)
                for param in layer.parameters():
                    param.requires_grad = False

            elif isinstance(layer, nn.Sequential):
                for seq_layers in layer.children():
                    if isinstance(layer, nn.BatchNorm2d):
                        # print(layer)
                        param.requires_grad = False


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
            img_size=(96, 288),
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

    def forward_features(self, x, spec_x=None):
        b, l, c = x.shape
        x = x.permute(0, 2, 1).contiguous().view(b * c, 1, l)
        x = self.conv1d_encoder(x)
        x = x.view(b, c, x.shape[1], x.shape[2])
        x = x[:, :, :, 12:-12]
        x = self.conv2d(x)
        x = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x, spec_x=None):
        x = self.forward_features(x, spec_x)
        x = self.classifier(x)
        return x

    def freeze_batch_norm(self):
        layers = [mod for mod in self.conv2d.children()]
        for layer in layers:
            if isinstance(layer, nn.BatchNorm2d):
                # print(layer)
                for param in layer.parameters():
                    param.requires_grad = False

            elif isinstance(layer, nn.Sequential):
                for seq_layers in layer.children():
                    if isinstance(layer, nn.BatchNorm2d):
                        # print(layer)
                        param.requires_grad = False


class InceptionSpecModel(nn.Module):
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
        super(InceptionSpecModel, self).__init__()
        self.conv1d_encoder = Conv1DInceptionEncoder(
            1,
            features=features,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            use_bnorm=use_bnorm,
            bnorm=bnorm,
        )
        self.preconv = nn.Conv2d(4, in_channels, 3, 1, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.SELU()
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

    def forward_features(self, x, spec_x):
        b, l, c = x.shape
        x = x.permute(0, 2, 1).contiguous().view(b * c, 1, l)
        x = self.conv1d_encoder(x)
        x = x.view(b, c, x.shape[1], x.shape[2])  # b, 16, 128, 312 -
        spec_x = self.preconv(spec_x.permute(0, 3, 2, 1))
        spec_x = self.bn(spec_x)
        spec_x = self.relu(spec_x)
        x = torch.cat([x[:, :, :, 6:-6], spec_x], dim=-1)
        x = self.conv2d(x)

        if self.old:
            x = torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)
        else:
            x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)

        # x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x, spec_x):
        x = self.forward_features(x, spec_x=spec_x)
        x = self.classifier(x)
        return x
