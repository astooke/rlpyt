
import torch
import torch.nn.functional as F

from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.utils import conv2d_output_shape


def weight_init(m):
    """Kaiming_normal is standard for relu networks, sometimes."""
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", 
            nonlinearity="relu")
        torch.nn.init.zeros_(m.bias)


class StDimEncoderModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            latent_size,
            channels=None,
            kernel_sizes=None,
            strides=None,
            paddings=None,
            hidden_sizes=None,
            kiaming_init=True,
            ):
        super().__init__()
        c, h, w = image_shape
        self.conv = Conv2dStdimModel(
            in_channels=c,
            channels=channels or [32, 64, 64],
            kernel_sizes=kernel_sizes or [8, 4, 3],
            strides=strides or [4, 2, 1],
            paddings=paddings,
            use_maxpool=False,
        )
        self._output_size = self.conv.conv_out_size(h, w)
        self._output_shape = self.conv.conv_out_shape(h, w)
        self._conv_layer_shapes = self.conv.conv_layer_shapes(h, w)
        self.head = MlpModel(
            input_size=self._output_size,
            hidden_sizes=hidden_sizes,
            output_size=latent_size,
        )
        if kiaming_init:
            self.apply(weight_init)

    def forward(self, observation):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation
        conv, conv_layers = self.conv(img.view(T * B, *img_shape))  # lists all layers
        c = self.head(conv_layers[-1].view(T * B, -1))

        c, conv = restore_leading_dims((c, conv), lead_dim, T, B)
        conv_layers = restore_leading_dims(conv_layers, lead_dim, T, B)
        return c, conv, conv_layers  # include conv_outs for local-stdim losses

    @property
    def conv_layer_shapes(self):
        return self._conv_layer_shapes

    @property
    def conv_out_shapes(self):
        return self._conv_layer_shapes
    
    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        return self._output_shape


class StDimGlobalLocalContrastModel(torch.nn.Module):
    """The anchor gets contrasted over the batch for each location in the
    positive."""

    def __init__(self, latent_size, local_size, anchor_hidden_sizes):
        super().__init__()
        self.anchor_mlp = MlpModel(
            input_size=latent_size,
            hidden_sizes=anchor_hidden_sizes,
            output_size=latent_size,
        )
        self.W = torch.nn.Linear(latent_size, local_size, bias=False)

    def forward(self, anchor, positive):
        # anchor shape is [B,Z], positive shape is [B,C,H,W]
        anchor = anchor + self.anchor_mlp(anchor)
        b, c, h, w = positive.shape

        # Vectorized form, way OOM:
        # positive = positive.permute(0, 2, 3, 1).reshape(b * h * w, c)
        # Wz = torch.matmul(self.W, positive.T)
        # logits = torch.matmul(anchor.repeat(h * w, 1), Wz)
        # logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        # logits = logits.view(b, h * w, -1)
        # logits_list = [logits[:, i] for i in range(h * w)]

        # Non-vectorized form -- fits in memory, more readable:
        logits_list = list()
        pred = self.W(anchor)
        for y in range(h):
            for x in range(w):
                logits = torch.matmul(pred, positive[:, :, y, x].T)
                logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
                logits_list.append(logits)
        return logits_list


class StDimLocalLocalContrastModel(torch.nn.Module):

    def __init__(self, local_size, anchor_hidden_sizes):
        super().__init__()
        self.anchor_mlp = MlpModel(
            input_size=local_size,
            hidden_sizes=anchor_hidden_sizes,
            output_size=local_size,
        )
        self.W = torch.nn.Linear(local_size, local_size, bias=False)

    def forward(self, anchor, positive):
        # Every location in the anchor gets contrasted over the batch of
        # positives at the same location.
        # anchor and positive both [B,C,H,W]
        b, c, h, w = anchor.shape

        # Vectorized form, way OOM:
        # anchor = anchor.permute(0, 2, 3, 1).reshape(b * h * w, c)
        # positive = positive.permute(0, 2, 3, 1).reshape(b * h * w, c)
        # anchor = anchor + self.anchor_mlp(anchor)
        # Wz = torch.matmul(self.W, positive.T)
        # logits = torch.matmul(anchor, Wz)
        # logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        # logits = logits.view(b, h * w, -1)
        # logits_list = [logits[:, i] for i in range(h * w)]

        # Non-vectorized form -- fits in memory, more readable:
        logits_list = list()
        for y in range(h):
            for x in range(w):
                anchor_xy = anchor[:, :, y, x]
                anchor_xy = anchor_xy + self.anchor_mlp(anchor_xy)
                pred_xy = self.W(anchor_xy)
                logits = torch.matmul(pred_xy, positive[:, :, y, x].T)
                logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
                logits_list.append(logits)
        return logits_list


class Conv2dStdimModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,
            use_maxpool=False,
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        maxp_layers = list()
        for conv_layer, maxp_stride in zip(conv_layers, maxp_strides):
            sequence.append(conv_layer)
            sequence.append(nonlinearity())
            if maxp_stride > 1:
                maxp_layer = torch.nn.MaxPool2d(maxp_stride)
                sequence.append(maxp_layer)  # No padding.
                maxp_layers.append(maxp_layer)
            else:
                maxp_layers.append(None)
        # Register parameters this way, so can still use the same state_dict
        # format for loading in the RL agent:
        self.conv = torch.nn.Sequential(*sequence)
        # But will actually use these lists for the forward pass
        # (they shouldn't register as parameters, would be duplicates,
        # hopefully this still works.)
        self.conv_layers = conv_layers  # without registering
        self.maxp_layers = maxp_layers

    def forward(self, input):
        conv_outs = list()
        x = input
        for conv, maxp in zip(self.conv_layers, self.maxp_layers):
            x = F.relu(conv(x))
            conv_outs.append(x)
            if maxp is not None:
                x = maxp(x)
        return x, conv_outs  # Return the post-ReLU conv at every layer.

    def conv_out_size(self, h, w, c=None):
        """Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return h * w * c

    def conv_layer_shapes(self, h, w, c=None):
        """Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        shapes = list()
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
                shapes.append((c, h, w))
            except AttributeError:
                pass  # Not a conv layer.
        return shapes

    def conv_out_shape(self, h, w, c=None):
        shapes = self.conv_layer_shapes(h=h, w=w, c=c)
        return shapes[-1]

    def conv_out_shapes(self, *args, **kwargs):
        return self.conv_layer_shapes(*args, **kwargs)
