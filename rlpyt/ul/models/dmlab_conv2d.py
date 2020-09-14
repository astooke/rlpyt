
import torch
import torch.nn.functional as F

from rlpyt.models.utils import conv2d_output_shape


class DmlabConv2dModel(torch.nn.Module):

    # A more hard-coded version, easier to work with.

    def __init__(
            self,
            in_channels,
            use_fourth_layer=True,
            skip_connections=True,
            use_maxpool=False,
            ):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=8,
            stride=1 if use_maxpool else 4,
            padding=2 if use_maxpool else 0,
        )
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=4, stride=4) if use_maxpool else None
        self.conv2 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=1 if use_maxpool else 2,
            padding=1 if use_maxpool else 0,
        )
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2) if use_maxpool else None
        self.conv3 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if use_fourth_layer:
            self.conv4 = torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv4 = None
        # if skip_connections:
        #     if self.conv4 is not None and channels4 != c3:
        #         self.skip4 = torch.nn.Conv2d(
        #             in_channels=c3,
        #             out_channels=channels4,
        #             kernel_size=1,
        #             stride=1,
        #             padding=0,
        #         )
        #     else:
        #         self.skip4 = None
        self.skip_connections = skip_connections

    def forward(self, input):
        conv1 = F.relu(self.conv1(input))
        if self.maxpool1 is not None:
            conv1 = self.maxpool1(conv1)
        conv2 = F.relu(self.conv2(conv1))
        if self.maxpool2 is not None:
            conv2 = self.maxpool2(conv2)
        conv3_pre = self.conv3(conv2)
        if self.skip_connections:
            conv3_pre = conv3_pre + conv2
        conv3 = F.relu(conv3_pre)
        if self.conv4 is None:
            return conv3
        conv4_pre = self.conv4(conv3)
        if self.skip_connections:
            # if self.skip4 is not None:
            #     conv3_pre = self.skip4(conv3_pre)
            conv4_pre = conv4_pre + conv3_pre
        conv4 = F.relu(conv4_pre)
        return conv4

    def output_shape(self, h, w, c=None):
        """Helper function ot return the output shape for a given input shape,
        without actually performing a forward pass through the model."""
        for child in self.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return c, h, w

    def output_size(self, h, w, c=None):
        """Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        c, h, w = self.output_shape(h=h, w=w, c=c)
        return c * h * w
    