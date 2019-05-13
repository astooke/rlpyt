
import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims


class AtariLstmModel(torch.nn.Module):

    def __init__(
            self,
            env_spec,
            # conv_channels,
            # conv_sizes,
            # conv_strides,
            # conv_pads,
            # pool_sizes,
            # hidden_size=256,
            lstm_size=256,
            lstm_layers=1,
            # name="atari_cnn_lstm",
            ):
        super().__init__()
        image_shape = env_spec.observation_space.shape
        action_size = env_spec.action_space.n
        self.env_spec = env_spec

        # Hard-code just to get it running.
        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=8,
            stride=1,
            padding=0,
        )
        self.maxp1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=1,
            padding=0,
        )
        self.maxp2 = torch.nn.MaxPool2d(2)

        test_mat = torch.zeros(1, *image_shape)
        test_mat = self.conv1(test_mat)
        test_mat = self.maxp1(test_mat)
        test_mat = self.conv2(test_mat)
        test_mat = self.maxp2(test_mat)
        lstm_in_size = test_mat.numel() + action_size + 1

        self.lstm = torch.nn.LSTM(lstm_in_size, lstm_size, lstm_layers)
        self.linear_pi = torch.nn.Linear(lstm_size, action_size)
        self.linear_v = torch.nn.Linear(lstm_size, 1)

        # in_channels = image_shape[0]
        # self.conv_layers = list()
        # self.conv_pool_layers = list()
        # for i in range(len(conv_channels)):
        #     conv_layer = torch.nn.Conv2d(
        #         in_channels=in_channels,
        #         out_channels=conv_channels[i],
        #         kernel_size=conv_sizes[i],
        #         stride=conv_strides[i],
        #         padding=conv_pads[i],
        #     )
        #     self.conv_layers.append(conv_layer)
        #     if pool_sizes[i] > 1:
        #         pool_layer = torch.nn.MaxPool2d(pool_sizes[i])
        #         self.conv_pool_layers.append(pool_layer)
        #     in_channels = conv_channels[i]

        # test_mat = torch.zeros(1, **image_shape)
        # for conv_pool_layer in self.conv_pool_layers:
        #     test_mat = conv_pool_layer(test_mat)
        # self.conv_out_size = int(np.prod(test_mat.shape))

        # if hidden_size > 0:
        #     self.hidden_layer = torch.nn.Linear(self.conv_out_size, hidden_size)
        #     lstm_input_size = hidden_size
        # else:
        #     self.hidden_layer = None
        #     lstm_input_size = self.conv_out_size
        # lstm_input_size += sum([s.size for s in env_spec.action_spaces]) + 1

        # self.lstm_layer = torch.nn.LSTM(
        #     input_size=lstm_input_size,
        #     hidden_size=lstm_size,
        #     num_layers=lstm_layers,
        # )

    def forward(self, image, prev_action, prev_reward, init_rnn_state):
        img = image.to(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer leading dimensions.
        T, B, img_shape, _T, _B = infer_leading_dims(img, 3)

        img = img.view(T * B, *img_shape)  # Fold time and batch dimensions.
        img = F.relu(self.maxp1(self.conv1(img)))
        img = F.relu(self.maxp2(self.conv2(img)))

        onehot_action = self.env_spec.action_space.to_onehot(prev_action)
        lstm_input = torch.cat([
            img.view(T, B, -1),
            onehot_action.view(T, B, -1),
            prev_reward.view(T, B, 1),
            ], dim=2)
        lstm_out, next_rnn_state = self.lstm(lstm_input, init_rnn_state)
        lstm_flat = lstm_out.view(T * B, -1)
        pi = F.softmax(self.linear_pi(lstm_flat), dim=-1)
        v = self.linear_v(lstm_flat)

        # Replace leading dimensions.
        if _T:
            pi = pi.view(T, B, -1)
            v = v.view(T, B)
        elif _B:
            pi = pi.view(B, -1)
            v = v.view(B)
        else:
            pi = pi.view(-1)
            v = v.squeeze()
            next_rnn_state = next_rnn_state.squeeze(1)  # Remove batch dim.

        return pi, v, next_rnn_state
