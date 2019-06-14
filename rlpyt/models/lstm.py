
import torch


class LstmModel(torch.nn.Lstm):

    def forward(self, input, init_rnn_state):
        if init_rnn_state is not None:  # [B,N,H] --> [N,B,H]
            init_rnn_state = tuple(hc.transpose(0, 1).contiguous()
                for hc in init_rnn_state)
        lstm_out, (hn, cn) = super().__call__(input, init_rnn_state)
        hn, cn = hn.transpose(0, 1), cn.transpose(0, 1)  # --> [B,N,H]
        return lstm_out, (hn, cn)
