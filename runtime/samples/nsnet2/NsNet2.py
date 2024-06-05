import argparse

import torch
import torch.nn as nn
from iree.turbine import aot

WIN_LEN = 0.02
HOP_FRAC = 0.5
FS = 16000
MIN_GAIN = -80

n_fft = int(WIN_LEN * FS)
frame_shift = WIN_LEN * HOP_FRAC
n_hop = n_fft * HOP_FRAC
spec_size = n_fft // 2 + 1

parser = argparse.ArgumentParser(prog='iree-turbine')
parser.add_argument('output', nargs='?')
parser.add_argument('--frames', dest='frames', metavar='N', type=int, default=1, nargs='?')
parser.add_argument('--dtype', dest='dtype', metavar='F', choices=['f32', 'f64'], default='f32')
parser.add_argument('-dump', dest='dump', action='store_true', default=False)
args = parser.parse_args()

name_to_dtype = {
    'f32': torch.float32,
    'f64': torch.float64,
}
dtype = name_to_dtype[args.dtype]


class NsNet2(nn.Module):
    def __init__(self, n_features, hidden_1, hidden_2, hidden_3):
        super().__init__()
        self.n_features = n_features
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.hidden_3 = hidden_3
        # fc1
        self.fc1 = nn.Linear(n_features, hidden_1, dtype=dtype)
        # rnn
        self.rnn1 = nn.GRU(input_size=hidden_1, hidden_size=hidden_2, num_layers=1, batch_first=True, dtype=dtype)
        self.rnn2 = nn.GRU(input_size=hidden_2, hidden_size=hidden_2, num_layers=1, batch_first=True, dtype=dtype)
        # fc2
        self.fc2 = nn.Linear(hidden_2, hidden_3, dtype=dtype)
        # fc3
        self.fc3 = nn.Linear(hidden_3, hidden_3, dtype=dtype)
        # fc4
        self.fc4 = nn.Linear(hidden_3, n_features, dtype=dtype)
        # other
        self.eps = 1e-9

    def forward(self, stft_noisy, *state_in):
        mask_pred, *state_out = self._forward(stft_noisy, *state_in)
        return mask_pred, *state_out

    def _forward(self, stft_noisy, *state_in):
        x = self.fc1(stft_noisy)
        state_out = [*state_in]
        x, state_out[0] = self.rnn1(x, state_in[0])
        x, state_out[1] = self.rnn2(x, state_in[1])
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        # sort shape
        mask_pred = x.permute(0, 2, 1).unsqueeze(1)
        return mask_pred, *state_out


model = NsNet2(n_features=spec_size, hidden_1=400, hidden_2=400, hidden_3=600)
model.train(False)


def with_frames(n_frames):
    size = 1, n_frames, model.n_features

    class CompiledNsNet2(aot.CompiledModule):
        # Make the hidden state globals that persist as long as the IREE session does.
        state1 = aot.export_global(aot.AbstractTensor(1, 1, 400, dtype=dtype), mutable=True)
        state2 = aot.export_global(aot.AbstractTensor(1, 1, 400, dtype=dtype), mutable=True)

        def main(self, x=aot.AbstractTensor(*size, dtype=dtype)):
            y, out1, out2 = aot.jittable(model.forward)(
                x, self.state1, self.state2,
                constraints=[]
            )
            self.state1 = out1
            self.state2 = out2
            return y

    return CompiledNsNet2


exported = aot.export(with_frames(n_frames=args.frames))
if args.dump:
    exported.print_readable()
else:
    exported.save_mlir(args.output)
