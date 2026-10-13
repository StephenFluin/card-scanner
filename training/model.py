"""
Small fully-convolutional CTC recognizer for the card info line.

Deliberately uses only conv / batch-norm / ReLU / max-pool so that the
exported network can be evaluated by a ~100 line pure TypeScript runtime
(src/app/vision/line-model.ts) with no ML library in the app.
Input: 1 x 32 x 176 grayscale. Output: 44 time steps x NUM_CLASSES logits.
"""
import torch
import torch.nn as nn

from common import NUM_CLASSES


def conv_bn(cin, cout, k=(3, 3), pad=(1, 1)):
    return [nn.Conv2d(cin, cout, k, padding=pad, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class InfoLineNet(nn.Module):
    def __init__(self, width: int = 1):
        super().__init__()
        c = [int(v * width) for v in (16, 32, 64, 64, 96, 128, 128)]
        self.body = nn.Sequential(
            *conv_bn(1, c[0]), nn.MaxPool2d((2, 2)),  # 16 x 88
            *conv_bn(c[0], c[1]), nn.MaxPool2d((2, 2)),  # 8 x 44
            *conv_bn(c[1], c[2]),
            *conv_bn(c[2], c[3]), nn.MaxPool2d((2, 1)),  # 4 x 44
            *conv_bn(c[3], c[4]), nn.MaxPool2d((2, 1)),  # 2 x 44
            *conv_bn(c[4], c[5], k=(2, 1), pad=(0, 0)),  # 1 x 44
            # Sequence context along the line (kernel 1x5 ~ +-2 steps = +-8 input px).
            *conv_bn(c[5], c[6], k=(1, 5), pad=(0, 2)),
            nn.Dropout2d(0.1),
            nn.Conv2d(c[6], NUM_CLASSES, (1, 1)),
        )

    def forward(self, x):
        # (N, C, 1, T) -> (T, N, C) log-probs for CTC.
        y = self.body(x).squeeze(2)
        return y.permute(2, 0, 1).log_softmax(-1)
