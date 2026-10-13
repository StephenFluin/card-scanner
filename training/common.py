"""Shared constants. Must stay in sync with src/app/vision/card-layout.ts and line-model.ts."""

# Characters the info-line model can emit. Index 0 is the CTC blank.
CHARSET = " /0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHARSET) + 1

# INFO_LINE region in normalized card coordinates (x, y, w, h).
INFO_LINE = (0.02, 0.926, 0.37, 0.05)

# Model input size (grayscale).
INPUT_H = 32
INPUT_W = 176

# Physical card aspect (63mm x 88mm).
CARD_ASPECT = 63 / 88


def encode(label: str) -> list[int]:
    return [CHARSET.index(c) + 1 for c in label]


def decode_greedy(indices) -> str:
    out = []
    prev = 0
    for i in indices:
        i = int(i)
        if i != prev and i != 0:
            out.append(CHARSET[i - 1])
        prev = i
    return "".join(out)
