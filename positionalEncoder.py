import torch

from const import DEVICE, MAIN


class PositionalEncoder:
    def __init__(self, posisionalEncodingSize: int, powerConstant=1e4) -> None:
        self.positionalEncodingSize = posisionalEncodingSize
        self.powerConstant = powerConstant

    def encode(self, k) -> torch.Tensor:
        out = torch.arange(0, self.positionalEncodingSize)

        out = (out % 2 == 1)*torch.sin(k/self.powerConstant**(out/self.positionalEncodingSize)) + \
            (out % 2 == 0)*torch.cos(k/self.powerConstant **
                                     ((out-1)/self.positionalEncodingSize))

        return out.to(DEVICE)


if __name__ == MAIN:
    posEncoder = PositionalEncoder(4)
    print(posEncoder.encode(2))
