from __future__ import annotations

import numpy as np


class PositionalEncoder:
    """
    NeRF-style positional encoding.

    For each input dimension x, outputs:
    [x,
     sin(2^0 x), cos(2^0 x),
     sin(2^1 x), cos(2^1 x),
     ...
     sin(2^(L-1) x), cos(2^(L-1) x)]
    """

    def __init__(self, num_freqs: int, include_input: bool = True):
        if num_freqs <= 0:
            raise ValueError("num_freqs must be positive")
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.0 ** np.arange(num_freqs, dtype=np.float32)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: input array with shape (..., input_dims)

        Returns:
            encoded: array with shape (..., encoded_dims)
        """
        if x.ndim < 1:
            raise ValueError("Input must have at least 1 dimension")

        outputs = []

        if self.include_input:
            outputs.append(x)

        for freq in self.freq_bands:
            outputs.append(np.sin(freq * x))
            outputs.append(np.cos(freq * x))

        return np.concatenate(outputs, axis=-1)

    def output_dim(self, input_dims: int) -> int:
        base = input_dims if self.include_input else 0
        return base + 2 * input_dims * self.num_freqs