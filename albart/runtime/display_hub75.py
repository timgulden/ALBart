"""HUB75 32x32 LED matrix display backend (Raspberry Pi production)."""

import logging

import numpy as np

from albart.runtime.display import DisplayBackend

logger = logging.getLogger(__name__)


class Hub75Display(DisplayBackend):
    """
    Drives a HUB75 32x32 LED matrix via rpi-rgb-led-matrix.
    Only usable on Raspberry Pi with the library installed.
    """

    def __init__(self) -> None:
        try:
            from rgbmatrix import RGBMatrix, RGBMatrixOptions
        except ImportError as e:
            raise RuntimeError(
                "rpi-rgb-led-matrix is not installed. "
                "Hub75Display is only available on Raspberry Pi."
            ) from e

        options = RGBMatrixOptions()
        options.rows = 32
        options.cols = 32
        options.chain_length = 1
        options.parallel = 1
        options.hardware_mapping = "regular"

        self.matrix = RGBMatrix(options=options)
        self.canvas = self.matrix.CreateFrameCanvas()
        logger.info("Hub75Display initialized (32x32 HUB75 panel)")

    def show_frame(self, rgb_array: np.ndarray) -> None:
        assert rgb_array.shape == (32, 32, 3), (
            f"Expected (32, 32, 3), got {rgb_array.shape}"
        )
        for y in range(32):
            for x in range(32):
                r, g, b = int(rgb_array[y, x, 0]), int(rgb_array[y, x, 1]), int(rgb_array[y, x, 2])
                self.canvas.SetPixel(x, y, r, g, b)
        self.canvas = self.matrix.SwapOnVSync(self.canvas)

    def close(self) -> None:
        self.matrix.Clear()
        logger.info("Hub75Display cleared and closed")
