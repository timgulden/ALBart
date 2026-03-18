"""pygame simulated 32x32 display backend (macOS prototype)."""

import logging

import numpy as np
import pygame

from albart.runtime.display import DisplayBackend

logger = logging.getLogger(__name__)

LED_SIZE = 32


class SimDisplay(DisplayBackend):
    """
    Renders the 32x32 LED grid in a pygame window.
    Each LED pixel is drawn as a (scale x scale) block.
    Must be instantiated on the main thread (pygame requirement).
    """

    def __init__(self, scale: int = 16) -> None:
        self.scale = scale
        window_size = LED_SIZE * scale
        pygame.init()
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("ALBart — Simulated Display")
        logger.info(
            "SimDisplay initialized: %dx%d window (scale=%d)",
            window_size,
            window_size,
            scale,
        )

    def show_frame(self, rgb_array: np.ndarray) -> None:
        assert rgb_array.shape == (LED_SIZE, LED_SIZE, 3), (
            f"Expected (32, 32, 3), got {rgb_array.shape}"
        )
        # Pump events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit(0)

        # Scale up: nearest-neighbor via pygame transform
        surface = pygame.surfarray.make_surface(
            rgb_array.swapaxes(0, 1)  # pygame uses (width, height, 3)
        )
        scaled = pygame.transform.scale(
            surface, (LED_SIZE * self.scale, LED_SIZE * self.scale)
        )
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()
        logger.info("SimDisplay closed")
