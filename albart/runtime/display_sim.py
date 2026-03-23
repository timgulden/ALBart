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

    Each LED is drawn as a circle on a black background, reflecting the
    physical panel geometry: 2mm round pixels on a 6mm pitch (1/3 fill ratio).
    dot_radius = round(scale / 6)

    Must be instantiated on the main thread (pygame requirement).
    """

    def __init__(self, scale: int = 22) -> None:
        self.scale = scale
        self.dot_radius = max(1, round(scale / 6))
        self._closed = False
        self._label = ""
        self._grid_size = LED_SIZE * scale
        pygame.init()
        pygame.font.init()
        self._font = pygame.font.SysFont("monospace", 13)
        self._text_bar = self._font.get_height() + 8  # padding above and below
        self.screen = pygame.display.set_mode((self._grid_size, self._grid_size + self._text_bar))
        pygame.display.set_caption("ALBart — Simulated Display")
        logger.info(
            "SimDisplay initialized: %dx%d window (scale=%d, dot_radius=%d)",
            self._grid_size,
            self._grid_size + self._text_bar,
            scale,
            self.dot_radius,
        )

    def show_frame(self, rgb_array: np.ndarray) -> None:
        if self._closed:
            return
        assert rgb_array.shape == (LED_SIZE, LED_SIZE, 3), (
            f"Expected (32, 32, 3), got {rgb_array.shape}"
        )
        # Pump events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit(0)

        # Black background
        self.screen.fill((0, 0, 0))

        # Draw each LED as a circle
        for row in range(LED_SIZE):
            for col in range(LED_SIZE):
                color = tuple(int(c) for c in rgb_array[row, col])
                cx = col * self.scale + self.scale // 2
                cy = row * self.scale + self.scale // 2
                pygame.draw.circle(self.screen, color, (cx, cy), self.dot_radius)

        if self._label:
            text_surf = self._font.render(self._label, True, (200, 200, 200))
            self.screen.blit(text_surf, (4, self._grid_size + 4))

        pygame.display.flip()

    def set_current_track(self, title: str, artist: str) -> None:
        self._label = f"{title} — {artist}"

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            pygame.quit()
            logger.info("SimDisplay closed")
