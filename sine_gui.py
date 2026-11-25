import math
import os
import time

import pygame

from color_theme import ColorTheme

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def load_font(size):
    if os.path.exists(FONT_PATH):
        return pygame.font.Font(FONT_PATH, size)
    return pygame.font.Font(None, int(size * 1.35))  # pygame's default font runs small


class SineGUI:
    """Fullscreen display for sessions.

    The GUI only draws and collects taps. A controller (see app.py) owns the
    session and sets the fields below each frame through tick():

        mode         "idle" | "starting" | "session" | "finishing"
        cue          large text top-left (e.g. "AIM HIGH"), "" for none
        cue_dim      draw the cue in gray (between blocks / sham)
        progress     text top-right (e.g. "Block 3/24")
        feedback     text bottom-right (e.g. "z +1.20")
        coherence    0..1, drives the wave
        summary      text on the idle screen
    """

    def __init__(self, width=800, height=480, fps=30, fullscreen=True):
        pygame.init()
        self.width = int(width)
        self.height = int(height)
        self.fullscreen = fullscreen
        self.set_display_mode()
        pygame.display.set_caption("DualRng")
        self.font = load_font(24)
        self.big_font = load_font(44)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.double_tap_ms = 300      # max gap between taps
        self._last_tap_ts = 0
        self._closed = False          # set once the display has been shut down
        self.theme = ColorTheme()

        # Controller hooks
        self.controller = None

        # Idle-screen selections
        self.session_type = "intention"
        self.sham = False
        self.plan_text = ""
        self.summary = ""

        # Session display state (set by the controller)
        self.mode = "idle"
        self.cue = ""
        self.cue_dim = False
        self.progress = ""
        self.feedback = ""
        self.status = ""
        self.coherence = 0.0
        self.smoothed_coherence = 0.0
        self.tally = []
        self._stop_armed_until = 0.0

        self.running = False
        self.start_time = time.time()
        self.flash_active = False
        self.flash_start_time = 0
        self.flash_duration = 3      # total duration in seconds
        self.flash_interval = 0.25   # flash every 0.25s
        self.flash_color = self.theme.color("white")

        # Precomputed interfering sine waves
        base_freq = 0.02
        self.extra_waves = []
        for i in range(5):
            freq = base_freq + (i - 2) * 0.002
            phase_scale = 0.8 + i * 0.1
            amp_scale = 0.2 + 0.05 * i
            self.extra_waves.append((freq, phase_scale, amp_scale))

    # ------------------------------------------------------------------
    def _close_display(self):
        if self._closed:
            return
        try:
            # Let any pending Wayland frame callbacks settle
            pygame.event.pump()
            # Quit the display first (destroys wl_surface)
            pygame.display.quit()
        finally:
            # Then shut down the rest of pygame
            pygame.quit()
            self._closed = True

    def set_display_mode(self):
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            pygame.mouse.set_visible(False)
            info = pygame.display.Info()
            self.width, self.height = info.current_w, info.current_h
        else:
            self.screen = pygame.display.set_mode((800, 600))
            pygame.mouse.set_visible(True)
            self.width, self.height = 800, 600

    def trigger_flash(self, flash_color=None):
        self.flash_active = True
        self.flash_start_time = time.time()
        self.flash_color = flash_color or self.theme.color("black")

    def add_tally(self, color):
        self.tally.append(color)
        self.trigger_flash(color)

    def reset(self):
        self.start_time = time.time()
        self.coherence = 0.0
        self.smoothed_coherence = 0.0
        self.cue = ""
        self.progress = ""
        self.feedback = ""
        self.status = ""
        self.tally = []
        self.flash_active = False

    # ------------------------------------------------------------------
    # Drawing (wave and colours unchanged from the original visualizer)
    def calculate_y(self, x, t, coherence):
        base_amp = (1.0 - coherence) * (self.height // 4)
        base_freq = 0.02
        y = math.sin(base_freq * x + t)

        count = int(coherence * len(self.extra_waves))
        for i in range(count):
            freq, phase_scale, amp_scale = self.extra_waves[i]
            y += math.sin(freq * x + t * phase_scale) * amp_scale

        y /= (1 + count)
        return int(self.height / 2 - y * base_amp)

    def draw_wave(self, t, coherence):
        col = self.theme.color("highlight")
        if coherence > 0.5:
            col = self.theme.color("heaven")
        elif coherence > 0.4:
            col = self.theme.color("green")
        elif coherence > 0.3:
            col = self.theme.color("yellow")
        elif coherence > 0.2:
            col = self.theme.color("lightamber")
        if self.cue_dim:
            col = self.theme.color("darkgray")
        bg = self.theme.color("background")
        if self.flash_active:
            elapsed = time.time() - self.flash_start_time
            if elapsed < self.flash_duration:
                if int(elapsed / self.flash_interval) % 2 == 0:
                    bg = self.flash_color
                    col = self.theme.color("black")
                else:
                    bg = self.theme.color("white")
            else:
                self.flash_active = False

        self.screen.fill(bg)
        step = 4
        prev_x = 0
        prev_y = self.calculate_y(0, t, coherence)
        for x in range(step, self.width, step):
            y = self.calculate_y(x, t, coherence)
            pygame.draw.line(self.screen, col, (prev_x, prev_y), (x, y), 4)
            prev_x, prev_y = x, y

    def draw_text(self, text, text_color=(255, 255, 255), x=0, y=0, align="topleft", font=None):
        surface = (font or self.font).render(str(text), True, text_color)
        rect = surface.get_rect()
        if align == "topleft":
            rect.topleft = (x, y)
        elif align == "topright":
            rect.topright = (x, y)
        elif align == "bottomleft":
            rect.bottomleft = (x, y)
        elif align == "bottomright":
            rect.bottomright = (x, y)
        elif align == "center":
            rect.center = (x, y)
        else:
            raise ValueError(f"Unsupported alignment: {align}")
        self.screen.blit(surface, rect)
        return rect

    def draw_tally_circles(self, colors, radius=10, spacing=10, y=20):
        total = len(colors)
        if total == 0:
            return
        total_width = total * (radius * 2) + (total - 1) * spacing
        start_x = (self.width - total_width) // 2 + radius
        for i, color in enumerate(colors):
            x = start_x + i * (2 * radius + spacing)
            pygame.draw.circle(self.screen, color, (x, y), radius)

    def draw_button(self, label, center, selected=False):
        text = self.font.render(label, True, self.theme.color("white"))
        rect = text.get_rect(center=center)
        box = rect.inflate(60, 36)
        pygame.draw.rect(self.screen, self.theme.color("darkgray"), box)
        if selected:
            pygame.draw.rect(self.screen, self.theme.color("highlight"), box, 3)
        self.screen.blit(text, rect)
        return box

    def draw_idle_screen(self):
        cx, cy = self.width // 2, self.height // 2
        gray = self.theme.color("gray")
        if self.summary:
            for i, line in enumerate(self.summary.split("\n")[:2]):
                self.draw_text(line, gray, cx, cy - 190 + 28 * i, align="center")
        buttons = {
            "intention": self.draw_button("Intention", (cx - 230, cy - 100), self.session_type == "intention"),
            "connection": self.draw_button("Connection", (cx, cy - 100), self.session_type == "connection"),
            "sham": self.draw_button("Sham: on" if self.sham else "Sham: off", (cx + 230, cy - 100), self.sham),
        }
        self.draw_text(self.plan_text, gray, cx, cy - 45, align="center")
        buttons["start"] = self.draw_button("Start Session", (cx, cy + 20))
        buttons["quit"] = self.draw_button("Quit", (cx, cy + 120))
        return buttons

    def draw_message(self, text):
        self.screen.fill(self.theme.color("background"))
        self.draw_text(text, self.theme.color("gray"), self.width // 2, self.height // 2, align="center")

    def draw_session(self):
        t = time.time() - self.start_time
        alpha = 0.05
        self.smoothed_coherence = (1 - alpha) * self.smoothed_coherence + alpha * self.coherence
        self.draw_wave(t * 2 * math.pi, self.smoothed_coherence)
        cue_color = self.theme.color("gray") if self.cue_dim else self.theme.color("highlight")
        if self.cue:
            self.draw_text(self.cue, cue_color, 14, 10, "topleft", font=self.big_font)
        self.draw_text(self.progress, self.theme.color("gray"), self.width - 10, 10, "topright")
        if self.feedback:
            self.draw_text(self.feedback, self.theme.color("gray"), self.width - 10, self.height - 10, "bottomright")
        if self.status:
            self.draw_text(self.status, self.theme.color("lightamber"), self.width // 2, self.height - 10,
                           "center")
        armed = time.time() < self._stop_armed_until
        stop_label = "Tap again to stop" if armed else "Stop"
        stop_rect = self.draw_text(stop_label, self.theme.color("lightamber" if armed else "gray"),
                                   10, self.height - 10, "bottomleft")
        self.draw_tally_circles(self.tally, y=self.height - 30)
        return stop_rect.inflate(30, 30)

    # ------------------------------------------------------------------
    def _handle_common(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.fullscreen:
                self.fullscreen = False
                self.set_display_mode()
            else:
                self.running = False
            return True
        return False

    def _double_tap(self):
        if not self.fullscreen:
            return False
        now = pygame.time.get_ticks()
        if now - self._last_tap_ts <= self.double_tap_ms:
            self.fullscreen = False
            self.set_display_mode()
            self._last_tap_ts = 0
            return True
        self._last_tap_ts = now
        return False

    def run(self):
        self.running = True
        try:
            while self.running:
                if self.controller is not None:
                    self.controller.tick()
                if self.mode == "idle":
                    self.screen.fill(self.theme.color("background"))
                    buttons = self.draw_idle_screen()
                    for event in pygame.event.get():
                        if self._handle_common(event):
                            continue
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if buttons["intention"].collidepoint(event.pos):
                                self.session_type = "intention"
                            elif buttons["connection"].collidepoint(event.pos):
                                self.session_type = "connection"
                            elif buttons["sham"].collidepoint(event.pos):
                                self.sham = not self.sham
                            elif buttons["start"].collidepoint(event.pos) and self.controller:
                                self.reset()
                                self.controller.start(self.session_type, self.sham)
                            elif buttons["quit"].collidepoint(event.pos):
                                self.running = False
                            if self.controller:
                                self.controller.selection_changed(self.session_type, self.sham)
                elif self.mode == "session":
                    stop_rect = self.draw_session()
                    for event in pygame.event.get():
                        if self._handle_common(event):
                            continue
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if stop_rect.collidepoint(event.pos):
                                if time.time() < self._stop_armed_until:
                                    self._stop_armed_until = 0
                                    if self.controller:
                                        self.controller.stop()
                                else:
                                    self._stop_armed_until = time.time() + 3
                            else:
                                self._double_tap()
                else:
                    self.draw_message("Starting recorder..." if self.mode == "starting" else "Saving and analyzing...")
                    for event in pygame.event.get():
                        self._handle_common(event)
                pygame.display.flip()
                self.clock.tick(self.fps)
        finally:
            if self.controller is not None:
                self.controller.shutdown()
            self._close_display()

    def stop(self):
        self.running = False
