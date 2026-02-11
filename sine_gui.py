import pygame
import math
import time
import sys
import threading
from color_theme import ColorTheme

class SineGUI:
    def __init__(self, width=800, height=480, fps=30):
        pygame.init()
        self.width = int(width)
        self.height = int(height)
        self.fullscreen = True
        self.set_display_mode()
        pygame.display.set_caption("Sine Wave Visualizer")
        self.font = pygame.font.Font("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.double_tap_ms = 300      # max gap between taps
        self._last_tap_ts = 0

        self.on_start_trial = None  # callback
        self.is_idle = True
        self.running = False
        self.duration_minutes = 15
        self.duration_text = "15"
        self.duration_focused = False
        self.thread = None
        self.start_time = time.time()

        self.theme = ColorTheme()

        self.target_coherence = 0.0      # externally controlled
        self.nvar = 0.0
        self.significant_nvars = []
        self.significant_nvar_tally = []
        self.raw_coherence = 0.0
        self.z_global = ""
        self.smoothed_coherence = 0.0    # for smoothing
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

    def set_coherence(self, coherence, nvar=False, raw_coherence=False):
        self.target_coherence = max(0.0, min(1.0, coherence))  # Clamp safely
        if nvar: self.nvar = int(nvar)
        if raw_coherence: self.raw_coherence = raw_coherence

    def set_z_global(self, z = False):
        if z: self.z_global = z

    def trigger_flash(self, flash_color = False):
        self.flash_active = True
        self.flash_start_time = time.time()
        if flash_color:
            self.flash_color = flash_color
        else:
            self.flash_color = self.theme.color("black")

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
        # Det default colors
        col = self.theme.color("highlight")
        if coherence > 0.5:
            col = self.theme.color("heaven")
        elif coherence > 0.4:
            col = self.theme.color("green")
        elif coherence > 0.3:
            col = self.theme.color("yellow")
        elif coherence > 0.2:
            col = self.theme.color("lightamber")
        bg = self.theme.color("background")
        # Overrides for flashing alert
        if self.flash_active:
            now = time.time()
            elapsed = now - self.flash_start_time
            if elapsed < self.flash_duration:
                # Flash white every 0.25 seconds
                if int(elapsed / self.flash_interval) % 2 == 0:
                    bg = self.flash_color
                    col = self.theme.color("black")
                else:
                    bg = self.theme.color("white")
            else:
                self.flash_active = False

        self.screen.fill(bg)  # black background
        step = 4
        prev_x = 0
        prev_y = self.calculate_y(0, t, coherence)

        for x in range(step, self.width, step):
            y = self.calculate_y(x, t, coherence)
            pygame.draw.line(self.screen, col, (prev_x, prev_y), (x, y), 4)
            prev_x, prev_y = x, y

    def draw_text(self, text, text_color=(255, 255, 255), x=0, y=0, align="topleft"):
        surface = self.font.render(str(text), True, text_color)
        if not surface:
            print("[ERROR] Font render failed")
            return
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

    def draw_tally_circles(self, colors, radius=10, spacing=10, y=20):
        total = len(colors)
        if total == 0:
            return

        # Calculate total width of all circles + spacing
        total_width = total * (radius * 2) + (total - 1) * spacing
        start_x = (self.width - total_width) // 2 + radius

        for i, color in enumerate(colors):
            x = start_x + i * (2 * radius + spacing)
            pygame.draw.circle(self.screen, color, (x, y), radius)

    def draw_duration_input(self, y_center):
        border_color = self.theme.color("white") if self.duration_focused else self.theme.color("gray")
        box_w, box_h = 120, 40
        box_rect = pygame.Rect(0, 0, box_w, box_h)
        box_rect.center = (self.width // 2 - 30, y_center)
        pygame.draw.rect(self.screen, self.theme.color("background"), box_rect)
        pygame.draw.rect(self.screen, border_color, box_rect, 2)
        # Render duration text inside the box
        text_surface = self.font.render(self.duration_text, True, self.theme.color("white"))
        text_rect = text_surface.get_rect(center=box_rect.center)
        self.screen.blit(text_surface, text_rect)
        # "min" label to the right
        label = self.font.render("min", True, self.theme.color("gray"))
        label_rect = label.get_rect(midleft=(box_rect.right + 10, y_center))
        self.screen.blit(label, label_rect)
        return box_rect

    def draw_idle_screen(self):
        summary_text = self.font.render(self.z_global, True, self.theme.color("gray"))
        summary_rect = summary_text.get_rect(center=(self.width // 2, self.height // 2 - 160))
        self.screen.blit(summary_text, summary_rect)

        # Duration input between summary and New Trial button
        duration_rect = self.draw_duration_input(self.height // 2 - 90)

        text = self.font.render("New Trial", True, self.theme.color("white"))
        rect = text.get_rect(center=(self.width // 2, self.height // 2 - 20))
        pygame.draw.rect(self.screen, self.theme.color("darkgray"), rect.inflate(80, 60))
        self.screen.blit(text, rect)

        # Draw Quit button below it
        quit_text = self.font.render("Quit", True, self.theme.color("white"))
        quit_rect = quit_text.get_rect(center=(self.width // 2, self.height // 2 + 100))
        pygame.draw.rect(self.screen, self.theme.color("darkgray"), quit_rect.inflate(80, 60))
        self.screen.blit(quit_text, quit_rect)

        return rect, quit_rect, summary_rect, duration_rect

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

    def run(self):
        self.running = True
        while self.running:

            if self.is_idle:
                self.screen.fill(self.theme.color("background"))
                btn_rect, quit_rect, summary_rect, duration_rect = self.draw_idle_screen()
                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if duration_rect.collidepoint(event.pos):
                            self.duration_focused = True
                        elif btn_rect.collidepoint(event.pos) and self.on_start_trial:
                            self.duration_focused = False
                            self.is_idle = False
                            self.start_time = time.time()
                            self.significant_nvars.clear()
                            self.significant_nvar_tally.clear()
                            self.on_start_trial()
                        elif quit_rect.collidepoint(event.pos):
                            self.running = False
                        else:
                            self.duration_focused = False
                    elif event.type == pygame.KEYDOWN and self.duration_focused:
                        if event.key == pygame.K_BACKSPACE:
                            self.duration_text = self.duration_text[:-1]
                        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            self.duration_focused = False
                        elif event.unicode.isdigit() and len(self.duration_text) < 3:
                            self.duration_text += event.unicode
                        # Update duration_minutes from text
                        if self.duration_text:
                            self.duration_minutes = int(self.duration_text)
                self.clock.tick(self.fps)
                continue  # Skip drawing rest of the frame

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if self.fullscreen:
                        self.fullscreen = False
                        self.set_display_mode()
                    else:
                        self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # allow double‑tap only while the trial is running
                    if self.fullscreen:
                        now = pygame.time.get_ticks()
                        if now - self._last_tap_ts <= self.double_tap_ms:
                            # double‑tap detected
                            self.fullscreen = False
                            self.set_display_mode()
                            # reset so triple taps don't chain multiple toggles
                            self._last_tap_ts = 0
                        else:
                            self._last_tap_ts = now

            t = time.time() - self.start_time

            # Smooth the coherence transition
            alpha = 0.05
            self.smoothed_coherence = (
                (1 - alpha) * self.smoothed_coherence + alpha * self.target_coherence
            )

            # Store current color for nvars
            if self.nvar > 125:
                nv_color = self.theme.get_color_by_nvar(self.nvar)
            else:
                nv_color = self.theme.color("gray")

            # Check for significant nvars
            if (
                self.theme.get_label_by_nvar(self.nvar) != "Normal" and 
                self.significant_nvars and
                self.nvar != self.significant_nvars[-1]
            ):
                self.significant_nvars.append(self.nvar)
                self.significant_nvar_tally.append(nv_color)
                self.trigger_flash(nv_color)

            # Draw stuff to screen
            self.draw_wave(t * 2 * math.pi, self.smoothed_coherence)
            self.draw_text(f"GCP: {self.nvar}", nv_color, 10, 10, "topleft")
            self.draw_text(f"{self.theme.get_label_by_nvar(self.nvar)}", nv_color, self.width - 10, 10, "topright")
            self.draw_text(f"Cor: {self.raw_coherence:.2f}", self.theme.color("gray"), self.width - 10, self.height - 10, align="bottomright")
            # self.draw_text(f"{self.z_global}", self.theme.color("gray"), 10, self.height - 10, align="bottomleft")
            self.draw_tally_circles(self.significant_nvar_tally)
            pygame.display.flip()
            self.clock.tick(self.fps)

        # pygame.quit()
        # sys.exit()
        self._close_display()
        return

    def start(self):
        """Launch the GUI loop in a background thread"""
        if not self.thread:
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        # If we launched a thread, wait for it to finish its cleanup
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        # In case run() wasn’t threaded or join timed out, ensure display is closed
        self._close_display()
        self.thread = None

    def reset(self):
        self.start_time = time.time()
        self.nvar = 0
        self.raw_coherence = 0.0
        self.smoothed_coherence = 0.0
        self.z_global = ""
        self.significant_nvars = []
        self.significant_nvar_tally = []
        self.is_idle = False
