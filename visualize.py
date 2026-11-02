"""Fullscreen, side-by-side (2 columns x 3 rows) replay of all six agents.

Design notes
------------
* Fullscreen by default; every dimension is derived from the real display size,
  so it also works windowed (``--windowed``) or headless (``--headless``).
* Each panel is one algorithm: its board on the left, and under the model name
  a clean report of the stored evaluation metrics plus an explicit convergence
  verdict (``rl_common.convergence``). A model is never called converged
  without meeting that criterion; missing data means "not evaluated".
* Every panel owns its own ``GridEnv``, episode counters and restart state, so
  one agent finishing its episode never pauses or resets the others.
* Stepping is strictly uniform: each frame advances *every* loaded panel by
  exactly one environment step. There is no per-panel freeze; an episode that
  ends is restarted on the next frame, and the terminal state is reported by a
  result chip and a short border flash.
* Only checkpoints are loaded here -- nothing is trained. Missing or
  unloadable checkpoints render a placeholder instead of aborting the window.
* A small ``Inferences`` button in the top bar (key ``I``) opens a modal that
  explains, in one short line per model, why the ranking looks the way it
  does -- the plain-language version of ``INFERENCES.md``.

Keys: Space pause, Up/Down speed, R restart every panel, F fullscreen,
Inferences button / I open the breakdown, Esc closes it (then quits).
Headless: ``python visualize.py --headless --frames 40 --save-frames frames``.
"""

import argparse
import glob
import os
import sys
import warnings

import pygame

from rl_common import (
    ALGORITHM_SPECS,
    PANEL_ORDER,
    SEED,
    CheckpointError,
    algorithm_rankings,
    comparison_metrics,
    convergence,
    load_policy,
    make_env,
)

# Medal colours for the rank badge (ranks are unique, so one gold/silver/bronze).
RANK_COLORS = {
    1: (255, 200, 60),   # gold
    2: (190, 200, 215),  # silver
    3: (210, 140, 80),   # bronze
}
RANK_COLOR_DEFAULT = (120, 170, 235)

# --- Grid / panel constants ----------------------------------------------
GRID_COLS = 7
GRID_ROWS = 8
COLUMNS = 2
ROWS = 3
RESULT_FLASH_FRAMES = 8  # cosmetic border flash, does not stop stepping

# --- Palette --------------------------------------------------------------
COLOR_BG = (16, 17, 20)
COLOR_CARD = (24, 25, 30)
COLOR_CARD_EDGE = (58, 60, 70)
COLOR_BG_HEADER = (30, 30, 36)
COLOR_TEXT = (240, 240, 240)
COLOR_TEXT_DIM = (158, 158, 170)
COLOR_GOAL_ROW = (46, 125, 50)       # Lush Green
COLOR_SAFE_ROW = (56, 142, 60)       # Dark Green
COLOR_START_ROW = (74, 90, 102)      # Pavement Gray
COLOR_ROAD = (40, 42, 48)            # Dark Asphalt
COLOR_LANE_DASH = (220, 220, 220)

CATEGORY_COLORS = {
    "policy-based": (96, 165, 250),
    "value-based": (251, 191, 36),
    "hybrid": (167, 139, 250),
}

GOAL_COLOR = (46, 204, 113)
CRASH_COLOR = (231, 76, 60)
TIMEOUT_COLOR = (230, 176, 80)
RUNNING_COLOR = (120, 170, 235)
MISSING_COLOR = (200, 120, 120)

RESULT_COLORS = {
    "GOAL": GOAL_COLOR,
    "CRASH": CRASH_COLOR,
    "TIME LIMIT": TIMEOUT_COLOR,
}

# Car colors for rows 1 to 5
CAR_COLORS = [
    (231, 76, 60),   # Red
    (230, 126, 34),  # Orange
    (52, 152, 219),  # Blue
    (155, 89, 182),  # Purple
    (241, 196, 15),  # Yellow
]

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}

# --- "Inferences" overlay: why each model did better ----------------------
# Plain-language version of INFERENCES.md, one short line per algorithm.
# All six models reach the goal 100%; reward = 135 - 0.1 * steps, so the only
# real difference is how many steps each policy wastes on its route.
INFERENCES_TITLE = "Why did each model do better?"

INFERENCES_INTRO = (
    "All six models reach the goal every time (100% success).",
    "Reward = 135 - 0.1 x steps, so the winner is simply whoever takes "
    "the fewest steps.",
)

# One plain sentence per algorithm (shown best rank first).
INFERENCES_WHY = {
    "ppo": "Shortest path (8 steps), found first at 15k steps - and it kept it.",
    "a3c": "Same 8-step path, but it forgot it and re-learned it; the saved model is from 35k.",
    "a2c": "Same 8-step path, but it found it only once, at 50k steps, then forgot it.",
    "dqn": "Always reaches the goal, just 4 steps longer (12 steps) - a safer route.",
    "ddqn": "Same 12-step route as DQN, but it needed 90k steps to lock it in.",
    "reinforce": "Noisiest learning of the six, so it settles for the longest route (15 steps).",
}

INFERENCES_TAKEAWAY = (
    "In short: policy gradient (PPO, A3C, A2C) found the shortest route, "
    "value-based (DQN, Double DQN) plays it safer and longer, and REINFORCE "
    "gets there but never sharpens the route.",
    "Ties are broken by the earliest good checkpoint: "
    "PPO 15k vs A3C 35k vs A2C 50k steps.",
)


class _Font:
    """Font wrapper with a ``pygame.font.Font``-style API.

    Some source builds of pygame are compiled without SDL_ttf, which makes
    ``pygame.font`` unusable (its fallback implementation fails with a circular
    import). In that case we render text through pygame's bundled FreeType
    module instead, so the visualizer works with either build.
    """

    _FONT_DIRS = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
    ]

    def __init__(self, size, bold=False):
        self._freetype = False

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._font = pygame.font.SysFont("dejavusans", size, bold=bold)
            return
        except Exception:
            pass

        from pygame import _freetype

        if not _freetype.get_init():
            _freetype.init()
        self._font = _freetype.Font(self._find_font_file(bold), size)
        self._freetype = True

    @classmethod
    def _find_font_file(cls, bold):
        filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        for directory in cls._FONT_DIRS:
            matches = glob.glob(
                os.path.join(directory, "**", filename), recursive=True
            )
            if matches:
                return matches[0]

        # pygame ships its own fallback font
        bundled = os.path.join(os.path.dirname(pygame.__file__), "freesansbold.ttf")
        if os.path.exists(bundled):
            return bundled
        raise FileNotFoundError("No usable TTF font found for the HUD")

    def render(self, text, antialias, color):
        if self._freetype:
            surface, _ = self._font.render(text, color)
            return surface
        return self._font.render(text, antialias, color)

    def get_width(self, text):
        if self._freetype:
            return self._font.get_rect(text).width
        return self._font.size(text)[0]


class Layout:
    """Screen geometry derived from the actual display size (fullscreen-safe)."""

    def __init__(self, width, height):
        self.width = max(480, int(width))
        self.height = max(360, int(height))

        self.top_bar = max(38, int(self.height * 0.052))
        self.margin = max(8, int(min(self.width, self.height) * 0.013))
        self.gap = max(8, int(min(self.width, self.height) * 0.015))

        avail_w = self.width - 2 * self.margin - (COLUMNS - 1) * self.gap
        avail_h = (
            self.height - self.top_bar - 2 * self.margin - (ROWS - 1) * self.gap
        )
        self.panel_w = max(300, avail_w // COLUMNS)
        self.panel_h = max(170, avail_h // ROWS)
        self.pad = max(6, int(self.panel_h * 0.032))

        self.font_name_size = max(12, int(self.panel_h * 0.085))
        self.font_meta_size = max(9, int(self.panel_h * 0.052))
        self.font_metric_size = max(10, int(self.panel_h * 0.060))
        self.font_top_size = max(11, int(self.top_bar * 0.34))

        # Board: as tall as the card allows, capped so the report keeps room.
        self.cell = max(
            8,
            min(
                (self.panel_h - 2 * self.pad) // GRID_ROWS,
                int(self.panel_w * 0.46) // GRID_COLS,
            ),
        )
        self.grid_w = self.cell * GRID_COLS
        self.grid_h = self.cell * GRID_ROWS

        block_w = COLUMNS * self.panel_w + (COLUMNS - 1) * self.gap
        block_h = ROWS * self.panel_h + (ROWS - 1) * self.gap
        self.origin_x = max(self.margin, (self.width - block_w) // 2)
        self.origin_y = self.top_bar + max(
            self.margin, (self.height - self.top_bar - block_h) // 2
        )

    def panel_rect(self, index):
        col = index % COLUMNS
        row = index // COLUMNS
        x = self.origin_x + col * (self.panel_w + self.gap)
        y = self.origin_y + row * (self.panel_h + self.gap)
        return pygame.Rect(x, y, self.panel_w, self.panel_h)

    def board_rect(self, rect):
        x = rect.x + self.pad
        y = rect.y + (rect.h - self.grid_h) // 2
        return pygame.Rect(x, y, self.grid_w, self.grid_h)

    def info_rect(self, rect):
        x = rect.x + 3 * self.pad + self.grid_w
        return pygame.Rect(x, rect.y, max(60, rect.right - self.pad - x), rect.h)


def build_fonts(layout):
    return {
        "name": _Font(layout.font_name_size, bold=True),
        "meta": _Font(layout.font_meta_size),
        "metric": _Font(layout.font_metric_size),
        "top": _Font(layout.font_top_size, bold=True),
        "top_meta": _Font(max(9, int(layout.font_top_size * 0.82))),
    }


def draw_agent(surface, x, y, size):
    """Draws a stylized chicken/character."""
    cx = x + size // 2
    cy = y + size // 2
    u = size / 80.0
    radius = max(4, int(size * 0.34))

    # Body (Yellow)
    pygame.draw.circle(surface, (255, 214, 0), (cx, cy), radius)
    pygame.draw.circle(surface, (220, 180, 0), (cx, cy), radius, max(1, int(2 * u)))

    # Comb on head (Red)
    comb_pts = [
        (cx - int(6 * u), cy - radius),
        (cx, cy - radius - int(7 * u)),
        (cx + int(6 * u), cy - radius),
    ]
    pygame.draw.polygon(surface, (220, 20, 60), comb_pts)

    # Beak (Orange)
    beak_pts = [
        (cx - int(4 * u), cy - int(4 * u)),
        (cx + int(4 * u), cy - int(4 * u)),
        (cx, cy - int(12 * u)),
    ]
    pygame.draw.polygon(surface, (255, 110, 0), beak_pts)

    # Eyes
    eye_r = max(1, int(3 * u))
    pygame.draw.circle(surface, (20, 20, 20), (cx - int(7 * u), cy - int(2 * u)), eye_r)
    pygame.draw.circle(surface, (20, 20, 20), (cx + int(7 * u), cy - int(2 * u)), eye_r)


def draw_car(surface, x, y, size, color):
    """Draws a stylized car with windshield and headlights."""
    u = size / 80.0
    padding_x = int(6 * u)
    padding_y = int(12 * u)
    car_w = size - (padding_x * 2)
    car_h = size - (padding_y * 2)
    car_rect = pygame.Rect(x + padding_x, y + padding_y, car_w, car_h)

    pygame.draw.rect(surface, color, car_rect, border_radius=max(2, int(8 * u)))
    pygame.draw.rect(
        surface, (20, 20, 20), car_rect, width=1, border_radius=max(2, int(8 * u))
    )

    roof_w = int(car_w * 0.5)
    roof_h = int(car_h * 0.6)
    roof_rect = pygame.Rect(
        x + padding_x + (car_w - roof_w) // 2,
        y + padding_y + (car_h - roof_h) // 2,
        roof_w,
        roof_h,
    )
    pygame.draw.rect(surface, (200, 230, 255), roof_rect, border_radius=max(1, int(4 * u)))

    hl_h = max(2, int(4 * u))
    hl_w = max(1, int(3 * u))
    hl_x = x + padding_x + car_w - hl_w
    pygame.draw.rect(surface, (255, 255, 180), (hl_x, y + padding_y + int(4 * u), hl_w, hl_h))
    pygame.draw.rect(
        surface,
        (255, 255, 180),
        (hl_x, y + padding_y + car_h - hl_h - int(4 * u), hl_w, hl_h),
    )


def draw_badge(screen, font, text, color, left, top, max_width):
    """Filled pill with dark text; returns the rect it occupied."""
    label = font.render(text, True, (14, 15, 18))
    width = min(max_width, label.get_width() + 2 * max(5, font.get_width("M")))
    rect = pygame.Rect(left, top, width, label.get_height() + 8)
    pygame.draw.rect(screen, color, rect, border_radius=5)
    screen.blit(label, (rect.x + max(5, font.get_width("M")), rect.y + 4))
    return rect


def wrap_text(font, text, max_width):
    """Greedy word wrap using the font's real glyph widths."""
    lines = []
    current = ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if current and font.get_width(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def inferences_button_rect(layout, font):
    """Top-bar ``Inferences`` button geometry.

    Deterministic for a given layout+font, so drawing and hit-testing always
    agree on the same rectangle.
    """
    pad = max(8, layout.margin)
    width = font.get_width("Inferences") + 2 * pad
    height = max(20, layout.top_bar - 12)
    return pygame.Rect(
        layout.width - layout.margin - width,
        (layout.top_bar - height) // 2,
        width,
        height,
    )


def draw_inferences_button(screen, font, layout, active):
    """Draw the Inferences button; filled accent while the overlay is open."""
    accent = CATEGORY_COLORS["policy-based"]
    rect = inferences_button_rect(layout, font)
    if active:
        bg, border, fg = accent, accent, (14, 15, 18)
    else:
        bg, border, fg = (30, 34, 44), accent, COLOR_TEXT
    pygame.draw.rect(screen, bg, rect, border_radius=6)
    pygame.draw.rect(screen, border, rect, 1, border_radius=6)
    label = font.render("Inferences", True, fg)
    screen.blit(label, label.get_rect(center=rect.center))
    return rect


def draw_inferences_overlay(screen, fonts, layout, panels):
    """Dimmed modal: one plain-language line per model, best rank first."""
    dim = pygame.Surface((layout.width, layout.height), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 170))
    screen.blit(dim, (0, 0))
    # Keep the button itself above the dim so it reads as the close toggle.
    draw_inferences_button(screen, fonts["top"], layout, True)

    title_font = fonts["top"]
    label_font = fonts["metric"]
    body_font = fonts["meta"]
    hint_font = fonts["top_meta"]

    accent = CATEGORY_COLORS["policy-based"]
    pad = max(12, int(min(layout.width, layout.height) * 0.022))
    gap = max(6, pad // 2)
    width = min(layout.width - 2 * layout.margin, max(340, int(layout.width * 0.68)))
    inner_w = width - 2 * pad

    # Rank-ordered rows: "#1 PPO" label + one short explanation.
    ordered = sorted(panels, key=lambda p: (p.rank is None, p.rank or 0))
    labels = []
    for panel in ordered:
        text = (
            f"#{panel.rank} {panel.display_name()}"
            if panel.rank is not None
            else panel.display_name()
        )
        labels.append((text, panel.rank_color(), INFERENCES_WHY.get(panel.algo, "")))

    label_w = max((label_font.get_width(lab) for lab, _, _ in labels), default=0)
    # Narrow window: stack the label above its line instead of side by side.
    stacked = label_w + gap > inner_w // 2
    label_col_w = 0 if stacked else label_w + gap

    rows = []
    for lab, color, text in labels:
        rows.append((lab, color, wrap_text(body_font, text, max(40, inner_w - label_col_w))))

    intro_lines = []
    for para in INFERENCES_INTRO:
        intro_lines.extend(wrap_text(body_font, para, inner_w))
    takeaway_lines = []
    for para in INFERENCES_TAKEAWAY:
        takeaway_lines.extend(wrap_text(body_font, para, inner_w))

    body_h = body_font.render("Ay", True, COLOR_TEXT).get_height()
    title_h = title_font.render("Ay", True, COLOR_TEXT).get_height()
    hint_h = hint_font.render("Ay", True, COLOR_TEXT).get_height()
    label_h = label_font.render("Ay", True, COLOR_TEXT).get_height()

    content_h = title_h + gap
    content_h += len(intro_lines) * (body_h + 1) + gap
    for _, _, lines in rows:
        if stacked:
            content_h += label_h + 1 + len(lines) * (body_h + 1)
        else:
            content_h += max(label_h, len(lines) * (body_h + 1)) + 1
        content_h += gap
    content_h += len(takeaway_lines) * (body_h + 1) + gap + hint_h

    card_h = min(layout.height - 2 * layout.margin, content_h + 2 * pad)
    card = pygame.Rect(
        (layout.width - width) // 2,
        (layout.height - card_h) // 2,
        width,
        card_h,
    )
    pygame.draw.rect(screen, COLOR_CARD, card, border_radius=10)
    pygame.draw.rect(screen, COLOR_CARD_EDGE, card, 1, border_radius=10)

    screen.set_clip(card)
    x = card.x + pad
    y = card.y + pad

    screen.blit(title_font.render(INFERENCES_TITLE, True, accent), (x, y))
    y += title_h + gap
    for line in intro_lines:
        screen.blit(body_font.render(line, True, COLOR_TEXT_DIM), (x, y))
        y += body_h + 1
    y += gap
    pygame.draw.line(
        screen,
        COLOR_CARD_EDGE,
        (x, y - gap // 2),
        (card.right - pad, y - gap // 2),
        1,
    )
    for lab, color, lines in rows:
        lab_surf = label_font.render(lab, True, color)
        if stacked:
            screen.blit(lab_surf, (x, y))
            y += label_h + 1
            for line in lines:
                screen.blit(body_font.render(line, True, COLOR_TEXT), (x, y))
                y += body_h + 1
        else:
            block_h = max(label_h, len(lines) * (body_h + 1))
            screen.blit(lab_surf, (x, y + max(0, (block_h - label_h) // 2)))
            ly = y
            for line in lines:
                screen.blit(body_font.render(line, True, COLOR_TEXT), (x + label_col_w, ly))
                ly += body_h + 1
            y += block_h + 1
        y += gap
    for line in takeaway_lines:
        screen.blit(body_font.render(line, True, (210, 220, 240)), (x, y))
        y += body_h + 1
    y += gap
    screen.blit(
        hint_font.render("Click Inferences (or press I) to close", True, COLOR_TEXT_DIM),
        (x, y),
    )
    screen.set_clip(None)


class AgentPanel:
    """One algorithm: own environment, checkpoint, counters and restarts."""

    def __init__(self, algo, rank=None):
        self.algo = algo
        self.spec = ALGORITHM_SPECS[algo]
        self.rank = rank  # unique 1-based rank (#1, #2, #3, ...), or None
        self.env = make_env()
        self.act = None
        self.error = None
        self.metrics = None  # comparison_metrics profile
        self.verdict = None

        self.episode = 0
        self.step_count = 0
        self.ep_reward = 0.0
        self.last_action = None
        self.result = None  # last finished episode: GOAL / CRASH / TIME LIMIT
        self.flash = 0
        self.pending_reset = False
        self.goals = 0
        self.episodes_done = 0

        self._load()

    def display_name(self):
        return self.spec["label"]

    def rank_color(self):
        if self.rank is None:
            return RANK_COLOR_DEFAULT
        return RANK_COLORS.get(self.rank, RANK_COLOR_DEFAULT)

    # -- setup ---------------------------------------------------------
    def _load(self):
        self.metrics = comparison_metrics(self.algo)
        self.verdict = convergence(
            {
                "mean_reward": self.metrics.get("mean_reward"),
                "success_rate": self.metrics.get("success_rate"),
                "episodes": self.metrics.get("eval_episodes"),
            }
        )
        try:
            self.act = load_policy(self.algo)
        except CheckpointError as exc:
            self.error = str(exc)
        except Exception as exc:  # never let one bad panel kill the window
            self.error = f"{type(exc).__name__}: {exc}"
        self.restart()

    def restart(self):
        self.goals = 0
        self.episodes_done = 0
        self.episode = 0
        self.result = None
        self.flash = 0
        self._start_episode()

    def _start_episode(self):
        self.obs, _ = self.env.reset(seed=SEED)
        self.episode += 1
        self.step_count = 0
        self.ep_reward = 0.0
        self.last_action = None
        self.pending_reset = False

    # -- simulation (exactly one env step per call) ---------------------
    def tick(self):
        if self.act is None:
            return
        if self.pending_reset:
            self._start_episode()
        if self.flash > 0:
            self.flash -= 1

        action = int(self.act(self.obs))
        self.last_action = action
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.ep_reward += float(reward)
        self.step_count += 1

        if terminated or truncated:
            reason = info.get("reason")
            if reason == "goal_reached":
                self.result = "GOAL"
                self.goals += 1
            elif reason == "collision":
                self.result = "CRASH"
            else:
                self.result = "TIME LIMIT"
            self.episodes_done += 1
            self.flash = RESULT_FLASH_FRAMES
            # Restart on the next frame: no panel ever stalls the others.
            self.pending_reset = True

    # -- text ----------------------------------------------------------
    def live_line(self):
        if self.act is None:
            return "no policy loaded"
        text = (
            f"Ep {self.episode}  ·  Step {self.step_count}"
            f"  ·  reward {self.ep_reward:+.1f}"
            f"  ·  {ACTION_NAMES.get(self.last_action, '-')}"
        )
        if self.episodes_done:
            text += f"  ·  goals {self.goals}/{self.episodes_done}"
        return text

    def report_rows(self):
        """Comparison rows that actually separate the six models."""
        if self.act is None:
            return []
        metrics = self.metrics or {}
        if metrics.get("mean_reward") is None:
            return []

        def fmt_steps(value):
            if value is None:
                return "—"
            return f"{int(value):,}"

        def fmt_secs(value):
            if value is None:
                return "—"
            value = float(value)
            if value < 60:
                return f"{value:.1f}s"
            return f"{value / 60.0:.1f}m"

        rows = [
            ("mean reward", f"{metrics['mean_reward']:+.2f}"),
            ("success rate", f"{100.0 * metrics['success_rate']:.0f}%"),
            # Path efficiency: shorter goal episodes rank higher.
            ("path length", f"{metrics['mean_length']:.1f} steps"),
            # Sample efficiency: earlier first convergence ranks higher, but the
            # restored-checkpoint step breaks ties between transient runs.
            ("first converged", fmt_steps(metrics.get("steps_to_converge"))),
            ("best checkpoint", fmt_steps(metrics.get("best_checkpoint_step"))),
            ("reward / step", f"{metrics['reward_per_step']:+.2f}"),
            ("train time", fmt_secs(metrics.get("train_duration_sec"))),
        ]
        if metrics.get("train_success_rate") is not None:
            rows.append(
                (
                    "train success",
                    f"{100.0 * metrics['train_success_rate']:.0f}%",
                )
            )
        return rows

    def verdict_chip(self):
        """(text, colour) for the convergence badge."""
        if self.act is None:
            return "NO CHECKPOINT", MISSING_COLOR
        verdict = self.verdict or {}
        if not verdict.get("evaluated"):
            return "NOT EVALUATED", COLOR_TEXT_DIM
        if verdict.get("converged"):
            return "CONVERGED", GOAL_COLOR
        return "NOT CONVERGED", TIMEOUT_COLOR

    def verdict_note(self):
        """Criterion / hint line shown under the badge."""
        if self.act is None:
            return f"python train.py --algo {self.algo}"
        verdict = self.verdict or {}
        criterion = verdict.get("criterion") or {}
        if not verdict.get("evaluated"):
            return "run python evaluate.py"
        return (
            f"needs >={criterion.get('min_success_rate', 0.9):.0%} success and "
            f">={criterion.get('min_mean_reward', 100.0):+.0f} mean reward"
        )

    def status_chip(self):
        if self.act is None:
            return "NO CHECKPOINT", MISSING_COLOR
        if self.result is None:
            return "running", RUNNING_COLOR
        return f"last: {self.result}", RESULT_COLORS.get(self.result, COLOR_TEXT_DIM)

    # -- drawing -------------------------------------------------------
    def draw(self, screen, fonts, layout, rect):
        screen.set_clip(rect)

        pygame.draw.rect(screen, COLOR_CARD, rect)
        pygame.draw.rect(screen, COLOR_CARD_EDGE, rect, 1)

        board = layout.board_rect(rect)
        self._draw_grid(screen, fonts, layout, board)

        flash_color = RESULT_COLORS.get(self.result)
        if self.flash > 0 and flash_color is not None:
            width = max(2, layout.cell // 9)
            pygame.draw.rect(
                screen, flash_color, board.inflate(width + 2, width + 2), width
            )

        if self.act is None:
            self._draw_missing(screen, fonts, layout, board)

        self._draw_info(screen, fonts, layout, rect)
        screen.set_clip(None)

    def _draw_info(self, screen, fonts, layout, rect):
        info = layout.info_rect(rect)
        pad = layout.pad
        y = rect.y + pad

        # Rank badge (#1 / #2 / …) then model name; status chip on the right.
        name_x = info.x
        if self.rank is not None:
            rank_rect = draw_badge(
                screen,
                fonts["name"],
                f"#{self.rank}",
                self.rank_color(),
                info.x,
                y,
                info.w,
            )
            name_x = rank_rect.right + max(6, pad // 2)

        name = fonts["name"].render(self.display_name(), True, COLOR_TEXT)
        screen.blit(name, (name_x, y + max(0, (rank_rect.h - name.get_height()) // 2) if self.rank is not None else y))

        chip_text, chip_color = self.status_chip()
        chip_font = fonts["top_meta"]
        chip_label = chip_font.render(chip_text, True, chip_color)
        chip = pygame.Rect(0, 0, chip_label.get_width() + 12, chip_label.get_height() + 6)
        chip.topright = (info.right, y)
        pygame.draw.rect(
            screen,
            (chip_color[0] // 6, chip_color[1] // 6, chip_color[2] // 6),
            chip,
            border_radius=4,
        )
        pygame.draw.rect(screen, chip_color, chip, 1, border_radius=4)
        screen.blit(chip_label, (chip.x + 6, chip.y + 3))

        title_h = name.get_height()
        if self.rank is not None:
            title_h = max(title_h, rank_rect.h)

        # Category + live counters
        live_y = y + title_h + max(2, pad // 2)
        live = f"{self.spec['category']}  ·  {self.live_line()}"
        live_surf = fonts["meta"].render(live, True, COLOR_TEXT_DIM)
        if live_surf.get_width() > info.w:
            live_surf = fonts["top_meta"].render(live, True, COLOR_TEXT_DIM)
        screen.blit(live_surf, (info.x, live_y))

        # Comparison report (metrics that separate the models)
        report_y = live_y + live_surf.get_height() + pad
        report_y = self._draw_report(screen, fonts, layout, info, report_y)

        # Convergence verdict anchored to the bottom of the card
        badge_font = fonts["meta"]
        note_font = fonts["meta"]
        chip_text, chip_color = self.verdict_chip()
        note = self.verdict_note()
        note_surf = note_font.render(note, True, COLOR_TEXT_DIM)
        if note_surf.get_width() > info.w:
            note_surf = fonts["top_meta"].render(note, True, COLOR_TEXT_DIM)

        badge_h = badge_font.render("CONVERGED", True, COLOR_TEXT).get_height() + 8
        block_h = badge_h + 4 + note_surf.get_height()
        badge_y = max(report_y, rect.bottom - pad - block_h)

        draw_badge(screen, badge_font, chip_text, chip_color, info.x, badge_y, info.w)
        screen.blit(note_surf, (info.x, badge_y + badge_h + 4))

    def _draw_report(self, screen, fonts, layout, info, top):
        pad = layout.pad
        label_font = fonts["meta"]
        value_font = fonts["metric"]

        header = label_font.render(
            "COMPARISON",
            True,
            CATEGORY_COLORS.get(self.spec["category"], COLOR_TEXT_DIM),
        )
        screen.blit(header, (info.x, top))
        y = top + header.get_height() + max(2, pad // 3)

        rows = self.report_rows()
        if not rows:
            msg = label_font.render(
                "no checkpoint" if self.act is None else "no evaluation metrics",
                True,
                COLOR_TEXT_DIM,
            )
            screen.blit(msg, (info.x, y))
            y += msg.get_height()
            if self.act is not None:
                hint = fonts["top_meta"].render(
                    "run: python evaluate.py", True, COLOR_TEXT_DIM
                )
                screen.blit(hint, (info.x, y))
                y += hint.get_height()
            return y + pad

        row_h = value_font.render("Ag", True, COLOR_TEXT).get_height() + max(2, pad // 3)
        for label, value in rows:
            lab = label_font.render(label, True, COLOR_TEXT_DIM)
            val = value_font.render(value, True, COLOR_TEXT)
            screen.blit(lab, (info.x, y + (row_h - lab.get_height()) // 2))
            screen.blit(
                val, (info.right - val.get_width(), y + (row_h - val.get_height()) // 2)
            )
            y += row_h
        return y + max(2, pad // 2)

    def _draw_grid(self, screen, fonts, layout, rect):
        cell = layout.cell
        env = self.env
        for r in range(GRID_ROWS):
            row_y = rect.y + r * cell
            if r == 0:
                color = COLOR_GOAL_ROW
            elif r == 6:
                color = COLOR_SAFE_ROW
            elif r == 7:
                color = COLOR_START_ROW
            else:
                color = COLOR_ROAD
            pygame.draw.rect(screen, color, (rect.x, row_y, rect.w, cell))

            if 1 <= r <= 5:
                pygame.draw.line(
                    screen, (28, 28, 32), (rect.x, row_y), (rect.right, row_y), 1
                )
                dash_len = max(4, cell // 3)
                gap_len = max(3, cell // 4)
                curr_x = rect.x
                while curr_x < rect.right:
                    pygame.draw.line(
                        screen,
                        COLOR_LANE_DASH,
                        (curr_x, row_y + cell // 2),
                        (curr_x + dash_len, row_y + cell // 2),
                        1,
                    )
                    curr_x += dash_len + gap_len

        goal_lbl = fonts["top_meta"].render("GOAL", True, (235, 255, 235))
        screen.blit(
            goal_lbl,
            (rect.x + rect.w // 2 - goal_lbl.get_width() // 2, rect.y + max(1, cell // 20)),
        )

        if env.agent_pos is None:
            return

        for idx, (car_row, car_x) in enumerate(zip(env.car_rows, env.car_positions)):
            draw_car(
                screen,
                rect.x + car_x * cell,
                rect.y + car_row * cell,
                cell,
                CAR_COLORS[idx % len(CAR_COLORS)],
            )

        draw_agent(
            screen,
            rect.x + env.agent_pos[0] * cell,
            rect.y + env.agent_pos[1] * cell,
            cell,
        )

    def _draw_missing(self, screen, fonts, layout, grid):
        overlay = pygame.Surface((grid.w, grid.h), pygame.SRCALPHA)
        overlay.fill((10, 10, 14, 205))
        screen.blit(overlay, grid.topleft)

        lines = [
            ("NO CHECKPOINT", MISSING_COLOR),
            (self.spec["checkpoint"], COLOR_TEXT_DIM),
            (f"python train.py --algo {self.algo}", (210, 210, 140)),
        ]
        line_h = fonts["top_meta"].render("Ay", True, COLOR_TEXT).get_height() + 3
        start_y = grid.y + grid.h // 2 - (len(lines) * line_h) // 2
        for i, (text, color) in enumerate(lines):
            surf = fonts["top_meta"].render(text, True, color)
            while surf.get_width() > grid.w - 2 and len(text) > 4:
                text = text[:-2]
                surf = fonts["top_meta"].render(text, True, color)
            screen.blit(
                surf, (grid.x + (grid.w - surf.get_width()) // 2, start_y + i * line_h)
            )


def draw_top_bar(screen, fonts, layout, fps, paused, fullscreen, show_inferences=False):
    pygame.draw.rect(screen, COLOR_BG_HEADER, (0, 0, layout.width, layout.top_bar))
    pygame.draw.line(
        screen, (50, 50, 60), (0, layout.top_bar), (layout.width, layout.top_bar), 2
    )

    title = fonts["top"].render(
        "Crossy Road  ·  ranked by success, reward, path length, sample efficiency",
        True,
        COLOR_TEXT,
    )
    screen.blit(title, (layout.margin, (layout.top_bar - title.get_height()) // 2))

    mode = "fullscreen" if fullscreen else "windowed"
    status = (
        f"{fps} FPS  ·  1 step/frame  ·  "
        f"{'PAUSED' if paused else 'running'}  ·  {mode}"
    )
    controls = (
        "Space pause · ↑/↓ speed · F fullscreen · R restart · I inferences · Esc quit"
    )
    button = inferences_button_rect(layout, fonts["top"])
    for candidate in (f"{status}   |   {controls}", status):
        surf = fonts["top_meta"].render(candidate, True, COLOR_TEXT_DIM)
        if (
            title.get_width() + surf.get_width() + button.w + 5 * layout.margin
            <= layout.width
        ):
            screen.blit(
                surf,
                (
                    button.left - layout.margin - surf.get_width(),
                    (layout.top_bar - surf.get_height()) // 2,
                ),
            )
            break

    draw_inferences_button(screen, fonts["top"], layout, show_inferences)


def draw_scene(
    screen, fonts, panels, layout, fps, paused, fullscreen, show_inferences=False
):
    screen.fill(COLOR_BG)
    draw_top_bar(screen, fonts, layout, fps, paused, fullscreen, show_inferences)

    for index, panel in enumerate(panels):
        panel.draw(screen, fonts, layout, layout.panel_rect(index))

    if show_inferences:
        draw_inferences_overlay(screen, fonts, layout, panels)

    # Present the frame. Without this the double-buffered window stays black.
    pygame.display.flip()


_FRAME_EXT = None


def save_frame(screen, directory, frame):
    """Save one frame; fall back to BMP when SDL_image is unavailable."""
    global _FRAME_EXT
    if _FRAME_EXT is None:
        candidate = os.path.join(directory, "frame_0000.png")
        try:
            pygame.image.save(screen, candidate)
            _FRAME_EXT = ".png"
        except (NotImplementedError, pygame.error):
            _FRAME_EXT = ".bmp"
        else:
            return candidate
    path = os.path.join(directory, f"frame_{frame:04d}{_FRAME_EXT}")
    pygame.image.save(screen, path)
    return path


def load_panels():
    ranks = algorithm_rankings(PANEL_ORDER)
    panels = [AgentPanel(algo, rank=ranks.get(algo)) for algo in PANEL_ORDER]
    print(
        "Rank key: converged → success → mean reward → shorter path → first "
        "convergence → earlier restored checkpoint (unique ranks; fixed panel "
        "order is the final tie-break)"
    )
    for panel in panels:
        spec = panel.spec
        rank_tag = f"#{panel.rank}" if panel.rank is not None else "--"
        metrics = panel.metrics or {}
        if panel.act is not None:
            converged = (panel.verdict or {}).get("converged")
            converge_steps = metrics.get("steps_to_converge")
            best_steps = metrics.get("best_checkpoint_step")
            converge_txt = (
                f"{int(converge_steps):,}" if converge_steps is not None else "—"
            )
            best_txt = f"{int(best_steps):,}" if best_steps is not None else "—"
            print(
                f"[ok]   {rank_tag:>3s} {spec['label']:11s} {spec['category']:13s} "
                f"mean {metrics.get('mean_reward', float('nan')):+8.2f}  "
                f"len {metrics.get('mean_length', float('nan')):5.1f}  "
                f"conv@{converge_txt:>6s}  best@{best_txt:>6s}  "
                f"{'converged' if converged else 'not converged'}"
            )
        else:
            print(
                f"[miss] {rank_tag:>3s} {spec['label']:11s} {spec['category']:13s} "
                f"{spec['checkpoint']:28s} {panel.error}"
            )
    return panels


def create_display(fullscreen, size):
    flags = pygame.FULLSCREEN if fullscreen else 0
    target = (0, 0) if fullscreen else size
    screen = pygame.display.set_mode(target, flags)
    pygame.display.set_caption(
        "Crossy Road - 6 algorithm comparison (policy / value / hybrid)"
    )
    layout = Layout(*screen.get_size())
    fonts = build_fonts(layout)
    return screen, layout, fonts


def parse_size(text):
    try:
        width, height = text.lower().split("x")
        return max(480, int(width)), max(360, int(height))
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 1280x800")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay the six trained Crossy Road agents side by side "
            "(2 columns x 3 rows), fullscreen by default."
        )
    )
    parser.add_argument(
        "--fps", type=int, default=4, help="Environment steps per second (default: 4)."
    )
    parser.add_argument(
        "--windowed", action="store_true", help="Run in a window instead of fullscreen."
    )
    parser.add_argument(
        "--size",
        type=parse_size,
        default=(1200, 900),
        help="Window size for --windowed / --headless (default: 1200x900).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use SDL's dummy video driver (no window; for CI/screenshots).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop after N frames (default: 0 = run until closed).",
    )
    parser.add_argument(
        "--save-frames",
        default=None,
        metavar="DIR",
        help="Save every rendered frame as an image in DIR.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    headless = args.headless
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    try:
        pygame.init()
        fullscreen = not (args.windowed or headless)
        screen, layout, fonts = create_display(fullscreen, args.size)
    except pygame.error as exc:
        print(f"Pygame Display Error: {exc}")
        print(
            "Note: if running in a headless or remote shell, ensure an X11/Wayland "
            "display server is available, or pass --headless."
        )
        return 1

    panels = load_panels()

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)

    clock = pygame.time.Clock()
    fps = max(1, args.fps)
    paused = False
    show_inferences = False
    frame = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if show_inferences:
                        show_inferences = False
                    else:
                        running = False
                elif event.key == pygame.K_i:
                    show_inferences = not show_inferences
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    fps = min(30, fps + 1)
                elif event.key == pygame.K_DOWN:
                    fps = max(1, fps - 1)
                elif event.key == pygame.K_r:
                    for panel in panels:
                        panel.restart()
                elif event.key == pygame.K_f and not headless:
                    fullscreen = not fullscreen
                    screen, layout, fonts = create_display(fullscreen, args.size)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if inferences_button_rect(layout, fonts["top"]).collidepoint(event.pos):
                    show_inferences = not show_inferences

        if not paused:
            # Strictly uniform: every loaded panel advances exactly one step.
            for panel in panels:
                panel.tick()

        draw_scene(
            screen, fonts, panels, layout, fps, paused, fullscreen, show_inferences
        )

        if args.save_frames:
            save_frame(screen, args.save_frames, frame)

        clock.tick(fps)
        frame += 1
        if args.frames and frame >= args.frames:
            running = False

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
