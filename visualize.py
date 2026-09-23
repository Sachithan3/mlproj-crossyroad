import sys
import time
import pygame
from stable_baselines3 import PPO
from crossy_env import GridEnv

# --- Constants & Configuration ---
CELL_SIZE = 80
GRID_WIDTH = 7
GRID_HEIGHT = 8

HUD_HEIGHT = 70
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE + HUD_HEIGHT

# Palette
COLOR_BG_HEADER = (30, 30, 36)
COLOR_TEXT = (240, 240, 240)
COLOR_TEXT_DIM = (160, 160, 170)
COLOR_GOAL_ROW = (46, 125, 50)       # Lush Green
COLOR_SAFE_ROW = (56, 142, 60)       # Dark Green
COLOR_START_ROW = (74, 90, 102)      # Pavement Gray
COLOR_ROAD = (40, 42, 48)            # Dark Asphalt
COLOR_LANE_DASH = (220, 220, 220)

# Car colors for rows 1 to 5
CAR_COLORS = [
    (231, 76, 60),   # Red
    (230, 126, 34),  # Orange
    (52, 152, 219),  # Blue
    (155, 89, 182),  # Purple
    (241, 196, 15),  # Yellow
]

ACTION_NAMES = {
    0: "UP ⬆",
    1: "DOWN ⬇",
    2: "LEFT ⬅",
    3: "RIGHT ➡",
    4: "STAY ⏸",
}


def draw_agent(surface, x, y, size):
    """Draws a stylized chicken/character."""
    cx = x + size // 2
    cy = y + size // 2
    radius = int(size * 0.36)

    # Body (Yellow)
    pygame.draw.circle(surface, (255, 214, 0), (cx, cy), radius)
    pygame.draw.circle(surface, (220, 180, 0), (cx, cy), radius, 2)

    # Comb on head (Red)
    comb_pts = [(cx - 6, cy - radius), (cx, cy - radius - 7), (cx + 6, cy - radius)]
    pygame.draw.polygon(surface, (220, 20, 60), comb_pts)

    # Beak (Orange)
    beak_pts = [(cx - 4, cy - 4), (cx + 4, cy - 4), (cx, cy - 12)]
    pygame.draw.polygon(surface, (255, 110, 0), beak_pts)

    # Eyes
    pygame.draw.circle(surface, (20, 20, 20), (cx - 7, cy - 2), 3)
    pygame.draw.circle(surface, (20, 20, 20), (cx + 7, cy - 2), 3)


def draw_car(surface, x, y, size, color):
    """Draws a stylized car with windshield and wheels."""
    padding_x = 6
    padding_y = 12
    car_w = size - (padding_x * 2)
    car_h = size - (padding_y * 2)
    car_rect = pygame.Rect(x + padding_x, y + padding_y, car_w, car_h)

    # Car body with rounded corners
    pygame.draw.rect(surface, color, car_rect, border_radius=8)
    pygame.draw.rect(surface, (20, 20, 20), car_rect, width=2, border_radius=8)

    # Windshield / Roof
    roof_w = int(car_w * 0.5)
    roof_h = int(car_h * 0.6)
    roof_rect = pygame.Rect(x + padding_x + (car_w - roof_w) // 2, y + padding_y + (car_h - roof_h) // 2, roof_w, roof_h)
    pygame.draw.rect(surface, (200, 230, 255), roof_rect, border_radius=4)

    # Headlights (right side since cars move right)
    hl_y1 = y + padding_y + 4
    hl_y2 = y + padding_y + car_h - 8
    hl_x = x + padding_x + car_w - 4
    pygame.draw.rect(surface, (255, 255, 180), (hl_x, hl_y1, 3, 4))
    pygame.draw.rect(surface, (255, 255, 180), (hl_x, hl_y2, 3, 4))


def render_game(screen, font_large, font_small, env, last_action, ep_num, ep_reward, step_count, status_msg, fps):
    # 1. Clear background
    screen.fill((20, 20, 20))

    # 2. Draw HUD header
    pygame.draw.rect(screen, COLOR_BG_HEADER, (0, 0, WINDOW_WIDTH, HUD_HEIGHT))
    pygame.draw.line(screen, (50, 50, 60), (0, HUD_HEIGHT), (WINDOW_WIDTH, HUD_HEIGHT), 2)

    act_str = ACTION_NAMES.get(last_action, "START")
    hud_l1 = font_large.render(f"Ep {ep_num} | Step {step_count} | Action: {act_str}", True, COLOR_TEXT)
    hud_l2 = font_small.render(f"Reward: {ep_reward:+.1f}  |  Speed: {fps} FPS  (Controls: Space=Pause, ↑/↓=Speed)", True, COLOR_TEXT_DIM)
    screen.blit(hud_l1, (12, 10))
    screen.blit(hud_l2, (12, 42))

    # 3. Draw Grid Rows
    for r in range(GRID_HEIGHT):
        row_y = HUD_HEIGHT + r * CELL_SIZE

        # Row Background
        if r == 0:
            color = COLOR_GOAL_ROW
        elif r == 6:
            color = COLOR_SAFE_ROW
        elif r == 7:
            color = COLOR_START_ROW
        else:
            color = COLOR_ROAD

        pygame.draw.rect(screen, color, (0, row_y, WINDOW_WIDTH, CELL_SIZE))

        # Lane dividers for road rows
        if 1 <= r <= 5:
            pygame.draw.line(screen, (30, 30, 30), (0, row_y), (WINDOW_WIDTH, row_y), 1)
            # Dashed middle line
            dash_len = 16
            gap_len = 12
            curr_x = 0
            while curr_x < WINDOW_WIDTH:
                pygame.draw.line(screen, COLOR_LANE_DASH, (curr_x, row_y + CELL_SIZE // 2), (curr_x + dash_len, row_y + CELL_SIZE // 2), 1)
                curr_x += dash_len + gap_len

        # Label Goal and Start
        if r == 0:
            goal_lbl = font_large.render("🏁 GOAL ROW 🏁", True, (240, 255, 240))
            screen.blit(goal_lbl, (WINDOW_WIDTH // 2 - goal_lbl.get_width() // 2, row_y + 25))
        elif r == 7:
            start_lbl = font_small.render("START ROW", True, (200, 220, 230))
            screen.blit(start_lbl, (WINDOW_WIDTH // 2 - start_lbl.get_width() // 2, row_y + 30))

    # 4. Draw Cars
    for idx, (car_row, car_x) in enumerate(zip(env.car_rows, env.car_positions)):
        px = car_x * CELL_SIZE
        py = HUD_HEIGHT + car_row * CELL_SIZE
        car_color = CAR_COLORS[idx % len(CAR_COLORS)]
        draw_car(screen, px, py, CELL_SIZE, car_color)

    # 5. Draw Agent
    if env.agent_pos is not None:
        ax = env.agent_pos[0] * CELL_SIZE
        ay = HUD_HEIGHT + env.agent_pos[1] * CELL_SIZE
        draw_agent(screen, ax, ay, CELL_SIZE)

    # 6. Status Banner (if goal or crash)
    if status_msg:
        banner_rect = pygame.Rect(30, HUD_HEIGHT + 240, WINDOW_WIDTH - 60, 80)
        banner_color = (39, 174, 96, 230) if "GOAL" in status_msg else (192, 57, 43, 230)
        banner_surf = pygame.Surface((banner_rect.width, banner_rect.height), pygame.SRCALPHA)
        banner_surf.fill(banner_color)
        screen.blit(banner_surf, (banner_rect.x, banner_rect.y))
        pygame.draw.rect(screen, (255, 255, 255), banner_rect, 2, border_radius=6)

        msg_surf = font_large.render(status_msg, True, (255, 255, 255))
        screen.blit(msg_surf, (WINDOW_WIDTH // 2 - msg_surf.get_width() // 2, banner_rect.y + 24))

    pygame.display.flip()


def main():
    try:
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Crossy Road - PPO Agent Visualization")
    except pygame.error as e:
        print(f"Pygame Display Error: {e}")
        print("Note: If running in a headless or remote shell, ensure an X11/Wayland display server is available.")
        sys.exit(1)

    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("dejavusans", 18, bold=True)
    font_small = pygame.font.SysFont("dejavusans", 13)

    # Load trained model & base environment
    print("Loading ppo_crossy_agent.zip...")
    model = PPO.load("ppo_crossy_agent")
    env = GridEnv()

    fps = 4
    paused = False
    episode = 1

    obs, _ = env.reset()
    ep_reward = 0
    step_count = 0
    last_action = None
    status_msg = None

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    fps = min(20, fps + 1)
                elif event.key == pygame.K_DOWN:
                    fps = max(1, fps - 1)
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    ep_reward = 0
                    step_count = 0
                    last_action = None
                    status_msg = None

        if not paused:
            if status_msg is not None:
                # Pause on end screen for a moment before restarting
                time.sleep(1.2)
                episode += 1
                obs, _ = env.reset()
                ep_reward = 0
                step_count = 0
                last_action = None
                status_msg = None

            # Get model action
            action, _ = model.predict(obs, deterministic=True)
            last_action = int(action)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(last_action)
            ep_reward += reward
            step_count += 1

            if terminated or truncated:
                reason = info.get("reason", "done")
                if reason == "goal_reached":
                    status_msg = "🎉 GOAL REACHED! 🎉"
                else:
                    status_msg = "💥 COLLISION! 💥"

        render_game(screen, font_large, font_small, env, last_action, episode, ep_reward, step_count, status_msg, fps)
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()
