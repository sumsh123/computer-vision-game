import math
import random
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Basic game settings
WIDTH, HEIGHT = 1000, 700
FPS = 60
PLAYER_SIZE = 48
STARTING_LIVES = 3

# Colours
BACKGROUND = (12, 16, 28)
WHITE = (245, 247, 255)
GREY = (160, 170, 195)
BLUE = (65, 130, 255)
CYAN = (70, 225, 255)
RED = (255, 80, 105)
GOLD = (255, 205, 70)
GREEN = (80, 230, 145)

PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_FOLDER / "models" / "hand_landmarker.task"
HIGH_SCORE_FILE = PROJECT_FOLDER / "highscore.txt"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

# Lines used to draw the hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def keep_between(value, low, high):
    return max(low, min(value, high))


def download_model_if_needed():
    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 1_000_000:
        return

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = MODEL_FILE.with_suffix(".download")

    print("Downloading the MediaPipe hand model...")

    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "Mozilla/5.0 FingerForce/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            temp_file.write_bytes(response.read())

        if temp_file.stat().st_size < 1_000_000:
            raise RuntimeError("The downloaded model file is incomplete.")

        temp_file.replace(MODEL_FILE)
        print("Model downloaded successfully.")

    except Exception as error:
        temp_file.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the hand model. Check your internet connection."
        ) from error


class HandTracker:
    def __init__(self, camera_number=0):
        download_model_if_needed()

        self.camera = cv2.VideoCapture(camera_number, cv2.CAP_DSHOW)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = cv2.VideoCapture(camera_number)

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        settings = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_FILE)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )

        self.detector = vision.HandLandmarker.create_from_options(settings)
        self.camera_available = self.camera.isOpened()
        self.last_timestamp = 0
        self.smoothed_x = None
        self.smoothed_y = None
        self.preview = None

    def update(self):
        if not self.camera_available:
            return None, None, False, False

        success, frame = self.camera.read()

        if not success:
            self.camera_available = False
            return None, None, False, False

        frame = cv2.flip(frame, 1)

        # MediaPipe still analyses the real webcam frame.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # This separate frame is only used for the visible preview.
        # It hides the camera image and shows the hand skeleton on black.
        skeleton_frame = np.zeros_like(frame)

        timestamp = int(time.monotonic() * 1000)
        timestamp = max(timestamp, self.last_timestamp + 1)
        self.last_timestamp = timestamp

        result = self.detector.detect_for_video(mp_image, timestamp)
        hand_found = bool(result.hand_landmarks)

        finger_x = None
        finger_y = None
        pinching = False

        if hand_found:
            landmarks = result.hand_landmarks[0]
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]

            if self.smoothed_x is None:
                self.smoothed_x = index_tip.x
                self.smoothed_y = index_tip.y
            else:
                smoothing = 0.28
                self.smoothed_x += (index_tip.x - self.smoothed_x) * smoothing
                self.smoothed_y += (index_tip.y - self.smoothed_y) * smoothing

            finger_x = keep_between(self.smoothed_x, 0.0, 1.0)
            finger_y = keep_between(self.smoothed_y, 0.0, 1.0)

            distance = math.hypot(
                index_tip.x - thumb_tip.x,
                index_tip.y - thumb_tip.y,
            )
            pinching = distance < 0.065

            self.draw_hand(skeleton_frame, landmarks, pinching)
        else:
            cv2.putText(
                skeleton_frame,
                "SHOW YOUR HAND",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (150, 150, 150),
                2,
            )

        self.preview = self.make_preview(skeleton_frame)
        return finger_x, finger_y, pinching, hand_found

    @staticmethod
    def draw_hand(frame, landmarks, pinching):
        frame_height, frame_width = frame.shape[:2]

        points = [
            (int(point.x * frame_width), int(point.y * frame_height))
            for point in landmarks
        ]

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 210, 60), 2)

        for number, point in enumerate(points):
            colour = (80, 255, 120)
            radius = 4

            if number == 8:
                colour = (0, 220, 255)
                radius = 8
            elif number == 4:
                colour = (255, 100, 100)
                radius = 7

            cv2.circle(frame, point, radius, colour, -1)

        if pinching:
            message = "PINCH: SHIELD"
            colour = (80, 255, 120)
        else:
            message = "MOVE INDEX FINGER"
            colour = (0, 220, 255)

        cv2.putText(
            frame,
            message,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            colour,
            2,
        )

    @staticmethod
    def make_preview(frame):
        frame = cv2.resize(frame, (260, 195))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.swapaxes(frame, 0, 1)
        return pygame.surfarray.make_surface(frame)

    def close(self):
        self.camera.release()
        self.detector.close()


class FallingObject:
    def __init__(self, kind, extra_speed):
        self.kind = kind
        self.size = random.randint(28, 55) if kind == "meteor" else 32

        if kind == "star":
            # triangular() makes the centre more likely than the far edges
            centre = WIDTH / 2 - self.size / 2
            self.x = int(
                random.triangular(
                    80,
                    WIDTH - 80 - self.size,
                    centre,
                )
            )
        else:
            self.x = random.randint(30, WIDTH - 30 - self.size)

        self.y = -self.size
        self.speed = random.randint(210, 290) + extra_speed

        if kind == "star":
            self.speed *= 0.78

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.size, self.size)

    def update(self, delta_time):
        self.y += self.speed * delta_time

    def draw(self, screen):
        centre = (
            int(self.x + self.size / 2),
            int(self.y + self.size / 2),
        )

        if self.kind == "meteor":
            pygame.draw.circle(screen, (115, 75, 82), centre, self.size // 2)
            pygame.draw.circle(
                screen,
                (65, 48, 62),
                (
                    centre[0] - self.size // 7,
                    centre[1] - self.size // 8,
                ),
                max(4, self.size // 8),
            )
        else:
            draw_star(screen, centre, self.size // 2, GOLD)


def draw_star(screen, centre, radius, colour):
    points = []

    for i in range(10):
        current_radius = radius if i % 2 == 0 else radius * 0.45
        angle = -math.pi / 2 + i * math.pi / 5

        x = centre[0] + math.cos(angle) * current_radius
        y = centre[1] + math.sin(angle) * current_radius
        points.append((x, y))

    pygame.draw.polygon(screen, colour, points)


def draw_player(screen, x, y, shield_active, recently_hit):
    # Flash briefly after taking damage
    if recently_hit and int(time.monotonic() * 12) % 2 == 0:
        return

    centre = (int(x), int(y))

    if shield_active:
        pygame.draw.circle(
            screen,
            CYAN,
            centre,
            PLAYER_SIZE // 2 + 18,
            4,
        )

    ship = [
        (x, y - PLAYER_SIZE // 2),
        (x - PLAYER_SIZE // 2, y + PLAYER_SIZE // 2),
        (x, y + PLAYER_SIZE // 5),
        (x + PLAYER_SIZE // 2, y + PLAYER_SIZE // 2),
    ]

    pygame.draw.polygon(screen, BLUE, ship)
    pygame.draw.polygon(screen, WHITE, ship, 2)
    pygame.draw.circle(screen, CYAN, centre, 7)


def draw_centered_text(screen, text, font, colour, y):
    image = font.render(text, True, colour)
    rect = image.get_rect(center=(WIDTH // 2, y))
    screen.blit(image, rect)


def read_high_score():
    try:
        return int(HIGH_SCORE_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def write_high_score(score):
    HIGH_SCORE_FILE.write_text(str(score))


def reset_game():
    return {
        "player_x": WIDTH / 2,
        "player_y": HEIGHT - 100,
        "objects": [],
        "score": 0,
        "lives": STARTING_LIVES,
        "shield": 100.0,
        "spawn_timer": 0.0,
        "hit_timer": 0.0,
        "elapsed": 0.0,
    }


def move_player_with_keyboard(game, keys, delta_time):
    speed = 430 * delta_time

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        game["player_x"] -= speed
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        game["player_x"] += speed
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        game["player_y"] -= speed
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        game["player_y"] += speed

    game["player_x"] = keep_between(game["player_x"], 35, WIDTH - 35)
    game["player_y"] = keep_between(game["player_y"], 80, HEIGHT - 55)


def draw_background(screen, elapsed_time):
    screen.fill(BACKGROUND)

    # The background stars are denser near the middle of the screen.
    for i in range(90):
        spread = ((i * 73) % 100) / 100
        direction = -1 if i % 2 == 0 else 1
        distance_from_middle = (spread ** 1.8) * (WIDTH / 2)
        star_x = int(WIDTH / 2 + direction * distance_from_middle)

        star_y = int(
            (i * 87 + elapsed_time * (18 + i % 4 * 7)) % HEIGHT
        )

        radius = 2 if i % 11 == 0 else 1
        pygame.draw.circle(screen, (90, 100, 125), (star_x, star_y), radius)


def draw_hud(screen, game, high_score, fonts):
    big_font, normal_font, small_font = fonts

    screen.blit(
        big_font.render(f"Score: {game['score']}", True, WHITE),
        (22, 18),
    )
    screen.blit(
        small_font.render(f"High score: {high_score}", True, GREY),
        (25, 57),
    )
    screen.blit(
        normal_font.render(f"Lives: {game['lives']}", True, RED),
        (22, 88),
    )

    screen.blit(small_font.render("Shield", True, GREY), (23, 128))

    pygame.draw.rect(
        screen,
        (55, 62, 82),
        (22, 151, 210, 18),
        border_radius=9,
    )

    shield_width = int(210 * game["shield"] / 100)

    pygame.draw.rect(
        screen,
        CYAN,
        (22, 151, shield_width, 18),
        border_radius=9,
    )
    pygame.draw.rect(
        screen,
        GREY,
        (22, 151, 210, 18),
        2,
        border_radius=9,
    )


def draw_camera_panel(screen, tracker, hand_found, normal_font, small_font):
    pygame.draw.rect(
        screen,
        (25, 31, 48),
        (714, 16, 270, 235),
        border_radius=14,
    )

    if tracker.preview is not None:
        screen.blit(tracker.preview, (719, 21))
    else:
        screen.blit(
            normal_font.render("Camera not available", True, RED),
            (745, 100),
        )

    if hand_found:
        status = "Hand detected"
        colour = GREEN
    else:
        status = "Show your hand"
        colour = GREY

    screen.blit(small_font.render(status, True, colour), (725, 220))


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Finger Force")
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("arial", 52, bold=True)
    big_font = pygame.font.SysFont("arial", 30, bold=True)
    normal_font = pygame.font.SysFont("arial", 21)
    small_font = pygame.font.SysFont("arial", 17)

    tracker = HandTracker(camera_number=0)
    game = reset_game()
    high_score = read_high_score()

    mode = "menu"
    previous_pinch = False
    running = True

    while running:
        delta_time = min(clock.tick(FPS) / 1000, 0.05)

        finger_x, finger_y, pinching, hand_found = tracker.update()
        pinch_started = pinching and not previous_pinch
        previous_pinch = pinching

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_r):
                    if mode != "playing" or event.key == pygame.K_r:
                        game = reset_game()
                        mode = "playing"

        if pinch_started and mode in ("menu", "game_over"):
            game = reset_game()
            mode = "playing"

        keys = pygame.key.get_pressed()

        if mode == "playing":
            game["elapsed"] += delta_time
            game["hit_timer"] = max(0, game["hit_timer"] - delta_time)

            if hand_found and finger_x is not None and finger_y is not None:
                target_x = 40 + finger_x * (WIDTH - 80)
                target_y = 90 + finger_y * (HEIGHT - 170)

                game["player_x"] += (
                    target_x - game["player_x"]
                ) * min(1, 12 * delta_time)

                game["player_y"] += (
                    target_y - game["player_y"]
                ) * min(1, 12 * delta_time)

            move_player_with_keyboard(game, keys, delta_time)

            shield_active = pinching and game["shield"] > 0

            if shield_active:
                game["shield"] = max(
                    0,
                    game["shield"] - 55 * delta_time,
                )
            else:
                game["shield"] = min(
                    100,
                    game["shield"] + 24 * delta_time,
                )

            extra_speed = min(
                300,
                game["score"] * 6 + game["elapsed"] * 2,
            )
            spawn_delay = max(0.34, 0.9 - game["score"] * 0.01)

            game["spawn_timer"] += delta_time

            if game["spawn_timer"] >= spawn_delay:
                game["spawn_timer"] = 0

                # Increased from 22% to 32%, so collectible stars appear more often.
                kind = "star" if random.random() < 0.32 else "meteor"
                game["objects"].append(FallingObject(kind, extra_speed))

            player_rect = pygame.Rect(
                int(game["player_x"] - PLAYER_SIZE / 2),
                int(game["player_y"] - PLAYER_SIZE / 2),
                PLAYER_SIZE,
                PLAYER_SIZE,
            )

            for item in game["objects"][:]:
                item.update(delta_time)

                if item.y > HEIGHT:
                    game["objects"].remove(item)
                    continue

                if not player_rect.colliderect(item.rect):
                    continue

                game["objects"].remove(item)

                if item.kind == "star":
                    game["score"] += 5
                elif shield_active:
                    game["score"] += 1
                elif game["hit_timer"] <= 0:
                    game["lives"] -= 1
                    game["hit_timer"] = 1.2

                    if game["lives"] <= 0:
                        mode = "game_over"
                        high_score = max(high_score, game["score"])
                        write_high_score(high_score)

        draw_background(screen, game["elapsed"])

        for item in game["objects"]:
            item.draw(screen)

        shield_active = (
            mode == "playing"
            and pinching
            and game["shield"] > 0
        )

        draw_player(
            screen,
            game["player_x"],
            game["player_y"],
            shield_active,
            game["hit_timer"] > 0,
        )

        draw_hud(
            screen,
            game,
            high_score,
            (big_font, normal_font, small_font),
        )

        draw_camera_panel(
            screen,
            tracker,
            hand_found,
            normal_font,
            small_font,
        )

        controls = (
            "Index finger: move   Pinch: shield   "
            "WASD: backup   Esc: quit"
        )
        screen.blit(
            small_font.render(controls, True, GREY),
            (22, HEIGHT - 32),
        )

        if mode == "menu":
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((5, 8, 16, 215))
            screen.blit(overlay, (0, 0))

            draw_centered_text(
                screen,
                "FINGER FORCE",
                title_font,
                CYAN,
                220,
            )
            draw_centered_text(
                screen,
                "Move your index finger to control the ship",
                normal_font,
                WHITE,
                300,
            )
            draw_centered_text(
                screen,
                "Pinch to turn on your shield",
                normal_font,
                WHITE,
                340,
            )
            draw_centered_text(
                screen,
                "Collect stars and avoid meteors",
                normal_font,
                GOLD,
                380,
            )
            draw_centered_text(
                screen,
                "Pinch once or press SPACE to start",
                big_font,
                GREEN,
                460,
            )

        elif mode == "game_over":
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((8, 4, 12, 220))
            screen.blit(overlay, (0, 0))

            draw_centered_text(
                screen,
                "GAME OVER",
                title_font,
                RED,
                245,
            )
            draw_centered_text(
                screen,
                f"Score: {game['score']}",
                big_font,
                WHITE,
                330,
            )
            draw_centered_text(
                screen,
                f"High score: {high_score}",
                normal_font,
                GOLD,
                375,
            )
            draw_centered_text(
                screen,
                "Pinch or press SPACE to restart",
                normal_font,
                GREEN,
                445,
            )

        pygame.display.flip()

    tracker.close()
    pygame.quit()


if __name__ == "__main__":
    main()