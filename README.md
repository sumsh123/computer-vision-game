# Finger Force

Finger Force is a computer vision game controlled through hand gestures.

The player moves a spaceship by moving their index finger in front of the webcam. Bringing the thumb and index finger together activates a temporary shield.

The project combines real-time hand tracking with game development using Python, OpenCV, MediaPipe, NumPy, and Pygame.

## Features

* Real-time hand tracking
* Index-finger character movement
* Pinch gesture recognition
* Temporary shield system
* Black webcam preview with hand landmarks only
* Falling meteors and collectible stars
* Lives and scoring system
* High-score saving
* Increasing game difficulty
* Movement smoothing
* Keyboard controls as a backup
* Automatic MediaPipe model download

## Technologies Used

* Python 3.12
* OpenCV
* MediaPipe
* NumPy
* Pygame

## How It Works

MediaPipe detects 21 landmarks on the player’s hand.

The game mainly uses:

* Landmark 8 for the index fingertip
* Landmark 4 for the thumb tip

The index-finger coordinates are converted from webcam coordinates into Pygame screen coordinates.

The distance between the index fingertip and thumb tip is calculated to detect a pinch gesture. When the distance becomes small enough, the shield is activated.

The real webcam frame is used for hand detection, but it is not displayed inside the game. The visible camera panel uses a black background and only shows the detected hand joints and connections.

## Project Structure

```text
finger_force_game/
│
├── models/
│   └── hand_landmarker.task
│
├── game.py
├── download_model.py
├── diagnose.py
├── requirements.txt
├── highscore.txt
└── README.md
```

### `game.py`

Contains the main game, including:

* Webcam processing
* Hand tracking
* Gesture detection
* Player movement
* Meteors and stars
* Collision detection
* Shield energy
* Score and lives
* Menus and game-over screen

### `download_model.py`

Downloads the MediaPipe Hand Landmarker model required for hand detection.

### `diagnose.py`

Checks whether Python and all required libraries are correctly installed.

### `requirements.txt`

Contains the Python packages required to run the project.

### `models/hand_landmarker.task`

The trained MediaPipe model used to detect hand landmarks.

### `highscore.txt`

Stores the highest score achieved by the player.

## Installation

### 1. Clone or download the project

Open the project folder in Visual Studio Code.

### 2. Create a Python 3.12 virtual environment

```powershell
py -3.12 -m venv .venv
```

### 3. Install the required libraries

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Download the hand-tracking model

```powershell
.\.venv\Scripts\python.exe download_model.py
```

### 5. Check the setup

```powershell
.\.venv\Scripts\python.exe diagnose.py
```

The final output should say:

```text
SETUP PASSED. The game is ready.
```

### 6. Run the game

```powershell
.\.venv\Scripts\python.exe game.py
```

## Controls

| Action              | Control                           |
| ------------------- | --------------------------------- |
| Move the spaceship  | Move your index finger            |
| Activate the shield | Pinch your thumb and index finger |
| Start the game      | Pinch or press Space              |
| Restart the game    | Pinch, Space, Enter, or R         |
| Move without webcam | WASD or arrow keys                |
| Exit the game       | Escape                            |

## Game Rules

* Avoid falling meteors.
* Collect stars to earn points.
* Each collected star gives five points.
* Destroying a meteor with the shield gives one point.
* Hitting a meteor without the shield removes one life.
* The player starts with three lives.
* Meteors become faster as the score and playing time increase.
* The game ends when all lives are lost.

## Skills Implemented and Learned

This project helped me practise:

* Real-time computer vision
* Hand landmark detection
* Gesture recognition
* Pinch-distance calculation
* Webcam frame processing
* Coordinate mapping
* Movement smoothing
* Collision detection
* Pygame game loops
* Game-state management
* Score and high-score systems
* Difficulty scaling
* Python virtual environments
* Package and dependency management
* Debugging computer vision and game logic together

## Privacy-Friendly Camera Preview

The game does not display the normal webcam image.

OpenCV and MediaPipe still process the real frame to detect the hand, but the visible preview contains only:

* A black background
* Hand landmark points
* Lines connecting the joints
* Highlighted thumb and index fingertips
* Gesture status text

## Possible Future Improvements

* Add different levels
* Add sound effects and background music
* Add more gesture-controlled abilities
* Add a pause gesture
* Add different enemy types
* Add power-ups
* Add a leaderboard
* Add character selection
* Improve the interface and animations
* Add two-hand controls

## Author

Created as a Python computer vision and game development project.
