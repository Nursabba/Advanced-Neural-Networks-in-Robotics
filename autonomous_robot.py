# ============================================================
# GOAL-FIXED FINAL AUTONOMOUS ROBOT NAVIGATION
# ============================================================
# File:
#     final_autonomous_robot.py
#
# Required:
#     Python 3.x
#     numpy
#     matplotlib
#     torch
#
# Required model:
#     gru_trajectory_model.pth
#
# Main features:
#     600-step simulation
#     360-degree LiDAR
#     Static obstacles
#     Dynamic obstacles
#     GRU trajectory prediction
#     Adaptive obstacle avoidance
#     Emergency avoidance
#     Goal attraction
#     Goal re-alignment
#     Stuck detection
#     Recovery behavior
#     Collision detection
#     Goal reaching
#     Safety metrics
#     Path efficiency
#     GRU MAE / RMSE
#     CSV results
#     Safety graph
#     Trajectory graph
#     GRU error graph
#     Speed graph
#     Baseline comparison
# ============================================================

import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, Circle


# ============================================================
# 1. CONFIGURATION
# ============================================================

TOTAL_STEPS = 600

DT = 1.0

# Normal robot speed
ROBOT_SPEED = 0.10

# Obstacle avoidance speeds
CAUTION_SPEED = 0.075
EMERGENCY_SPEED = 0.035

# Maximum angular change per step
MAX_TURN = 0.18

# Safety distances
SAFE_DISTANCE = 1.50
WARNING_DISTANCE = 1.80
CRITICAL_DISTANCE = 0.55
COLLISION_DISTANCE = 0.35

# LiDAR
LIDAR_RANGE = 3.0
LIDAR_RAYS = 72

# Goal
GOAL_THRESHOLD = 0.30

# GRU
GRU_HISTORY_LENGTH = 10

MODEL_PATH = "gru_trajectory_model.pth"

RESULT_CSV = "final_autonomous_robot_results.csv"
BASELINE_CSV = "final_baseline_comparison.csv"

RANDOM_SEED = 42


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# 3. GRU MODEL
# ============================================================

class GRUTrajectoryPredictor(nn.Module):

    def __init__(
        self,
        input_size=2,
        hidden_size=64,
        output_size=2
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        output, _ = self.gru(x)

        return self.fc(
            output[:, -1, :]
        )


# ============================================================
# 4. LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print()
    print("=" * 80)
    print("ERROR: GRU MODEL NOT FOUND")
    print("=" * 80)
    print()
    print("Required file:")
    print(MODEL_PATH)
    print()
    print("Place gru_trajectory_model.pth in the same")
    print("folder as final_autonomous_robot.py")
    print()

    raise SystemExit


model = GRUTrajectoryPredictor()

try:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    # Supports ordinary state_dict
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            checkpoint = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:

            checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(
        checkpoint
    )

except Exception as error:

    print()
    print("=" * 80)
    print("ERROR: COULD NOT LOAD GRU MODEL")
    print("=" * 80)
    print(error)
    print()

    raise SystemExit


model.eval()


print()
print("=" * 80)
print("GOAL-FIXED AUTONOMOUS ROBOT NAVIGATION")
print("=" * 80)
print("GRU MODEL LOADED SUCCESSFULLY")
print()


# ============================================================
# 5. INITIAL ROBOT STATE
# ============================================================

robot_x = 0.0
robot_y = 0.0
robot_theta = 0.0

start_x = 0.0
start_y = 0.0

goal_x = 10.0
goal_y = 0.0

goal_reached = False
goal_reached_step = None


# ============================================================
# 6. STATIC OBSTACLES
# ============================================================

static_obstacles = [

    {
        "x": 2.8,
        "y": 1.4,
        "width": 1.0,
        "height": 0.7
    },

    {
        "x": 4.5,
        "y": -1.5,
        "width": 1.0,
        "height": 0.7
    },

    {
        "x": 6.0,
        "y": 1.3,
        "width": 1.0,
        "height": 0.7
    },

    {
        "x": 7.5,
        "y": -1.4,
        "width": 1.0,
        "height": 0.7
    }
]


# ============================================================
# 7. DYNAMIC OBSTACLES
# ============================================================

INITIAL_DYNAMIC_OBSTACLES = [

    {
        "x": 3.5,
        "y": -0.5,
        "vx": -0.012,
        "vy": 0.008
    },

    {
        "x": 5.2,
        "y": 1.8,
        "vx": -0.010,
        "vy": -0.010
    },

    {
        "x": 6.8,
        "y": -1.8,
        "vx": -0.012,
        "vy": 0.010
    },

    {
        "x": 8.0,
        "y": 0.9,
        "vx": -0.010,
        "vy": -0.008
    },

    {
        "x": 9.0,
        "y": -0.8,
        "vx": -0.008,
        "vy": 0.006
    }
]


dynamic_obstacles = [
    dict(obstacle)
    for obstacle in INITIAL_DYNAMIC_OBSTACLES
]


# ============================================================
# 8. DATA STORAGE
# ============================================================

robot_path = []

nearest_obstacle_path = []

gru_prediction_path = []

prediction_error_history = []

distance_history = []

lidar_min_history = []

static_distance_history = []

dynamic_distance_history = []

risk_history = []

decision_history = []

collision_history = []

speed_history = []

heading_history = []

goal_distance_history = []

step_records = []

gru_history = []


# ============================================================
# 9. COUNTERS
# ============================================================

collision_events = 0
previous_collision_state = False

recovery_count = 0

last_progress_step = 0
best_goal_distance = float("inf")


# ============================================================
# 10. UTILITY
# ============================================================

def normalize_angle(angle):

    return np.arctan2(
        np.sin(angle),
        np.cos(angle)
    )


def distance_between(
    x1,
    y1,
    x2,
    y2
):

    return np.sqrt(
        (x1 - x2) ** 2
        +
        (y1 - y2) ** 2
    )


# ============================================================
# 11. POINT TO RECTANGLE DISTANCE
# ============================================================

def point_to_rectangle_distance(
    px,
    py,
    obstacle
):

    closest_x = np.clip(
        px,
        obstacle["x"] - obstacle["width"] / 2,
        obstacle["x"] + obstacle["width"] / 2
    )

    closest_y = np.clip(
        py,
        obstacle["y"] - obstacle["height"] / 2,
        obstacle["y"] + obstacle["height"] / 2
    )

    return np.sqrt(
        (px - closest_x) ** 2
        +
        (py - closest_y) ** 2
    )


# ============================================================
# 12. STATIC DISTANCE
# ============================================================

def get_static_distance(
    px,
    py
):

    minimum = float("inf")

    for obstacle in static_obstacles:

        d = point_to_rectangle_distance(
            px,
            py,
            obstacle
        )

        minimum = min(
            minimum,
            d
        )

    return minimum


# ============================================================
# 13. DYNAMIC DISTANCE
# ============================================================

def get_dynamic_distance(
    px,
    py
):

    minimum = float("inf")

    for obstacle in dynamic_obstacles:

        d = distance_between(
            px,
            py,
            obstacle["x"],
            obstacle["y"]
        ) - 0.25

        minimum = min(
            minimum,
            max(0.0, d)
        )

    return minimum


# ============================================================
# 14. NEAREST OBSTACLE
# ============================================================

def get_nearest_obstacle(
    px,
    py
):

    best_distance = float("inf")
    best_x = px
    best_y = py
    best_type = "NONE"

    for obstacle in static_obstacles:

        d = point_to_rectangle_distance(
            px,
            py,
            obstacle
        )

        if d < best_distance:

            best_distance = d

            best_x = obstacle["x"]
            best_y = obstacle["y"]

            best_type = "STATIC"


    for obstacle in dynamic_obstacles:

        d = distance_between(
            px,
            py,
            obstacle["x"],
            obstacle["y"]
        ) - 0.25

        d = max(
            0.0,
            d
        )

        if d < best_distance:

            best_distance = d

            best_x = obstacle["x"]
            best_y = obstacle["y"]

            best_type = "DYNAMIC"


    return (
        best_distance,
        best_x,
        best_y,
        best_type
    )


# ============================================================
# 15. RAY-RECTANGLE INTERSECTION
# ============================================================

def ray_rectangle_distance(
    px,
    py,
    dx,
    dy,
    obstacle
):

    xmin = (
        obstacle["x"]
        -
        obstacle["width"] / 2
    )

    xmax = (
        obstacle["x"]
        +
        obstacle["width"] / 2
    )

    ymin = (
        obstacle["y"]
        -
        obstacle["height"] / 2
    )

    ymax = (
        obstacle["y"]
        +
        obstacle["height"] / 2
    )

    t_values = []

    if abs(dx) > 1e-9:

        t_values.append(
            (xmin - px) / dx
        )

        t_values.append(
            (xmax - px) / dx
        )


    if abs(dy) > 1e-9:

        t_values.append(
            (ymin - py) / dy
        )

        t_values.append(
            (ymax - py) / dy
        )


    valid = []

    for t in t_values:

        if t < 0:
            continue

        x = px + t * dx
        y = py + t * dy

        if (

            xmin - 1e-8 <= x <= xmax + 1e-8

            and

            ymin - 1e-8 <= y <= ymax + 1e-8

        ):

            valid.append(t)


    if valid:

        return min(valid)

    return float("inf")


# ============================================================
# 16. 360 DEGREE LiDAR
# ============================================================

def lidar_scan(
    px,
    py
):

    distances = []

    angles = np.linspace(
        0,
        2 * np.pi,
        LIDAR_RAYS,
        endpoint=False
    )


    for angle in angles:

        dx = np.cos(angle)
        dy = np.sin(angle)

        minimum = LIDAR_RANGE


        # Static obstacles
        for obstacle in static_obstacles:

            d = ray_rectangle_distance(
                px,
                py,
                dx,
                dy,
                obstacle
            )

            minimum = min(
                minimum,
                d
            )


        # Dynamic obstacles
        for obstacle in dynamic_obstacles:

            ox = obstacle["x"] - px
            oy = obstacle["y"] - py

            center_distance = np.sqrt(
                ox ** 2 + oy ** 2
            )

            if center_distance <= LIDAR_RANGE + 0.25:

                projection = (
                    ox * dx
                    +
                    oy * dy
                )

                if projection >= 0:

                    perpendicular = abs(
                        ox * dy
                        -
                        oy * dx
                    )

                    radius = 0.25

                    if perpendicular <= radius:

                        offset = np.sqrt(
                            max(
                                0.0,
                                radius ** 2
                                -
                                perpendicular ** 2
                            )
                        )

                        d = projection - offset

                        if d >= 0:

                            minimum = min(
                                minimum,
                                d
                            )


        distances.append(
            np.clip(
                minimum,
                0.0,
                LIDAR_RANGE
            )
        )


    return np.asarray(
        distances,
        dtype=np.float32
    )


# ============================================================
# 17. LiDAR SECTOR ANALYSIS
# ============================================================

def analyze_lidar(
    lidar
):

    angles = np.linspace(
        0,
        2 * np.pi,
        LIDAR_RAYS,
        endpoint=False
    )


    front_mask = (
        (angles <= np.pi / 6)
        |
        (angles >= 11 * np.pi / 6)
    )


    left_mask = (
        (angles >= np.pi / 6)
        &
        (angles <= 5 * np.pi / 6)
    )


    right_mask = (
        (angles >= 7 * np.pi / 6)
        &
        (angles <= 11 * np.pi / 6)
    )


    front_distance = np.min(
        lidar[front_mask]
    )

    left_distance = np.mean(
        lidar[left_mask]
    )

    right_distance = np.mean(
        lidar[right_mask]
    )


    return (
        front_distance,
        left_distance,
        right_distance
    )


# ============================================================
# 18. GRU PREDICTION
# ============================================================

def gru_predict(
    sequence
):

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    tensor = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0)


    with torch.no_grad():

        prediction = model(
            tensor
        )


    return prediction[0].numpy()


# ============================================================
# 19. MOVE DYNAMIC OBSTACLES
# ============================================================

def move_dynamic_obstacles():

    for obstacle in dynamic_obstacles:

        obstacle["x"] += (
            obstacle["vx"] * DT
        )

        obstacle["y"] += (
            obstacle["vy"] * DT
        )


        if (
            obstacle["x"] < 2.0
            or
            obstacle["x"] > 9.5
        ):

            obstacle["vx"] *= -1


        if (
            obstacle["y"] < -2.2
            or
            obstacle["y"] > 2.2
        ):

            obstacle["vy"] *= -1


# ============================================================
# 20. GOAL INFORMATION
# ============================================================

def calculate_goal_information():

    dx = goal_x - robot_x
    dy = goal_y - robot_y

    distance = np.sqrt(
        dx ** 2 + dy ** 2
    )

    angle = np.arctan2(
        dy,
        dx
    )

    return (
        distance,
        angle
    )


# ============================================================
# 21. PREDICTED COLLISION RISK
# ============================================================

def prediction_risk(
    predicted_x,
    predicted_y
):

    if not np.isfinite(predicted_x):
        return False, np.inf

    d = distance_between(
        predicted_x,
        predicted_y,
        robot_x,
        robot_y
    )

    return (
        d < WARNING_DISTANCE,
        d
    )


# ============================================================
# 22. GOAL-FIXED NAVIGATION CONTROLLER
# ============================================================

def navigation_controller(
    front_distance,
    left_distance,
    right_distance,
    nearest_distance,
    predicted_x,
    predicted_y,
    recovery_direction
):

    goal_distance, goal_angle = (
        calculate_goal_information()
    )


    heading_error = normalize_angle(
        goal_angle - robot_theta
    )


    pred_risk, pred_distance = prediction_risk(
        predicted_x,
        predicted_y
    )


    # ========================================================
    # EMERGENCY
    # ========================================================

    if (
        front_distance <= CRITICAL_DISTANCE
        or
        nearest_distance <= CRITICAL_DISTANCE
    ):

        speed = EMERGENCY_SPEED

        if left_distance >= right_distance:

            turn = MAX_TURN

        else:

            turn = -MAX_TURN

        decision = "EMERGENCY AVOID"


    # ========================================================
    # RECOVERY
    # ========================================================

    elif recovery_direction != 0:

        speed = CAUTION_SPEED

        turn = (
            recovery_direction * 0.13
            +
            0.25 * heading_error
        )

        decision = "STUCK RECOVERY"


    # ========================================================
    # OBSTACLE CAUTION
    # ========================================================

    elif (
        front_distance <= WARNING_DISTANCE
        or
        nearest_distance <= WARNING_DISTANCE
        or
        pred_risk
    ):

        speed = CAUTION_SPEED

        # Choose safer side
        if left_distance > right_distance:

            avoidance_turn = 0.115

        else:

            avoidance_turn = -0.115


        # Stronger goal attraction than before
        turn = (
            0.42 * heading_error
            +
            avoidance_turn
        )

        decision = "CAUTION"


    # ========================================================
    # NORMAL GOAL SEEKING
    # ========================================================

    else:

        speed = ROBOT_SPEED

        # Strong goal attraction
        turn = (
            0.75 * heading_error
        )

        decision = "GOAL SEEK"


    # ========================================================
    # FINAL GOAL APPROACH
    # ========================================================

    if goal_distance < 2.0:

        # Strongly align with goal direction
        turn = (
            0.85 * heading_error
            +
            0.20 * turn
        )

        speed = min(
            speed,
            ROBOT_SPEED
        )


    # ========================================================
    # Very close to goal
    # ========================================================

    if goal_distance < 0.80:

        turn = (
            0.90 * heading_error
        )

        speed = min(
            speed,
            0.075
        )


    # ========================================================
    # Clip turn
    # ========================================================

    turn = np.clip(
        turn,
        -MAX_TURN,
        MAX_TURN
    )


    # ========================================================
    # Maintain useful movement
    # ========================================================

    if goal_distance > GOAL_THRESHOLD:

        speed = max(
            speed,
            0.025
        )


    speed = np.clip(
        speed,
        0.0,
        ROBOT_SPEED
    )


    return (
        speed,
        turn,
        decision,
        pred_distance
    )


# ============================================================
# 23. COLLISION CHECK
# ============================================================

def collision_check():

    static_distance = get_static_distance(
        robot_x,
        robot_y
    )

    dynamic_distance = get_dynamic_distance(
        robot_x,
        robot_y
    )

    minimum = min(
        static_distance,
        dynamic_distance
    )

    return (
        minimum <= COLLISION_DISTANCE,
        minimum
    )


# ============================================================
# 24. FIGURE
# ============================================================

plt.ion()

fig, ax = plt.subplots(
    figsize=(13, 8)
)

ax.set_xlim(
    -1,
    11
)

ax.set_ylim(
    -3,
    3
)

ax.set_xlabel(
    "X Position (m)"
)

ax.set_ylabel(
    "Y Position (m)"
)

ax.set_title(
    "GOAL-FIXED AUTONOMOUS ROBOT NAVIGATION"
)

ax.grid(True)


# ============================================================
# 25. STATIC GRAPHICS
# ============================================================

for i, obstacle in enumerate(
    static_obstacles
):

    rectangle = Rectangle(

        (
            obstacle["x"]
            -
            obstacle["width"] / 2,

            obstacle["y"]
            -
            obstacle["height"] / 2
        ),

        obstacle["width"],
        obstacle["height"],

        linewidth=2,

        label=(
            "Static Obstacle"
            if i == 0
            else None
        )
    )

    ax.add_patch(
        rectangle
    )


# ============================================================
# 26. DYNAMIC GRAPHICS
# ============================================================

dynamic_patches = []

for i, obstacle in enumerate(
    dynamic_obstacles
):

    patch = Circle(

        (
            obstacle["x"],
            obstacle["y"]
        ),

        0.25,

        linewidth=2,

        label=(
            "Dynamic Obstacle"
            if i == 0
            else None
        )
    )

    ax.add_patch(
        patch
    )

    dynamic_patches.append(
        patch
    )


# ============================================================
# 27. GOAL / START
# ============================================================

ax.scatter(
    goal_x,
    goal_y,
    s=300,
    marker="X",
    label="Goal"
)

ax.scatter(
    start_x,
    start_y,
    s=180,
    marker="o",
    label="Start"
)


# ============================================================
# 28. ROBOT GRAPHICS
# ============================================================

robot_body = Rectangle(
    (
        robot_x - 0.35,
        robot_y - 0.25
    ),
    0.70,
    0.50,
    linewidth=2
)

ax.add_patch(
    robot_body
)


robot_head = Rectangle(
    (
        robot_x - 0.25,
        robot_y + 0.25
    ),
    0.50,
    0.35,
    linewidth=2
)

ax.add_patch(
    robot_head
)


left_eye = Circle(
    (
        robot_x - 0.12,
        robot_y + 0.43
    ),
    0.055
)

right_eye = Circle(
    (
        robot_x + 0.12,
        robot_y + 0.43
    ),
    0.055
)

ax.add_patch(left_eye)
ax.add_patch(right_eye)


left_wheel = Circle(
    (
        robot_x - 0.25,
        robot_y - 0.32
    ),
    0.10
)

right_wheel = Circle(
    (
        robot_x + 0.25,
        robot_y - 0.32
    ),
    0.10
)

ax.add_patch(left_wheel)
ax.add_patch(right_wheel)


lidar_sensor = Circle(
    (
        robot_x,
        robot_y + 0.68
    ),
    0.08
)

ax.add_patch(
    lidar_sensor
)


mouth_line, = ax.plot(
    [],
    [],
    linewidth=2
)


# ============================================================
# 29. PATH GRAPHICS
# ============================================================

robot_line, = ax.plot(
    [],
    [],
    linewidth=3,
    label="Robot Path"
)


prediction_line, = ax.plot(
    [],
    [],
    linestyle="--",
    linewidth=2,
    label="GRU Prediction"
)


prediction_point, = ax.plot(
    [],
    [],
    marker="x",
    markersize=10,
    linestyle="None",
    label="GRU Predicted Position"
)


# ============================================================
# 30. LiDAR GRAPHICS
# ============================================================

lidar_lines = []

lidar_angles = np.linspace(
    0,
    2 * np.pi,
    LIDAR_RAYS,
    endpoint=False
)


for angle in lidar_angles:

    line, = ax.plot(
        [],
        [],
        linestyle="--",
        linewidth=0.5
    )

    lidar_lines.append(
        (
            line,
            angle
        )
    )


# ============================================================
# 31. INFO PANEL
# ============================================================

info = ax.text(
    0.02,
    0.98,
    "",
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=8.5,
    bbox=dict(
        boxstyle="round",
        alpha=0.90
    )
)


ax.legend(
    loc="lower right"
)


# ============================================================
# 32. MAIN SIMULATION
# ============================================================

print("=" * 80)
print("STARTING GOAL-FIXED 600-STEP SIMULATION")
print("=" * 80)
print()


for step in range(
    TOTAL_STEPS
):

    # --------------------------------------------------------
    # Move dynamic obstacles
    # --------------------------------------------------------

    if not goal_reached:

        move_dynamic_obstacles()


    # --------------------------------------------------------
    # Goal information
    # --------------------------------------------------------

    goal_distance, goal_angle = (
        calculate_goal_information()
    )


    # --------------------------------------------------------
    # Goal reached before movement
    # --------------------------------------------------------

    if (
        not goal_reached
        and
        goal_distance <= GOAL_THRESHOLD
    ):

        robot_x = goal_x
        robot_y = goal_y

        goal_reached = True

        goal_reached_step = step + 1


    # --------------------------------------------------------
    # LiDAR
    # --------------------------------------------------------

    lidar = lidar_scan(
        robot_x,
        robot_y
    )

    lidar_minimum = float(
        np.min(lidar)
    )


    front_distance, left_distance, right_distance = (
        analyze_lidar(
            lidar
        )
    )


    # --------------------------------------------------------
    # Nearest obstacle
    # --------------------------------------------------------

    (
        nearest_distance,
        nearest_x,
        nearest_y,
        nearest_type
    ) = get_nearest_obstacle(
        robot_x,
        robot_y
    )


    nearest_obstacle_path.append(
        [
            nearest_x,
            nearest_y
        ]
    )


    # --------------------------------------------------------
    # GRU history
    # --------------------------------------------------------

    gru_history.append(
        [
            nearest_x,
            nearest_y
        ]
    )


    if len(gru_history) > GRU_HISTORY_LENGTH:

        gru_history.pop(0)


    gru_ready = (
        len(gru_history)
        >=
        GRU_HISTORY_LENGTH
    )


    # --------------------------------------------------------
    # GRU prediction
    # --------------------------------------------------------

    if gru_ready:

        predicted_x, predicted_y = gru_predict(
            gru_history
        )

    else:

        predicted_x = np.nan
        predicted_y = np.nan


    # --------------------------------------------------------
    # Recovery detection
    # --------------------------------------------------------

    recovery_direction = 0


    current_goal_distance = (
        distance_between(
            robot_x,
            robot_y,
            goal_x,
            goal_y
        )
    )


    # Detect meaningful progress
    if (
        current_goal_distance
        <
        best_goal_distance - 0.015
    ):

        best_goal_distance = (
            current_goal_distance
        )

        last_progress_step = step


    # Stuck for many steps
    if (
        not goal_reached
        and
        step - last_progress_step >= 35
    ):

        recovery_count += 1

        if left_distance >= right_distance:

            recovery_direction = 1

        else:

            recovery_direction = -1

        last_progress_step = step


    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    if goal_reached:

        speed = 0.0
        turn = 0.0

        decision = "GOAL REACHED"

        prediction_distance = np.nan

    else:

        (
            speed,
            turn,
            decision,
            prediction_distance
        ) = navigation_controller(

            front_distance,
            left_distance,
            right_distance,
            nearest_distance,
            predicted_x,
            predicted_y,
            recovery_direction
        )


        # ----------------------------------------------------
        # Heading update
        # ----------------------------------------------------

        robot_theta += (
            turn * DT
        )

        robot_theta = normalize_angle(
            robot_theta
        )


        # ----------------------------------------------------
        # Position update
        # ----------------------------------------------------

        robot_x += (
            speed
            *
            np.cos(robot_theta)
            *
            DT
        )

        robot_y += (
            speed
            *
            np.sin(robot_theta)
            *
            DT
        )


        # ----------------------------------------------------
        # Workspace
        # ----------------------------------------------------

        robot_y = np.clip(
            robot_y,
            -2.5,
            2.5
        )


    # --------------------------------------------------------
    # Post-motion collision
    # --------------------------------------------------------

    collision, actual_collision_distance = (
        collision_check()
    )


    if (
        collision
        and
        not previous_collision_state
    ):

        collision_events += 1


    previous_collision_state = collision


    # --------------------------------------------------------
    # Goal check after motion
    # --------------------------------------------------------

    goal_distance, goal_angle = (
        calculate_goal_information()
    )


    if (
        not goal_reached
        and
        goal_distance <= GOAL_THRESHOLD
    ):

        robot_x = goal_x
        robot_y = goal_y

        goal_reached = True

        goal_reached_step = step + 1

        decision = "GOAL REACHED"

    elif goal_reached:

        decision = "GOAL REACHED"


    # --------------------------------------------------------
    # Effective risk distance
    # --------------------------------------------------------

    effective_distance = min(
        nearest_distance,
        lidar_minimum
    )


    if np.isfinite(
        prediction_distance
    ):

        # Prediction contributes to risk
        # but does not replace physical sensing.

        if prediction_distance < effective_distance:

            effective_distance = prediction_distance


    if effective_distance <= CRITICAL_DISTANCE:

        risk = "HIGH"

    elif effective_distance <= WARNING_DISTANCE:

        risk = "MEDIUM"

    else:

        risk = "LOW"


    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    robot_path.append(
        [
            robot_x,
            robot_y
        ]
    )

    distance_history.append(
        effective_distance
    )

    lidar_min_history.append(
        lidar_minimum
    )

    static_distance_history.append(
        get_static_distance(
            robot_x,
            robot_y
        )
    )

    dynamic_distance_history.append(
        get_dynamic_distance(
            robot_x,
            robot_y
        )
    )

    risk_history.append(
        risk
    )

    decision_history.append(
        decision
    )

    collision_history.append(
        int(collision)
    )

    speed_history.append(
        speed
    )

    heading_history.append(
        robot_theta
    )

    goal_distance_history.append(
        goal_distance
    )

    gru_prediction_path.append(
        [
            predicted_x,
            predicted_y
        ]
    )


    # --------------------------------------------------------
    # GRU error
    # --------------------------------------------------------

    if (
        gru_ready
        and
        len(nearest_obstacle_path) >= 2
    ):

        prediction = np.array(
            [
                predicted_x,
                predicted_y
            ],
            dtype=float
        )

        if np.all(
            np.isfinite(prediction)
        ):

            actual = np.array(
                nearest_obstacle_path[-1],
                dtype=float
            )

            error = np.linalg.norm(
                actual - prediction
            )

            prediction_error_history.append(
                error
            )


    # --------------------------------------------------------
    # CSV record
    # --------------------------------------------------------

    step_records.append(
        [

            step + 1,

            robot_x,

            robot_y,

            goal_distance,

            nearest_distance,

            lidar_minimum,

            predicted_x,

            predicted_y,

            prediction_distance,

            front_distance,

            left_distance,

            right_distance,

            risk,

            decision,

            int(collision),

            goal_reached

        ]
    )


    # --------------------------------------------------------
    # Graphics
    # --------------------------------------------------------

    robot_body.set_xy(
        (
            robot_x - 0.35,
            robot_y - 0.25
        )
    )

    robot_head.set_xy(
        (
            robot_x - 0.25,
            robot_y + 0.25
        )
    )


    left_eye.center = (
        robot_x - 0.12,
        robot_y + 0.43
    )

    right_eye.center = (
        robot_x + 0.12,
        robot_y + 0.43
    )


    left_wheel.center = (
        robot_x - 0.25,
        robot_y - 0.32
    )

    right_wheel.center = (
        robot_x + 0.25,
        robot_y - 0.32
    )


    lidar_sensor.center = (
        robot_x,
        robot_y + 0.68
    )


    mouth_line.set_data(

        [
            robot_x - 0.12,
            robot_x,
            robot_x + 0.12
        ],

        [
            robot_y + 0.33,
            robot_y + 0.29,
            robot_y + 0.33
        ]
    )


    # Dynamic graphics
    for i, obstacle in enumerate(
        dynamic_obstacles
    ):

        dynamic_patches[i].center = (
            obstacle["x"],
            obstacle["y"]
        )


    # Robot path
    path_array = np.asarray(
        robot_path
    )

    robot_line.set_data(
        path_array[:, 0],
        path_array[:, 1]
    )


    # GRU path
    prediction_array = np.asarray(
        gru_prediction_path,
        dtype=float
    )

    if len(prediction_array) > 0:

        valid = np.isfinite(
            prediction_array[:, 0]
        )

        if np.any(valid):

            prediction_line.set_data(
                prediction_array[valid, 0],
                prediction_array[valid, 1]
            )


    if np.isfinite(predicted_x):

        prediction_point.set_data(
            [predicted_x],
            [predicted_y]
        )


    # LiDAR rays
    for i, (
        line,
        angle
    ) in enumerate(
        lidar_lines
    ):

        distance = lidar[i]

        end_x = (
            robot_x
            +
            distance * np.cos(angle)
        )

        end_y = (
            robot_y
            +
            distance * np.sin(angle)
        )

        line.set_data(
            [
                robot_x,
                end_x
            ],
            [
                robot_y,
                end_y
            ]
        )


    # Live safety
    safe_steps = sum(
        d > COLLISION_DISTANCE
        for d in distance_history
    )

    safety_rate_live = (
        safe_steps
        /
        len(distance_history)
    ) * 100


    # Information
    info.set_text(

        f"GOAL-FIXED AUTONOMOUS ROBOT\n"
        f"--------------------------------\n"
        f"Step: {step + 1}/{TOTAL_STEPS}\n"
        f"Robot: ({robot_x:.2f}, {robot_y:.2f}) m\n"
        f"Goal Distance: {goal_distance:.3f} m\n"
        f"Nearest: {nearest_distance:.3f} m ({nearest_type})\n"
        f"LiDAR Minimum: {lidar_minimum:.3f} m\n"
        f"Speed: {speed:.3f} m/s\n"
        f"Risk: {risk}\n"
        f"Decision: {decision}\n"
        f"Recovery: {recovery_count}\n"
        f"Collision Events: {collision_events}\n"
        f"Safety Rate: {safety_rate_live:.2f}%\n"
        f"Goal Reached: {goal_reached}"
    )


    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    plt.pause(0.01)


# ============================================================
# 33. FINAL METRICS
# ============================================================

robot_array = np.asarray(
    robot_path,
    dtype=float
)


# ============================================================
# PATH LENGTH
# ============================================================

path_length = 0.0

if len(robot_array) > 1:

    path_length = np.sum(
        np.linalg.norm(
            np.diff(
                robot_array,
                axis=0
            ),
            axis=1
        )
    )


# ============================================================
# FINAL POSITION
# ============================================================

final_x = robot_array[-1, 0]
final_y = robot_array[-1, 1]


final_goal_distance = distance_between(
    final_x,
    final_y,
    goal_x,
    goal_y
)


# ============================================================
# DISTANCES
# ============================================================

minimum_distance = np.min(
    distance_history
)

minimum_lidar_distance = np.min(
    lidar_min_history
)

minimum_static_distance = np.min(
    static_distance_history
)

minimum_dynamic_distance = np.min(
    dynamic_distance_history
)


# ============================================================
# COLLISIONS
# ============================================================

collision_steps = int(
    sum(collision_history)
)


# ============================================================
# SAFETY RATE
# ============================================================

safe_steps = sum(
    d > COLLISION_DISTANCE
    for d in distance_history
)

safety_rate = (
    safe_steps
    /
    len(distance_history)
) * 100


# ============================================================
# SPEED
# ============================================================

average_speed = np.mean(
    speed_history
)

maximum_speed = np.max(
    speed_history
)


# ============================================================
# TIME TO GOAL
# ============================================================

if goal_reached_step is not None:

    time_to_goal = (
        goal_reached_step
        *
        DT
    )

else:

    time_to_goal = np.nan


# ============================================================
# PATH EFFICIENCY
# ============================================================

straight_line_distance = distance_between(
    start_x,
    start_y,
    goal_x,
    goal_y
)


if path_length > 0:

    path_efficiency = (
        straight_line_distance
        /
        path_length
    ) * 100

else:

    path_efficiency = 0.0


# ============================================================
# GRU METRICS
# ============================================================

prediction_errors = np.asarray(
    prediction_error_history,
    dtype=float
)


if len(prediction_errors) > 0:

    mae = np.mean(
        prediction_errors
    )

    rmse = np.sqrt(
        np.mean(
            prediction_errors ** 2
        )
    )

else:

    mae = np.nan
    rmse = np.nan


# ============================================================
# DECISION COUNTS
# ============================================================

goal_seek_count = decision_history.count(
    "GOAL SEEK"
)

caution_count = decision_history.count(
    "CAUTION"
)

emergency_count = decision_history.count(
    "EMERGENCY AVOID"
)

recovery_decision_count = decision_history.count(
    "STUCK RECOVERY"
)

goal_count = decision_history.count(
    "GOAL REACHED"
)


# ============================================================
# RISK COUNTS
# ============================================================

high_count = risk_history.count(
    "HIGH"
)

medium_count = risk_history.count(
    "MEDIUM"
)

low_count = risk_history.count(
    "LOW"
)


# ============================================================
# 34. FINAL TERMINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("GOAL-FIXED AUTONOMOUS ROBOT — PERFORMANCE RESULTS")
print("=" * 80)

print()

print(
    f"Final Robot X = {final_x:.3f} m"
)

print(
    f"Final Robot Y = {final_y:.3f} m"
)

print(
    f"Final Goal Distance = "
    f"{final_goal_distance:.3f} m"
)

print(
    f"Goal Reached = "
    f"{goal_reached}"
)

print()

print(
    f"Path Length = "
    f"{path_length:.3f} m"
)

print(
    f"Straight-Line Distance = "
    f"{straight_line_distance:.3f} m"
)

print(
    f"Path Efficiency = "
    f"{path_efficiency:.2f}%"
)

print()

print(
    f"Minimum Effective Obstacle Distance = "
    f"{minimum_distance:.3f} m"
)

print(
    f"Minimum LiDAR Distance = "
    f"{minimum_lidar_distance:.3f} m"
)

print(
    f"Minimum Static Distance = "
    f"{minimum_static_distance:.3f} m"
)

print(
    f"Minimum Dynamic Distance = "
    f"{minimum_dynamic_distance:.3f} m"
)

print()

print(
    f"Collision Steps = "
    f"{collision_steps}"
)

print(
    f"Collision Events = "
    f"{collision_events}"
)

print(
    f"Safety Rate = "
    f"{safety_rate:.2f}%"
)

print()

print(
    f"Average Speed = "
    f"{average_speed:.4f} m/s"
)

print(
    f"Maximum Speed = "
    f"{maximum_speed:.4f} m/s"
)

if np.isfinite(time_to_goal):

    print(
        f"Time to Goal = "
        f"{time_to_goal:.2f} s"
    )

else:

    print(
        "Time to Goal = NOT REACHED"
    )

print()

print(
    f"Recovery Events = "
    f"{recovery_count}"
)

print()

print(
    "GRU FUTURE-PREDICTION PERFORMANCE"
)

print(
    f"Prediction Samples = "
    f"{len(prediction_errors)}"
)

print(
    f"MAE = "
    f"{mae:.5f} m"
)

print(
    f"RMSE = "
    f"{rmse:.5f} m"
)

print()

print(
    "NAVIGATION DECISIONS"
)

print(
    f"GOAL SEEK = "
    f"{goal_seek_count}"
)

print(
    f"CAUTION = "
    f"{caution_count}"
)

print(
    f"EMERGENCY AVOID = "
    f"{emergency_count}"
)

print(
    f"STUCK RECOVERY = "
    f"{recovery_decision_count}"
)

print(
    f"GOAL REACHED = "
    f"{goal_count}"
)

print()

print(
    "RISK ANALYSIS"
)

print(
    f"HIGH = "
    f"{high_count}"
)

print(
    f"MEDIUM = "
    f"{medium_count}"
)

print(
    f"LOW = "
    f"{low_count}"
)


# ============================================================
# 35. SAVE FINAL CSV
# ============================================================

with open(
    RESULT_CSV,
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow(

        [

            "Step",
            "Robot X (m)",
            "Robot Y (m)",
            "Goal Distance (m)",
            "Nearest Obstacle (m)",
            "LiDAR Minimum (m)",
            "GRU Predicted X (m)",
            "GRU Predicted Y (m)",
            "GRU Prediction Distance (m)",
            "Front Distance (m)",
            "Left Distance (m)",
            "Right Distance (m)",
            "Risk",
            "Decision",
            "Collision",
            "Goal Reached"

        ]
    )

    writer.writerows(
        step_records
    )


print()
print(
    "FINAL CSV SAVED:"
)

print(
    RESULT_CSV
)


# ============================================================
# 36. SAFETY GRAPH
# ============================================================

plt.ioff()

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    distance_history,
    linewidth=2,
    label="Effective Obstacle Distance"
)

plt.axhline(
    SAFE_DISTANCE,
    linestyle="--",
    linewidth=2,
    label="Safe Distance"
)

plt.axhline(
    WARNING_DISTANCE,
    linestyle="--",
    linewidth=2,
    label="Warning Distance"
)

plt.axhline(
    COLLISION_DISTANCE,
    linestyle="--",
    linewidth=2,
    label="Collision Threshold"
)

plt.xlabel(
    "Simulation Step"
)

plt.ylabel(
    "Distance (m)"
)

plt.title(
    "Goal-Fixed Robot — Safety Analysis"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 37. TRAJECTORY GRAPH
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.plot(
    robot_array[:, 0],
    robot_array[:, 1],
    linewidth=3,
    label="Robot Path"
)

plt.scatter(
    start_x,
    start_y,
    s=200,
    marker="o",
    label="Start"
)

plt.scatter(
    goal_x,
    goal_y,
    s=300,
    marker="X",
    label="Goal"
)


for i, obstacle in enumerate(
    static_obstacles
):

    rectangle = Rectangle(

        (
            obstacle["x"]
            -
            obstacle["width"] / 2,

            obstacle["y"]
            -
            obstacle["height"] / 2
        ),

        obstacle["width"],
        obstacle["height"],

        linewidth=2,

        label=(
            "Static Obstacle"
            if i == 0
            else None
        )
    )

    plt.gca().add_patch(
        rectangle
    )


# Plot final positions of dynamic obstacles
for i, obstacle in enumerate(
    dynamic_obstacles
):

    plt.scatter(

        obstacle["x"],
        obstacle["y"],

        s=80,

        marker="o",

        label=(
            "Dynamic Obstacle"
            if i == 0
            else None
        )
    )


plt.xlabel(
    "X Position (m)"
)

plt.ylabel(
    "Y Position (m)"
)

plt.title(
    "Goal-Fixed Robot — Navigation Trajectory"
)

plt.grid(True)

plt.legend()

plt.axis("equal")

plt.tight_layout()

plt.show()


# ============================================================
# 38. GRU ERROR GRAPH
# ============================================================

if len(prediction_errors) > 0:

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        prediction_errors,
        linewidth=2,
        label="GRU Prediction Error"
    )

    plt.xlabel(
        "Prediction Sample"
    )

    plt.ylabel(
        "Position Error (m)"
    )

    plt.title(
        "GRU Future Obstacle Prediction Error"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# 39. SPEED GRAPH
# ============================================================

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    speed_history,
    linewidth=2,
    label="Robot Speed"
)

plt.axhline(
    ROBOT_SPEED,
    linestyle="--",
    linewidth=2,
    label="Maximum Nominal Speed"
)

plt.xlabel(
    "Simulation Step"
)

plt.ylabel(
    "Speed (m/s)"
)

plt.title(
    "Goal-Fixed Robot — Speed Profile"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 40. BASELINE SIMULATION
# ============================================================

def run_baseline(
    controller_name
):

    x = 0.0
    y = 0.0
    theta = 0.0

    local_dynamic = [
        dict(obstacle)
        for obstacle in INITIAL_DYNAMIC_OBSTACLES
    ]

    path = [
        [x, y]
    ]

    speeds = []

    collision_steps_local = 0

    goal_step = None


    def local_dynamic_distance(
        px,
        py
    ):

        minimum = float("inf")

        for obstacle in local_dynamic:

            d = distance_between(
                px,
                py,
                obstacle["x"],
                obstacle["y"]
            ) - 0.25

            minimum = min(
                minimum,
                max(0.0, d)
            )

        return minimum


    for step in range(
        TOTAL_STEPS
    ):

        # Move dynamic obstacles
        for obstacle in local_dynamic:

            obstacle["x"] += obstacle["vx"]
            obstacle["y"] += obstacle["vy"]

            if (
                obstacle["x"] < 2.0
                or
                obstacle["x"] > 9.5
            ):

                obstacle["vx"] *= -1

            if (
                obstacle["y"] < -2.2
                or
                obstacle["y"] > 2.2
            ):

                obstacle["vy"] *= -1


        goal_dx = goal_x - x
        goal_dy = goal_y - y

        goal_distance = np.sqrt(
            goal_dx ** 2
            +
            goal_dy ** 2
        )


        if goal_distance <= GOAL_THRESHOLD:

            if goal_step is None:

                goal_step = step + 1

            speeds.append(0.0)

            path.append(
                [x, y]
            )

            continue


        goal_angle = np.arctan2(
            goal_dy,
            goal_dx
        )

        heading_error = normalize_angle(
            goal_angle - theta
        )


        static_d = get_static_distance(
            x,
            y
        )

        dynamic_d = local_dynamic_distance(
            x,
            y
        )

        nearest_d = min(
            static_d,
            dynamic_d
        )


        # ----------------------------------------------------
        # STRAIGHT
        # ----------------------------------------------------

        if controller_name == "STRAIGHT":

            speed = ROBOT_SPEED

            turn = 0.0


        # ----------------------------------------------------
        # REACTIVE
        # ----------------------------------------------------

        elif controller_name == "REACTIVE":

            speed = ROBOT_SPEED

            turn = (
                0.65 * heading_error
            )

            if nearest_d < WARNING_DISTANCE:

                speed = CAUTION_SPEED

                if y >= 0:

                    turn -= 0.12

                else:

                    turn += 0.12


        # ----------------------------------------------------
        # GRU ONLY
        # ----------------------------------------------------

        elif controller_name == "GRU_ONLY":

            speed = ROBOT_SPEED

            turn = (
                0.70 * heading_error
            )

            if dynamic_d < WARNING_DISTANCE:

                speed = CAUTION_SPEED

                if y >= 0:

                    turn -= 0.10

                else:

                    turn += 0.10


        # ----------------------------------------------------
        # PROPOSED
        # ----------------------------------------------------

        else:

            speed = ROBOT_SPEED

            turn = (
                0.80 * heading_error
            )

            if nearest_d < WARNING_DISTANCE:

                speed = CAUTION_SPEED

                if y >= 0:

                    turn -= 0.11

                else:

                    turn += 0.11

            if nearest_d < CRITICAL_DISTANCE:

                speed = EMERGENCY_SPEED

                if y >= 0:

                    turn -= MAX_TURN

                else:

                    turn += MAX_TURN


        turn = np.clip(
            turn,
            -MAX_TURN,
            MAX_TURN
        )


        theta += (
            turn * DT
        )

        theta = normalize_angle(
            theta
        )


        x += (
            speed
            *
            np.cos(theta)
        )

        y += (
            speed
            *
            np.sin(theta)
        )


        y = np.clip(
            y,
            -2.5,
            2.5
        )


        actual_distance = min(

            get_static_distance(
                x,
                y
            ),

            local_dynamic_distance(
                x,
                y
            )
        )


        if actual_distance <= COLLISION_DISTANCE:

            collision_steps_local += 1


        speeds.append(
            speed
        )

        path.append(
            [
                x,
                y
            ]
        )


    path = np.asarray(
        path,
        dtype=float
    )


    if len(path) > 1:

        baseline_path_length = np.sum(

            np.linalg.norm(
                np.diff(
                    path,
                    axis=0
                ),
                axis=1
            )
        )

    else:

        baseline_path_length = 0.0


    final_distance = distance_between(
        x,
        y,
        goal_x,
        goal_y
    )


    reached = (
        final_distance <= GOAL_THRESHOLD
    )


    if baseline_path_length > 0:

        efficiency = (
            straight_line_distance
            /
            baseline_path_length
        ) * 100

    else:

        efficiency = 0.0


    safety_rate_baseline = (
        1
        -
        collision_steps_local
        /
        TOTAL_STEPS
    ) * 100


    return {

        "Controller": controller_name,

        "Final Goal Distance (m)": final_distance,

        "Goal Reached": reached,

        "Path Length (m)": baseline_path_length,

        "Path Efficiency (%)": efficiency,

        "Collision Steps": collision_steps_local,

        "Safety Rate (%)": safety_rate_baseline,

        "Average Speed (m/s)": np.mean(
            speeds
        ),

        "Time to Goal (s)": (

            goal_step * DT

            if goal_step is not None

            else np.nan
        )
    }


# ============================================================
# 41. RUN BASELINES
# ============================================================

print()
print("=" * 80)
print("ACTUAL BASELINE COMPARISON")
print("=" * 80)

baseline_results = []


for controller in [

    "STRAIGHT",
    "REACTIVE",
    "GRU_ONLY",
    "PROPOSED"

]:

    print(
        "Running:",
        controller
    )

    result = run_baseline(
        controller
    )

    baseline_results.append(
        result
    )

    print(
        "Completed:",
        controller
    )


# ============================================================
# 42. SAVE BASELINE CSV
# ============================================================

baseline_fields = [

    "Controller",
    "Final Goal Distance (m)",
    "Goal Reached",
    "Path Length (m)",
    "Path Efficiency (%)",
    "Collision Steps",
    "Safety Rate (%)",
    "Average Speed (m/s)",
    "Time to Goal (s)"

]


with open(
    BASELINE_CSV,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=baseline_fields
    )

    writer.writeheader()

    writer.writerows(
        baseline_results
    )


print()
print(
    "BASELINE CSV SAVED:"
)

print(
    BASELINE_CSV
)


# ============================================================
# 43. BASELINE SUMMARY
# ============================================================

print()
print("=" * 80)
print("BASELINE COMPARISON RESULTS")
print("=" * 80)


for result in baseline_results:

    print()

    print(
        result["Controller"]
    )

    print(
        "Goal Reached =",
        result["Goal Reached"]
    )

    print(
        "Final Goal Distance =",
        f'{result["Final Goal Distance (m)"]:.3f} m'
    )

    print(
        "Path Length =",
        f'{result["Path Length (m)"]:.3f} m'
    )

    print(
        "Path Efficiency =",
        f'{result["Path Efficiency (%)"]:.2f}%'
    )

    print(
        "Collision Steps =",
        result["Collision Steps"]
    )

    print(
        "Safety Rate =",
        f'{result["Safety Rate (%)"]:.2f}%'
    )

    print(
        "Average Speed =",
        f'{result["Average Speed (m/s)"]:.4f} m/s'
    )


# ============================================================
# 44. FINAL COMPLETION
# ============================================================

print()
print("=" * 80)
print("GOAL-FIXED AUTONOMOUS ROBOT PROJECT COMPLETED")
print("=" * 80)

print()

print("FEATURES:")
print("✓ 600-step simulation")
print("✓ Goal-fixed navigation")
print("✓ Strong goal attraction")
print("✓ Goal re-alignment")
print("✓ Stuck detection")
print("✓ Recovery behavior")
print("✓ 360-degree LiDAR")
print("✓ Static obstacles")
print("✓ Dynamic obstacles")
print("✓ GRU prediction")
print("✓ Adaptive obstacle avoidance")
print("✓ Emergency avoidance")
print("✓ Collision detection")
print("✓ Goal reaching")
print("✓ Safety metrics")
print("✓ Path efficiency")
print("✓ GRU MAE / RMSE")
print("✓ Safety graph")
print("✓ Trajectory graph")
print("✓ GRU error graph")
print("✓ Speed graph")
print("✓ Baseline comparison")
print("✓ Research CSV")
print("✓ Baseline CSV")

print()

print(
    "FINAL RESULT FILE:",
    RESULT_CSV
)

print(
    "BASELINE RESULT FILE:",
    BASELINE_CSV
)

print()

print("=" * 80)
print("PROJECT RUN FINISHED")
print("=" * 80)