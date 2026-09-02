import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================================================
# 1. GRU MODEL
# =========================================================

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

        output, hidden = self.gru(x)

        last_output = output[:, -1, :]

        prediction = self.fc(last_output)

        return prediction


# =========================================================
# 2. LOAD TRAINED MODEL
# =========================================================

MODEL_PATH = "gru_trajectory_model.pth"

if not os.path.exists(MODEL_PATH):

    print()
    print("ERROR: GRU model file not found!")
    print()
    print("Current folder files:")

    for file in os.listdir("."):
        print(file)

    print()
    print("MODEL_PATH me apni .pth file ka exact naam likhiye.")

    raise SystemExit


model = GRUTrajectoryPredictor(
    input_size=2,
    hidden_size=64,
    output_size=2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )
)

model.eval()

print()
print("======================================")
print("TRAINED GRU MODEL LOADED SUCCESSFULLY")
print("======================================")


# =========================================================
# 3. SETTINGS
# =========================================================

SAFE_DISTANCE = 1.5
WARNING_DISTANCE = 2.5

ROBOT_STEP = 0.10

robot_x = 0.0
robot_y = 0.0

goal_x = 10.0
goal_y = 0.0


# =========================================================
# 4. MOVING OBSTACLE
# =========================================================

obstacle_x = 7.0
obstacle_y = 1.8

velocity_x = -0.05
velocity_y = -0.015


# =========================================================
# 5. HISTORY FOR GRU
# =========================================================

history = []

robot_path = []
obstacle_path = []
prediction_path = []

distance_history = []
risk_history = []
decision_history = []


# =========================================================
# 6. PREDICTION FUNCTION
# =========================================================

def gru_predict(history_data):

    sequence = np.array(
        history_data,
        dtype=np.float32
    )

    sequence = torch.tensor(
        sequence,
        dtype=torch.float32
    )

    sequence = sequence.unsqueeze(0)

    with torch.no_grad():

        prediction = model(sequence)

    prediction = prediction.squeeze(0)

    return prediction.numpy()


# =========================================================
# 7. COLLISION RISK
# =========================================================

def collision_risk(distance):

    if distance <= SAFE_DISTANCE:

        return "HIGH"

    elif distance <= WARNING_DISTANCE:

        return "MEDIUM"

    else:

        return "LOW"


# =========================================================
# 8. NAVIGATION DECISION
# =========================================================

def navigation_decision(
    robot_x,
    robot_y,
    predicted_x,
    predicted_y,
    risk
):

    dx = predicted_x - robot_x
    dy = predicted_y - robot_y

    if risk == "HIGH":

        if dy >= 0:
            return "RIGHT"

        else:
            return "LEFT"

    elif risk == "MEDIUM":

        if dy >= 0:
            return "LEFT"

        else:
            return "RIGHT"

    else:

        return "FORWARD"


# =========================================================
# 9. SIMULATION
# =========================================================
# =========================================================
# 9. SIMULATION
# =========================================================

for step in range(250):

    # ---------------------------------------------
    # Move obstacle
    # ---------------------------------------------

    obstacle_x += velocity_x
    obstacle_y += velocity_y

    # ---------------------------------------------
    # Boundary
    # ---------------------------------------------

    if obstacle_x < 3.0 or obstacle_x > 8.0:
        velocity_x *= -1

    if obstacle_y < -2.0 or obstacle_y > 2.0:
        velocity_y *= -1

    # ---------------------------------------------
    # Store LiDAR-like measurement
    # ---------------------------------------------

    history.append([
        obstacle_x,
        obstacle_y
    ])

    # Keep last 10 measurements

    if len(history) > 10:
        history.pop(0)

    # ---------------------------------------------
    # GRU prediction
    # ---------------------------------------------

    if len(history) >= 10:

        predicted_x, predicted_y = gru_predict(
            history
        )

    else:

        predicted_x = obstacle_x
        predicted_y = obstacle_y

    # ---------------------------------------------
    # Distance to predicted obstacle
    # ---------------------------------------------

    dx = predicted_x - robot_x
    dy = predicted_y - robot_y

    predicted_distance = np.sqrt(
        dx ** 2 + dy ** 2
    )

    # ---------------------------------------------
    # Collision risk
    # ---------------------------------------------

    risk = collision_risk(
        predicted_distance
    )

    # ---------------------------------------------
    # Goal direction
    # ---------------------------------------------

    goal_dx = goal_x - robot_x
    goal_dy = goal_y - robot_y

    goal_distance = np.sqrt(
        goal_dx ** 2 + goal_dy ** 2
    )

    # ---------------------------------------------
    # STEP 18 - Adaptive Navigation
    # ---------------------------------------------

    if risk == "HIGH":

        # Move around obstacle

        robot_x += ROBOT_STEP * 0.4

        if predicted_y >= robot_y:
            robot_y -= ROBOT_STEP
        else:
            robot_y += ROBOT_STEP

        decision = "AVOID"

    elif risk == "MEDIUM":

        # Continue forward but avoid obstacle

        robot_x += ROBOT_STEP * 0.7

        if predicted_y >= robot_y:
            robot_y -= ROBOT_STEP * 0.5
        else:
            robot_y += ROBOT_STEP * 0.5

        decision = "CAUTION"

    else:

        # -----------------------------------------
        # Move toward goal
        # -----------------------------------------

        angle = np.arctan2(
            goal_dy,
            goal_dx
        )

        robot_x += (
            ROBOT_STEP * np.cos(angle)
        )

        robot_y += (
            ROBOT_STEP * np.sin(angle)
        )

        decision = "FORWARD"

    # ---------------------------------------------
    # Workspace limitation
    # ---------------------------------------------

    robot_y = np.clip(
        robot_y,
        -2.5,
        2.5
    )

    # ---------------------------------------------
    # Save ALL data INSIDE LOOP
    # ---------------------------------------------

    robot_path.append([
        robot_x,
        robot_y
    ])

    obstacle_path.append([
        obstacle_x,
        obstacle_y
    ])

    prediction_path.append([
        predicted_x,
        predicted_y
    ])

    distance_history.append(
        predicted_distance
    )

    risk_history.append(
        risk
    )

    decision_history.append(
        decision
    )

    # ---------------------------------------------
    # Goal reached
    # ---------------------------------------------

    if goal_distance < 0.30:

        robot_x = goal_x
        robot_y = goal_y

        print(
            f"Goal reached at step {step}"
        )

        # Don't use break for now


# =========================================================
# 10. CONVERT DATA
# =========================================================

robot_path = np.array(
    robot_path,
    dtype=float
)

obstacle_path = np.array(
    obstacle_path,
    dtype=float
)

prediction_path = np.array(
    prediction_path,
    dtype=float
)

distance_history = np.array(
    distance_history,
    dtype=float
)

# =========================================================
# 11. STATISTICS
# =========================================================

forward_count = decision_history.count(
    "FORWARD"
)

left_count = decision_history.count(
    "LEFT"
)

right_count = decision_history.count(
    "RIGHT"
)

high_count = risk_history.count(
    "HIGH"
)

medium_count = risk_history.count(
    "MEDIUM"
)

low_count = risk_history.count(
    "LOW"
)


# =========================================================
# 12. RESULTS
# =========================================================

print()
print("=" * 55)
print("ACTUAL GRU BASED AUTONOMOUS NAVIGATION")
print("=" * 55)

print()

print("Final Robot Position:")

print(
    f"X = {robot_x:.2f} m"
)

print(
    f"Y = {robot_y:.2f} m"
)

print()

print("Navigation Decisions")

print(
    f"FORWARD = {forward_count}"
)

print(
    f"LEFT    = {left_count}"
)

print(
    f"RIGHT   = {right_count}"
)

print()

print("Collision Risk")

print(
    f"HIGH    = {high_count}"
)

print(
    f"MEDIUM  = {medium_count}"
)

print(
    f"LOW     = {low_count}"
)
#==============================================
print("robot_path:", robot_path)
print("robot_path length:", len(robot_path))

robot_path = np.array(robot_path, dtype=float)

print("robot_path shape:", robot_path.shape)


#================================================
#PERFORMANCE EVALUATION
#================================================
# =========================================================
# STEP 19 - PERFORMANCE EVALUATION
# =========================================================

print()
print("=" * 60)
print("STEP 19 - PERFORMANCE EVALUATION")
print("=" * 60)


# ---------------------------------------------------------
# 1. Final Robot Position
# ---------------------------------------------------------

final_robot_x = robot_path[-1, 0]
final_robot_y = robot_path[-1, 1]


# ---------------------------------------------------------
# 2. Distance from Robot to Goal
# ---------------------------------------------------------

final_goal_distance = np.sqrt(
    (goal_x - final_robot_x) ** 2 +
    (goal_y - final_robot_y) ** 2
)

GOAL_THRESHOLD = 0.5

if final_goal_distance <= GOAL_THRESHOLD:
    goal_reached = True
else:
    goal_reached = False


# ---------------------------------------------------------
# 3. Total Robot Path Length
# ---------------------------------------------------------

path_length = 0.0

for i in range(1, len(robot_path)):

    dx = (
        robot_path[i, 0]
        - robot_path[i - 1, 0]
    )

    dy = (
        robot_path[i, 1]
        - robot_path[i - 1, 1]
    )

    path_length += np.sqrt(
        dx ** 2 + dy ** 2
    )


# ---------------------------------------------------------
# 4. Actual Robot-Obstacle Distance
# ---------------------------------------------------------

actual_distances = []

for i in range(
    min(
        len(robot_path),
        len(obstacle_path)
    )
):

    dx = (
        obstacle_path[i, 0]
        - robot_path[i, 0]
    )

    dy = (
        obstacle_path[i, 1]
        - robot_path[i, 1]
    )

    distance = np.sqrt(
        dx ** 2 + dy ** 2
    )

    actual_distances.append(
        distance
    )


actual_distances = np.array(
    actual_distances
)


# ---------------------------------------------------------
# 5. Minimum Obstacle Distance
# ---------------------------------------------------------

if len(actual_distances) > 0:

    minimum_distance = np.min(
        actual_distances
    )

else:

    minimum_distance = 0.0


# ---------------------------------------------------------
# 6. Collision Detection
# ---------------------------------------------------------

COLLISION_DISTANCE = 0.5

collision_count = np.sum(
    actual_distances <= COLLISION_DISTANCE
)


# ---------------------------------------------------------
# 7. Safety Rate
# ---------------------------------------------------------

if len(actual_distances) > 0:

    safe_steps = np.sum(
        actual_distances > COLLISION_DISTANCE
    )

    safety_rate = (
        safe_steps
        / len(actual_distances)
    ) * 100

else:

    safety_rate = 0.0


# ---------------------------------------------------------
# 8. GRU Prediction Error
# ---------------------------------------------------------

evaluation_length = min(
    len(obstacle_path),
    len(prediction_path)
)

if evaluation_length > 0:

    actual_prediction = obstacle_path[
        :evaluation_length
    ]

    gru_prediction = prediction_path[
        :evaluation_length
    ]

    # MAE
    mae = np.mean(
        np.abs(
            actual_prediction
            -
            gru_prediction
        )
    )

    # RMSE
    rmse = np.sqrt(
        np.mean(
            (
                actual_prediction
                -
                gru_prediction
            ) ** 2
        )
    )

else:

    mae = 0.0
    rmse = 0.0


# ---------------------------------------------------------
# 9. Print Final Results
# ---------------------------------------------------------

print()

print("FINAL ROBOT POSITION")
print("--------------------")

print(
    f"X = {final_robot_x:.3f} m"
)

print(
    f"Y = {final_robot_y:.3f} m"
)

print()

print("GOAL INFORMATION")
print("--------------------")

print(
    f"Goal X = {goal_x:.3f} m"
)

print(
    f"Goal Y = {goal_y:.3f} m"
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

print("NAVIGATION PERFORMANCE")
print("--------------------")

print(
    f"Total Path Length = "
    f"{path_length:.3f} m"
)

print(
    f"Minimum Obstacle Distance = "
    f"{minimum_distance:.3f} m"
)

print(
    f"Collision Count = "
    f"{collision_count}"
)

print(
    f"Safety Rate = "
    f"{safety_rate:.2f}%"
)

print()

print("GRU PREDICTION PERFORMANCE")
print("--------------------")

print(
    f"MAE = "
    f"{mae:.5f} m"
)

print(
    f"RMSE = "
    f"{rmse:.5f} m"
)

print()

print("=" * 60)
print("PERFORMANCE EVALUATION COMPLETED")
print("=" * 60)



# =========================================================
# 13. GRAPH
# =========================================================

plt.figure(
    figsize=(10, 7)
)

plt.plot(
    robot_path[:, 0],
    robot_path[:, 1],
    linewidth=2,
    label="Robot Path"
)

plt.plot(
    obstacle_path[:, 0],
    obstacle_path[:, 1],
    linewidth=2,
    label="Actual Obstacle"
)

plt.plot(
    prediction_path[:, 0],
    prediction_path[:, 1],
    linestyle="--",
    linewidth=2,
    label="GRU Prediction"
)

plt.scatter(
    0,
    0,
    s=180,
    label="Robot Start"
)

plt.scatter(
    goal_x,
    goal_y,
    s=180,
    marker="X",
    label="Goal"
)

plt.xlabel(
    "X Position (m)"
)

plt.ylabel(
    "Y Position (m)"
)

plt.title(
    "Actual GRU Based Adaptive Navigation"
)

plt.grid()

plt.legend()

plt.axis("equal")

plt.show()


# =========================================================
# 14. COLLISION RISK GRAPH
# =========================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    distance_history,
    linewidth=2,
    label="GRU Predicted Distance"
)

plt.axhline(
    SAFE_DISTANCE,
    linestyle="--",
    linewidth=2,
    label="Safety Distance"
)

plt.axhline(
    WARNING_DISTANCE,
    linestyle="--",
    linewidth=2,
    label="Warning Distance"
)

plt.xlabel(
    "Time Step"
)

plt.ylabel(
    "Distance (m)"
)

plt.title(
    "GRU Based Collision Risk Prediction"
)

plt.grid()

plt.legend()

plt.show()