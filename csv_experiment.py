#!/usr/bin/env python3

import csv
import os
import subprocess
import time
from datetime import datetime

WORKSPACE = os.path.expanduser("~/phd_robot_ws")
RESULT_DIR = os.path.join(WORKSPACE, "results")

TRIALS = 5

# Change these ONLY if your actual launch/config names are different.
BASELINE_COMMAND = [
    "ros2", "launch",
    "phd_robot_simulation", "full_system.launch.py",
    "algorithm:=baseline"
]

ADAPTIVE_GRU_COMMAND = [
    "ros2", "launch",
    "phd_robot_simulation", "full_system.launch.py",
    "algorithm:=adaptive_gru"
]

FIELDS = [
    "trial",
    "path_length",
    "min_distance",
    "collisions",
    "safety_rate",
    "goal_distance",
    "goal_reached"
]


def create_csv(filename):
    path = os.path.join(RESULT_DIR, filename)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

    return path


def run_trial(method, command, trial):
    print("\n" + "=" * 70)
    print(f"{method} | TRIAL {trial}/{TRIALS}")
    print("=" * 70)

    print("Starting simulation...")

    process = subprocess.Popen(command)

    try:
        # Allow Gazebo/ROS nodes to start
        time.sleep(15)

        print("Waiting for navigation experiment...")
        print("Run your goal/navigation evaluation here.")

        # ---------------------------------------------------------
        # IMPORTANT:
        # Replace this section with your evaluation_node output
        # or /navigation/metrics subscriber.
        # ---------------------------------------------------------

        input(
            "\nPress ENTER after the robot reaches the goal "
            "or the trial finishes..."
        )

        # Temporary placeholders.
        # These must be replaced by values received from
        # evaluation_node /navigation/metrics.
        result = {
            "trial": trial,
            "path_length": "",
            "min_distance": "",
            "collisions": "",
            "safety_rate": "",
            "goal_distance": "",
            "goal_reached": ""
        }

        return result

    finally:
        print("Stopping simulation...")
        process.terminate()

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

        time.sleep(3)


def save_result(filename, result):
    path = os.path.join(RESULT_DIR, filename)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow(result)


def run_experiment(method, command, filename):

    print("\n\n")
    print("#" * 70)
    print(f"STARTING 5-TRIAL EXPERIMENT: {method}")
    print("#" * 70)

    for trial in range(1, TRIALS + 1):

        result = run_trial(
            method,
            command,
            trial
        )

        save_result(filename, result)

        print(f"Trial {trial} saved.")

    print("\nExperiment completed:", method)


def main():

    os.makedirs(RESULT_DIR, exist_ok=True)

    baseline_file = "baseline_results.csv"
    adaptive_file = "adaptive_gru_results.csv"

    create_csv(baseline_file)
    create_csv(adaptive_file)

    print("\n5-TRIAL AUTOMATED NAVIGATION EXPERIMENT")
    print("========================================")

    run_experiment(
        "Baseline",
        BASELINE_COMMAND,
        baseline_file
    )

    run_experiment(
        "Adaptive GRU",
        ADAPTIVE_GRU_COMMAND,
        adaptive_file
    )

    print("\n" + "=" * 70)
    print("ALL 10 TRIALS COMPLETED")
    print("=" * 70)

    print(
        f"\nResults saved in:\n"
        f"{RESULT_DIR}/baseline_results.csv\n"
        f"{RESULT_DIR}/adaptive_gru_results.csv"
    )


if __name__ == "__main__":
    main()