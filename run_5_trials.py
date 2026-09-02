import csv

TRIALS = 5

results = []

def get_float(message):
    while True:
        try:
            return float(input(message))
        except ValueError:
            print("Please enter a number, for example: 24.327")

def get_int(message):
    while True:
        try:
            return int(input(message))
        except ValueError:
            print("Please enter a whole number, for example: 0")

for trial in range(1, TRIALS + 1):

    print("\n" + "=" * 50)
    print(f"TRIAL {trial}/5")
    print("=" * 50)

    input("Experiment complete hone ke baad ENTER press karein.")

    path_length = get_float("Path Length (m): ")
    min_distance = get_float("Minimum Distance (m): ")
    collisions = get_int("Collision Count: ")
    safety_rate = get_float("Safety Rate (%): ")
    goal_distance = get_float("Goal Distance (m): ")

    goal_reached = input(
        "Goal Reached? (True/False): "
    ).strip()

    results.append([
        trial,
        path_length,
        min_distance,
        collisions,
        safety_rate,
        goal_distance,
        goal_reached
    ])

with open("baseline_5_trials.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Trial",
        "Path Length (m)",
        "Minimum Distance (m)",
        "Collision Count",
        "Safety Rate (%)",
        "Goal Distance (m)",
        "Goal Reached"
    ])

    writer.writerows(results)

print("\n" + "=" * 50)
print("5 TRIALS COMPLETED")
print("=" * 50)
print("CSV saved: 5_trial_results.csv")