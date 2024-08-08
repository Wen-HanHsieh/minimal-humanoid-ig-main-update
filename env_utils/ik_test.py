import numpy as np
from h1_ik_solver import compute_ik

# Define target positions
target_left_wrist = [0.2, 0.1, 0.3]
target_right_wrist = [0.2, -0.1, 0.3]
target_torso = [0.0, 0.0, 0.5]

# Compute IK
ik_results = compute_ik(target_left_wrist, target_right_wrist, target_torso)

# Print results
print("Left Arm Angles (Shoulder Pitch, Shoulder Roll, Elbow Pitch):", ik_results["left_arm"])
print("Right Arm Angles (Shoulder Pitch, Shoulder Roll, Elbow Pitch):", ik_results["right_arm"])
print("Torso Rotation:", ik_results["torso"])

