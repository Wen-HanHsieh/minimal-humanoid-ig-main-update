import numpy as np

# Define the H1 robot kinematics parameters
L1 = 0.1  # Length from torso to shoulder
L2 = 0.3  # Length from shoulder to elbow
L3 = 0.3  # Length from elbow to wrist

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def inverse_kinematics(target_position, side='left'):
    x, y, z = target_position

    # Calculate shoulder roll and pitch angles
    shoulder_pitch = np.arctan2(z, np.sqrt(x**2 + y**2))
    shoulder_roll = np.arctan2(y, x) if side == 'left' else np.arctan2(-y, x)

    # Calculate elbow pitch angle using the law of cosines
    d = np.sqrt(x**2 + y**2 + z**2)
    elbow_pitch = np.pi - np.arccos(clamp((L2**2 + L3**2 - d**2) / (2 * L2 * L3), -1.0, 1.0))

    # Placeholder for additional DoFs
    elbow_roll = 0.0  # Placeholder value, should be computed based on kinematics
    wrist_pitch = 0.0  # Placeholder value, should be computed based on kinematics
    wrist_yaw = 0.0  # Placeholder value, should be computed based on kinematics
    wrist_roll = 0.0  # Placeholder value, should be computed based on kinematics

    return shoulder_pitch, shoulder_roll, elbow_pitch, elbow_roll, wrist_pitch, wrist_yaw, wrist_roll

def torso_rotation(target_position):
    x, y, _ = target_position
    return np.arctan2(y, x)

def compute_ik(target_left_wrist, target_right_wrist, target_torso):
    left_arm_angles = inverse_kinematics(target_left_wrist, side='left')
    right_arm_angles = inverse_kinematics(target_right_wrist, side='right')
    torso_angle = [torso_rotation(target_torso)]  # Ensure this is a list to match the expected format

    return {
        "left_arm": left_arm_angles,
        "right_arm": right_arm_angles,
        "torso": torso_angle
    }

