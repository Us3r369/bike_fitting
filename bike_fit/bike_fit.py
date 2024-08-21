import json

class Bike_Fit:
    def __init__(self, target_angles=None):
        """
        Initialize a BikeFit object with target angles.

        :param target_angles: Dictionary containing target angles for various body parts.
        """
        self.target_angles = target_angles if target_angles is not None else {}
        self.actual_angles = {}
        self.fitting = {}

    def set_actual_angles(self, actual_angles):
        """
        Set the actual angles measured during a bike fitting session.

        :param actual_angles: Dictionary containing actual angles for various body parts.
        """
        self.actual_angles = actual_angles

    def set_target_angle(self, angle_name, angle_value):
        """
        Set a specific target angle.
        :param angle_name: The name of the angle (e.g., 'knee_angle').
        :param angle_value: The value of the angle as a tuple (min, max).
        """
        self.target_angles[angle_name] = angle_value

    def update_actual_angles(self, angle_name, new_angle):
        """
        Update the actual angle for the given angle_name.
        If new_angle is outside the current min/max range, update it.
        """
        #if the current angle is empty, set the new angle as the min and max
        if angle_name not in self.actual_angles:
            self.actual_angles[angle_name] = [new_angle, new_angle]
        else:
            current_min, current_max = self.actual_angles.get(angle_name, [0, 0])
            self.actual_angles[angle_name] = [
                min(current_min, new_angle),
                max(current_max, new_angle)
            ]

    def compare_angles(self):
        """
        Compare the target angles with the actual angles.

        :return: A dictionary containing the comparison between target and actual angles.
        """
        comparison = {}
        for angle_name, actual_range in self.actual_angles.items():
            target_range = self.target_angles.get(angle_name, [None, None])
            actual_min, actual_max = actual_range
            target_min, target_max = target_range

            # Initialize the result for this angle
            comparison[angle_name] = {
                "status": "in_range",
                "issues": [],
                "deviation": {}
            }

            # Check if the actual min angle is too low
            if target_min is not None and actual_min < target_min:
                comparison[angle_name]["status"] = "out_of_range"
                comparison[angle_name]["issues"].append("too low")
                comparison[angle_name]["deviation"]["min_deviation"] = target_min - actual_min

            # Check if the actual max angle is too high
            if target_max is not None and actual_max > target_max:
                comparison[angle_name]["status"] = "out_of_range"
                comparison[angle_name]["issues"].append("too high")
                comparison[angle_name]["deviation"]["max_deviation"] = actual_max - target_max

            # If there are no issues, remove unnecessary fields
            if not comparison[angle_name]["issues"]:
                comparison[angle_name].pop("issues")
                comparison[angle_name].pop("deviation")

        return comparison