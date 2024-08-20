class Bike_Fit:
    def __init__(self, target_angles=None):
        """
        Initialize a BikeFit object with target angles.

        :param target_angles: Dictionary containing target angles for various body parts.
        """
        self.target_angles = target_angles if target_angles is not None else {}
        self.actual_angles = {}

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

    def save_target_ranges(self, filepath):
        """
        Save the target angles to a JSON file.
        :param filepath: The path to the JSON file where the target ranges should be saved.
        """
        with open(filepath, 'w') as f:
            json.dump(self.target_angles, f)

    def update_actual_angle(self, angle_name, new_angle):
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
        for angle in self.target_angles:
            target_value = self.target_angles[angle]
            actual_value = self.actual_angles.get(angle, None)
            if actual_value:
                comparison[angle] = {
                    "target": target_value,
                    "actual": actual_value,
                    "difference": (actual_value[0] - target_value[0], actual_value[1] - target_value[1])
                }
            else:
                comparison[angle] = {
                    "target": target_value,
                    "actual": "Not set",
                    "difference": "Not available"
                }
        return comparison