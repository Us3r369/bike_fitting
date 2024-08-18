class Bike_Fit:
    def __init__(self, target_angles):
        """
        Initialize a BikeFit object with target angles.

        :param target_angles: Dictionary containing target angles for various body parts.
        """
        if target_angles is None:
            target_angles = {}
        self.target_angles = target_angles
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

    def update_actual_angle(self, angle_name, angle_value):
        """
        Update a specific actual angle.

        :param angle_name: The name of the angle to update (e.g., 'knee_angle').
        :param angle_value: The new value of the angle as a list or tuple of two integers [min, max].
        """
        self.actual_angles[angle_name] = angle_value

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