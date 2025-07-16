import pybullet as p
import pybullet_data
import time
import numpy as np

class OrionEnv:
    def __init__(self, render=False):
        """
        Initializes the simulation environment.
        - render: If True, the simulation GUI will be displayed.
        """
        if render:
            self._physics_client_id = p.connect(p.GUI)
        else:
            self._physics_client_id = p.connect(p.DIRECT) # No GUI

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # --- Basic Environment Setup ---
        self.time_step = 1./240.
        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        # --- Load Robot ---
        # The robot's base is positioned at (0,0,0) with no rotation.
        robot_start_pos = [0, 0, 0]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", robot_start_pos, robot_start_orientation, useFixedBase=1)
        
        # --- Load Objects ---
        self.object_ids = []
        self.object_positions = {
            "red_cube": [0.7, 0.2, 0.65],
            "blue_sphere": [0.7, -0.2, 0.65],
            "green_cube": [0.5, 0.0, 0.65]
        }
        self._load_objects()

        # --- Camera Setup ---
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.2, 0, 1.2],
            cameraTargetPosition=[0.6, 0, 0.6],
            cameraUpVector=[0, 0, 1])
        
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.1)

    def _load_objects(self):
        """Loads objects into the scene."""
        red_cube = p.loadURDF("cube_small.urdf", self.object_positions["red_cube"])
        p.changeVisualShape(red_cube, -1, rgbaColor=[1, 0, 0, 1])
        
        blue_sphere = p.loadURDF("sphere_small.urdf", self.object_positions["blue_sphere"])
        p.changeVisualShape(blue_sphere, -1, rgbaColor=[0, 0, 1, 1])

        green_cube = p.loadURDF("cube_small.urdf", self.object_positions["green_cube"])
        p.changeVisualShape(green_cube, -1, rgbaColor=[0, 1, 0, 1])
        
        self.object_ids.extend([red_cube, blue_sphere, green_cube])

    def get_observation(self):
        """Returns a camera image of the scene."""
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=128,
            height=128,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix)
        
        # Reshape and normalize image
        obs = np.reshape(rgb_img, (128, 128, 4))[:,:,:3] # Get RGB
        obs = obs / 255.0 # Normalize to [0, 1]
        return obs

    def step(self, action):
        """
        Executes an action in the simulation.
        This is a placeholder for your RL agent's action execution.
        """
        # In a real implementation, 'action' would control the robot's joints
        p.stepSimulation()
        time.sleep(self.time_step)
        
        # Placeholder for reward and done signal
        reward = 0
        done = False
        
        return self.get_observation(), reward, done, {}

    def reset(self):
        """Resets the environment to its initial state."""
        # This function would reset robot and object positions
        return self.get_observation()
        
    def close(self):
        """Closes the simulation."""
        p.disconnect()

# --- Example of how to run the environment ---
if __name__ == "__main__":
    env = OrionEnv(render=True)
    
    # Run simulation for a few seconds
    for _ in range(1000):
        obs = env.get_observation()
        env.step(action=None) # No action yet
        
    env.close()
