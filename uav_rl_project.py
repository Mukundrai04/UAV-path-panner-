import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

# ==========================================
# 1. Custom UAV Environment
# ==========================================

class IndoorUAVEnv(gym.Env):
    """
    A custom 2D environment for Indoor UAV Path Planning.
    The drone must navigate from a start point to a goal while avoiding obstacles.
    """
    
    def __init__(self):
        super(IndoorUAVEnv, self).__init__()
        
        # --- Environment Parameters ---
        self.grid_size = 10.0          # Room size (10m x 10m)
        self.dt = 0.1                  # Time step
        self.max_steps = 200           # Max steps per episode
        self.goal_threshold = 0.3      # Distance to consider goal reached
        self.obstacle_radius = 0.5     # Radius of obstacles
        
        # Define Obstacles (x, y) positions
        self.obstacles = np.array([
            [3.0, 3.0],
            [5.0, 7.0],
            [7.0, 4.0],
            [2.0, 8.0]
        ])
        
        # --- Action Space ---
        # Action: [velocity_x, velocity_y] continuous values between -1 and 1
        # We scale this internally to max_speed
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # --- Observation Space ---
        # State: [drone_x, drone_y, relative_goal_x, relative_goal_y, lidar_reading_1, ...]
        # For simplicity, we use relative vector to goal + 4 simple distance sensors
        # Observation: [rel_goal_x, rel_goal_y, dist_N, dist_S, dist_E, dist_W]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # Initialize state
        self.state = None
        self.steps_taken = 0
        self.start_pos = np.array([0.5, 0.5])
        self.goal_pos = np.array([9.0, 9.0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset drone to start position
        self.drone_pos = self.start_pos.copy()
        self.steps_taken = 0
        
        # Calculate initial observation
        observation = self._get_obs()
        self.state = observation
        
        return observation, {}

    def _get_obs(self):
        # Relative position to goal
        rel_goal = self.goal_pos - self.drone_pos
        
        # Simple "Fake" LIDAR: Calculate distance to nearest obstacle in 4 directions
        # (In a real sim, you would use ray-casting)
        dist_n, dist_s, dist_e, dist_w = 10.0, 10.0, 10.0, 10.0 # Default max range
        
        for obs in self.obstacles:
            dx = obs[0] - self.drone_pos[0]
            dy = obs[1] - self.drone_pos[1]
            
            # Very simplified sensor logic
            if dx > 0 and abs(dy) < self.obstacle_radius: dist_e = min(dist_e, dx)
            if dx < 0 and abs(dy) < self.obstacle_radius: dist_w = min(dist_w, -dx)
            if dy > 0 and abs(dx) < self.obstacle_radius: dist_n = min(dist_n, dy)
            if dy < 0 and abs(dx) < self.obstacle_radius: dist_s = min(dist_s, -dy)
                
        observation = np.array([
            rel_goal[0], rel_goal[1], 
            dist_n, dist_s, dist_e, dist_w
        ], dtype=np.float32)
        
        return observation

    def step(self, action):
        self.steps_taken += 1
        
        # --- 1. Apply Action (Kinematics) ---
        # Scale action to actual speed (max 1.0 m/s)
        speed_scale = 1.0 
        velocity = action * speed_scale
        
        # Update position: pos = pos + vel * dt
        self.drone_pos += velocity * self.dt
        
        # Clip to grid boundaries
        self.drone_pos = np.clip(self.drone_pos, 0, self.grid_size)
        
        # --- 2. Calculate Reward ---
        terminated = False
        truncated = False
        reward = 0.0
        
        # Check Collision with Obstacles
        collision = False
        for obs in self.obstacles:
            dist_to_obs = np.linalg.norm(self.drone_pos - obs)
            if dist_to_obs < self.obstacle_radius:
                collision = True
                break
        
        # Check Goal Reached
        dist_to_goal = np.linalg.norm(self.drone_pos - self.goal_pos)
        
        if collision:
            reward = -100.0      # Heavy penalty for crashing
            terminated = True
        elif dist_to_goal < self.goal_threshold:
            reward = 100.0       # Big reward for reaching goal
            terminated = True
        else:
            # Shaping Reward: Encourage moving towards goal
            # Reward is higher if closer to goal
            reward = -dist_to_goal * 0.1 
            
            # Small time penalty to encourage speed
            reward -= 0.1 
            
            # Proximity penalty (optional: discourage getting too close to walls)
            # Simplified here by the collision check
            
        # Truncate if taking too long
        if self.steps_taken >= self.max_steps:
            truncated = True
            
        # Get new observation
        observation = self._get_obs()
        
        return observation, reward, terminated, truncated, {}

    def render(self, mode='human'):
        # Optional: For visualization during training (usually done separately)
        pass

# ==========================================
# 2. Training Script
# ==========================================

def train_agent():
    print("Initializing Environment...")
    
    # Create the environment
    env = IndoorUAVEnv()
    
    # Wrap it for vectorized training (optional but recommended for SB3)
    # Note: Custom envs need careful handling, usually we pass the class instance directly for simple testing
    # But for SB3, it's easier to just pass the instance if not parallelized.
    
    print("Initializing PPO Model...")
    # MlpPolicy = Multi-Layer Perceptron (standard neural network)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
    
    print("Training started...")
    # Train for 50,000 timesteps
    model.learn(total_timesteps=50000)
    
    print("Training finished. Saving model...")
    model.save("uav_path_planner")
    
    return model

# ==========================================
# 3. Visualization Script
# ==========================================

def test_and_visualize(model_path=None):
    # Create a fresh environment for testing
    env = IndoorUAVEnv()
    
    # Load the model or create a random agent for comparison
    if model_path:
        model = PPO.load(model_path, env=env)
    else:
        model = None # Will act random
        
    # Setup Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Run one episode
    obs, _ = env.reset()
    done = False
    
    # Store path for plotting
    path_x = [env.drone_pos[0]]
    path_y = [env.drone_pos[1]]
    
    while not done:
        if model:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
        else:
            # Random action
            action = env.action_space.sample()
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        path_x.append(env.drone_pos[0])
        path_y.append(env.drone_pos[1])

    # --- Plotting Logic ---
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_title(f"UAV Path Planning (Steps: {env.steps_taken})")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    # Draw Obstacles
    for obs in env.obstacles:
        circle = plt.Circle((obs[0], obs[1]), env.obstacle_radius, color='red', alpha=0.5)
        ax.add_patch(circle)
        
    # Draw Start and Goal
    ax.scatter(env.start_pos[0], env.start_pos[1], marker='o', color='blue', s=100, label='Start')
    ax.scatter(env.goal_pos[0], env.goal_pos[1], marker='*', color='green', s=200, label='Goal')
    
    # Draw Path
    ax.plot(path_x, path_y, linestyle='--', marker='.', color='black', label='Drone Path')
    ax.legend()
    ax.grid(True)
    
    plt.show()

# ==========================================
# Main Execution Block
# ==========================================

if __name__ == "__main__":
    # Check if model exists, if not, train it
    if os.path.exists("uav_path_planner.zip"):
        print("Pre-trained model found. Testing...")
        test_and_visualize("uav_path_planner")
    else:
        print("No model found. Training a new one...")
        train_agent()
        print("Testing the newly trained model...")
        test_and_visualize("uav_path_planner")