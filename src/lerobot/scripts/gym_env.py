import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DeskCleanEnv(gym.Env):
    def __init__(self):
        super().__init__()

        #basic settings
        self.obs_dim = 16
        self.num_objects = 4
        self.num_bins = 4
        self.image_width = 640
        self.image_height = 480

        self.object_names = ["Screwdriver","Battery","Tape","Cup"]
        self.bin_names = ["first_drawer","second_drawer","gray_bin","white_bin"]

        #object id
        self.SCREWDRIVER = 0
        self.BATTERY = 1
        self.BLACK_TAPE = 2
        self.CUP = 3
        #bin id
        self.FIRST_DRAWER = 0
        self.SECOND_DRAWER = 1
        self.GRAY_BIN = 2
        self.WHITE_BIN = 3
        #bin location
        self.bin_locations = np.array([
            [470, 100],  # first_drawer
            [470, 100],  # second_drawer
            [60, 150],  # gray_bin
            [590, 150],  # white_bin
        ], dtype=np.float32)
        self.ARM_LOCATION = np.array([580, 340], dtype=np.float32)

        #object weights
        self.object_weights = np.array([
            2.0,  # Screwdriver
            4.0,  # Battery
            1.0,  # Black tape
            1.0   # Cup
        ], dtype=np.float32)

        #bin capacities
        self.bin_capacities = np.array([
            5.0,  # first_drawer
            5.0,  # second_drawer
            20.0,  # gray_bin
            10.0   # white_bin
        ], dtype=np.float32)

        #action space (4 objects * 4 bins = 16 actions)
        self.action_space = spaces.Discrete(self.num_objects * self.num_bins)

        #observation space
        obs_low = np.zeros(self.obs_dim, dtype=np.float32)
        obs_high = np.array([self.image_width, self.image_height, 1.0, self.image_width, self.image_height, 1.0,
            self.image_width, self.image_height, 1.0, self.image_width, self.image_height, 1.0,
            self.bin_capacities[0], self.bin_capacities[1], self.bin_capacities[2], self.bin_capacities[3],], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        #cycle settings for multi cycle
        self.current_cycle = 1
        self.max_cycles = 2

        #run variables
        self.current_step = 0
        self.max_steps_per_cycle = self.num_objects
        self.max_steps = self.num_objects * self.max_cycles
        self.current_obs = np.zeros(self.obs_dim, dtype=np.float32)
        self.bin_weights = np.zeros(self.num_bins, dtype=np.float32)

        #reward value
        self.cup_drawer_penalty = 5 #strong penalty => ban
        self.bin_distance_penalty_ratio = 0.00002
        self.capacity_bonus_ratio = 0.16
        self.heavy_early_bonus_ratio = 0.8
        
        self.spawn_x_range = [275, 550]
        self.spawn_y_range = [125, 450]

    def _get_obs(self):
        return self.current_obs.copy().astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.current_cycle = 1
        self.current_obs = np.zeros(self.obs_dim, dtype=np.float32)

        #spawn object (random)
        for obj_id in range(self.num_objects):
            base_idx = obj_id * 3

            x = self.np_random.uniform(self.spawn_x_range[0], self.spawn_x_range[1])
            y = self.np_random.uniform(self.spawn_y_range[0], self.spawn_y_range[1])

            self.current_obs[base_idx] = x
            self.current_obs[base_idx + 1] = y
            self.current_obs[base_idx + 2] = 1.0

        #set bin weight (random)
        self.bin_weights = self.np_random.uniform(low=0.0, high=self.bin_capacities * 0.5).astype(np.float32)
        self.current_obs[12:16] = self.bin_weights  # update obs bin weights

        observation = self._get_obs()
        info = {}

        return observation, info
    
    def step(self, action):
        self.current_step += 1
        reward = 0.0
        success = False

        #get info from action space
        target_object = action // self.num_bins
        target_bin = action % self.num_bins
        object_idx = target_object * 3

        #invalid action check
        if self.current_obs[object_idx + 2] == 0.0:
            reward = -1.0
            terminated = False
            truncated = self.current_step >= self.max_steps
            info = {"success": False, "invalid_action": True}
            return self._get_obs(), reward, terminated, truncated, info

        #remove object from table
        object_pos = self.current_obs[object_idx : object_idx + 2].copy()   #save object's location
        self.current_obs[object_idx] = 0.0
        self.current_obs[object_idx + 1] = 0.0
        self.current_obs[object_idx + 2] = 0.0
        #update bin weight
        object_weight = self.object_weights[target_object]
        self.bin_weights[target_bin] += object_weight
        self.current_obs[12:16] = self.bin_weights
        success = True

        # <REWARD CALCULATE>

        #penalty if cup into drawer
        if target_object == self.CUP and target_bin in [self.FIRST_DRAWER, self.SECOND_DRAWER]:
            reward -= self.cup_drawer_penalty
        
        #penalty for long distance from bin
        bin_pos = self.bin_locations[target_bin]
        bin_distance = np.linalg.norm(object_pos - bin_pos)
        bin_distance_penalty = bin_distance * bin_distance * self.bin_distance_penalty_ratio
        reward -= bin_distance_penalty

        #bonus for capacity left (penalty if overweight)
        remaining_capacity = self.bin_capacities[target_bin] - self.bin_weights[target_bin]
        capacity_bonus = remaining_capacity * self.capacity_bonus_ratio
        reward += capacity_bonus
        if remaining_capacity < 0:
            reward -= 1.0

        #bonus for cleaning heavy object early
        cycle_step = self.current_step - ((self.current_cycle - 1) * self.num_objects)
        early_factor = 1.0 - (cycle_step / self.num_objects)    #0.25 per step
        heavy_early_bonus = (self.object_weights[target_object] * early_factor * self.heavy_early_bonus_ratio)
        reward += heavy_early_bonus

        #check the end
        object_exists = self.current_obs[2:12:3]
        cycle_done = np.sum(object_exists) == 0
        #cycle check
        if cycle_done:
            if self.current_cycle < self.max_cycles:
                self.current_cycle += 1
                #respawn objects only
                for obj_id in range(self.num_objects):
                    base_idx = obj_id * 3

                    x = self.np_random.uniform(self.spawn_x_range[0], self.spawn_x_range[1])
                    y = self.np_random.uniform(self.spawn_y_range[0], self.spawn_y_range[1])

                    self.current_obs[base_idx] = x
                    self.current_obs[base_idx + 1] = y
                    self.current_obs[base_idx + 2] = 1.0
                terminated = False
            else:
                terminated = True
        else:
            terminated = False
        truncated = self.current_step >= self.max_steps

        observation = self._get_obs()
        info = {"current_cycle": self.current_cycle, "cycle_done": cycle_done}
        return observation, reward, terminated, truncated, info
    
    def obs_to_action_mask(self, obs):
        mask = np.zeros(self.action_space.n, dtype=np.int32)

        for obj_id in range(self.num_objects):
            object_idx = obj_id * 3
            object_exist = obs[object_idx + 2]

            if object_exist == 1.0:
                for bin_id in range(self.num_bins):
                    action = (obj_id * self.num_bins + bin_id)
                    mask[action] = 1

        return mask