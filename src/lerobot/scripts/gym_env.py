import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RLtest(gym.Env):
    def __init__(self):
        #basics
        self.num_zones = 9
        self.num_colors = 2
        self.num_bins = 2
        self.max_objects = 10
        self.max_bin_weight = 50
        self.max_steps = 50

        #define weight of each box
        self.w_blue = 1
        self.w_green = 3

        #define grasp probablity of each zone
        self.zone_grasp_prob = {
            0:0.7, 1:1.0, 2:0.7,
            3:0.8, 4:1.0, 5:0.8,
            6:0.9, 7:1.0, 8:0.9
        } 

        #define action space
        self.action_space = spaces.Discrete(self.num_zones * self.num_colors * self.num_bins)
        #define observation space
        obs_dim = (self.num_zones * self.num_colors) + (self.num_bins + 1)  #1: diff of bin's weight
        obs_low = np.zeros(obs_dim,dtype=np.int32)
        obs_high = np.array([self.max_objects]*(self.num_zones * self.num_colors) + 
                            [(self.max_objects*3), (self.max_objects*3), self.max_bin_weight])
        self.observation_space = spaces.Box(obs_low,obs_high,dtype=np.int32)

        #variables to use
        self.left_total = 0
        self.right_total = 0
        self.bin_diff = 0
        self.current_steps = 0
        #array to save current box location
        self.table_info = np.zeros((self.num_zones,self.num_colors),dtype=np.int32)
        #[zone / blue / green]
        #[        0       0]     < 9 lines of this

    def _get_obs(self):
        obs=[]
        #append info of boxes of each zone
        for zone in range(self.num_zones):
            for color in range(self.num_colors):
                obs.append(self.table_info[zone, color])
        #apped rest info
        obs.append(self.left_total)
        obs.append(self.right_total)
        obs.append(self.bin_diff)

        return np.array(obs, dtype=np.int32)    #total 21
    
    def reset(self,seed=None):
        super().reset(seed=seed)    #to use random
        self.left_total = 0
        self.right_total = 0
        self.bin_diff = 0
        self.current_steps = 0

        #reset table state
        self.table_info = np.zeros((self.num_zones,self.num_colors),dtype=np.int32)

        #generate random boxes
        num_objects = np.random.randint(1, self.max_objects + 1)
        for _ in range(num_objects):
            zone = np.random.randint(0,self.num_zones)
            color = np.random.randint(0,self.num_colors)
            self.table_info[zone,color] += 1

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self,action):
        self.current_steps += 1
        reward = 0
        prev_diff = abs(self.left_total - self.right_total)
        success = False
        
        #get info from action array
        #color: 0:blue, 1:green
        #bin: 0:left, 1:right
        target_zone = action // (self.num_colors * self.num_bins)
        rest = action % (self.num_colors * self.num_bins)
        target_color = rest // self.num_bins
        destin_bin = rest % self.num_bins

        #invalid action check
        if self.table_info[target_zone, target_color] <= 0:
            reward = -1
            terminated = False
            truncated = self.current_steps >= self.max_steps
            observation = self._get_obs()
            info = {"invalid_action": True}
            return observation, reward, terminated, truncated, info
        
        #calculate grasp prob -> actuate
        grasp_prob = self.zone_grasp_prob[target_zone]
        if np.random.rand() < grasp_prob:
            self.table_info[target_zone, target_color] -= 1 #remove box
            weight = self.w_blue if target_color == 0 else self.w_green
            #put into bin
            if destin_bin == 0:
                self.left_total += weight
            else:
                self.right_total += weight
            success = True
        else:
            success = False
            
        #update bin diff
        self.bin_diff = abs(self.left_total - self.right_total)
        new_diff = self.bin_diff
        
        #calculate reward
        if success:
            success_reward = 1
            diff = abs(self.left_total - self.right_total)
            balance_bonus = 10 / (1 + diff)
            reward += success_reward + balance_bonus
            if self.left_total == self.right_total: reward += 2
        else:
            reward -= 2
        
        #detect end of task
        terminated = np.sum(self.table_info) == 0
        truncated = self.current_steps >= self.max_steps
        
        observation = self._get_obs()
        info = {}
        return observation, reward, terminated, truncated, info
    
    def get_valid_mask(self):
        mask = np.zeros(self.action_space.n, dtype=np.int32)

        for zone in range(self.num_zones):
            for color in range(self.num_colors):
                if self.table_info[zone,color] > 0:
                    for bin in range(self.num_bins):
                        avail_action = zone * (self.num_colors * self.num_bins) + color * self.num_bins + bin
                        mask[avail_action] = 1
        return mask
    
    def obs_to_action_mask(self, obs):
        mask = np.zeros(self.action_space.n, dtype=np.int32)
        zone_info = obs[: self.num_zones * self.num_colors] #get only zone info

        for zone in range(self.num_zones):
            for color in range(self.num_colors):
                idx = zone * self.num_colors + color
                if zone_info[idx] > 0:
                    for bin in range(self.num_bins):
                        avail_action = zone * (self.num_colors * self.num_bins) + color * self.num_bins + bin
                        mask[avail_action] = 1
        return mask