import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append("..")

from detect_test import detect_zone, detected_to_obs
from gym_env import RLtest

# Dueling Network
class DuelingNetwork(nn.Module):
    def __init__(self, input_dim, n_outputs, hidden_layer=128):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_layer),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer // 2),
            nn.ReLU(),
            nn.Linear(hidden_layer // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer // 2),
            nn.ReLU(),
            nn.Linear(hidden_layer // 2, n_outputs),
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
    
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#load env
env = RLtest()
obs_size = env.observation_space.shape[0]
num_actions = env.action_space.n
model = DuelingNetwork(obs_size, num_actions, hidden_layer=128).to(device)
model.load_state_dict(torch.load("../rltrain/models/rltest_dqn.pth", map_location=device))
model.eval()

#current bin state
left_total = 0
right_total = 0

def decide_action(image, left_total, right_total):
    #get obs from image
    detected_objects = detect_zone(image)
    obs = detected_to_obs(detected_objects, env.num_zones, env.num_colors, left_total, right_total)

    #mask invalid actions
    mask = env.obs_to_action_mask(obs)

    #get action from rl model
    with torch.no_grad():
        decision_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = model(decision_obs).squeeze(0).cpu().numpy()
        q_values[mask == 0] = -1e9
        action = int(np.argmax(q_values))

    #action to human language
    zone = action // (env.num_colors * env.num_bins)
    rest = action % (env.num_colors * env.num_bins)
    color = rest // env.num_bins
    target_bin = rest % env.num_bins
    color_name = "blue box" if color == 0 else "green box"
    bin_name = "left bin" if target_bin == 0 else "right bin"
    weight = 1 if color == 0 else 3
    script = f"Pick the {color_name} from zone {zone} and place it into the {bin_name}."
    
    return {"script": script,
            "target_bin_id": target_bin,
            "weight": weight}

#test run
if __name__ == "__main__":
    image_path = "../rltrain/images/test_image.png"

    # current bin state
    left_total = 0
    right_total = 0

    decision = decide_action(image=image_path, left_total=left_total, right_total=right_total)

    print("\n========== RL Decision ==========")
    print(decision["script"])
    
    if decision["target_bin_id"] == 0:
        left_total += decision["weight"]
    else:
        right_total += decision["weight"]