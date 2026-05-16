import os
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from lerobot.scripts.yolo_detect import detect_object, detected_to_obs
from lerobot.scripts.gym_env import DeskCleanEnv

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

# Load env
env = DeskCleanEnv()
obs_size = env.observation_space.shape[0]
num_actions = env.action_space.n

# Load RL model
model = DuelingNetwork(obs_size, num_actions, hidden_layer=128).to(device)
MODEL_PATH = os.path.join(BASE_DIR, "../vla_rl/models/deskclean_dqn.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def decide_action(obs):
    # Mask invalid actions
    action_mask = env.obs_to_action_mask(obs)

    if np.sum(action_mask) == 0:
        print("정리할 물건이 없습니다.")
        return {"success": False,}

    # action selection from RL model
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model(obs_t).squeeze(0).cpu().numpy()
        q_values[action_mask == 0] = -1e9
        action = int(np.argmax(q_values))

    # Convert to human language
    target_object = action // env.num_bins
    target_bin = action % env.num_bins
    object_name = env.object_names[target_object]
    bin_name = env.bin_names[target_bin]
    object_weight = float(env.object_weights[target_object])

    script = f"Pick up the {object_name} and place it into the {bin_name}."

    return {
        "success": True,
        "script": script,
        "target_bin_name": bin_name,
        "weight": object_weight,
        "target_object_id": target_object
    }

def update_bin_weights(bin_weights, decision):
    if not decision.get("success", False):
        return bin_weights

    target_bin_name = decision["target_bin_name"]
    weight = decision["weight"]

    bin_weights[target_bin_name] += weight
    return bin_weights
'''
if __name__ == "__main__":

    bin_weights = {
    "first_drawer": 0.0,
    "second_drawer": 0.0,
    "gray_bin": 0.0,
    "white_bin": 0.0,
}
    
    image_path = os.path.join(BASE_DIR, "test_image.png")
    detected_objects = detect_object(image_path)
    obs = detected_to_obs(detected_objects, bin_weights)

    decision = decide_action(obs)

    print("\n===== RL Decision =====")

    if decision["success"]:
        print(decision["script"])

        bin_weights = update_bin_weights(bin_weights, decision)
        print("Updated bin weights:", bin_weights)
'''