from ultralytics import YOLO
import numpy as np
import os
import cv2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

object_id = {
    "Screwdriver": 0,
    "Battery": 1,
    "Black tape": 2,
    "Cup": 3
}

WHITE_BIN_POLYGON = np.array([
    [540, 80],  # top_left
    [610, 80],  # top_right
    [635, 180],  # bottom_right
    [560, 180],  # bottom_left
], dtype=np.int32)

GRAY_BIN_POLYGON = np.array([
    [20, 60],  # top_left
    [110, 60],  # top_right
    [80, 215],  # bottom_right
    [0, 215],  # bottom_left
], dtype=np.int32)

def is_inside_polygon(cx, cy, polygon):
    result = cv2.pointPolygonTest(
        polygon,
        (float(cx), float(cy)),
        False
    )
    return result >= 0


def is_inside_bin(cx, cy):
    return (
        is_inside_polygon(cx, cy, WHITE_BIN_POLYGON)
        or is_inside_polygon(cx, cy, GRAY_BIN_POLYGON)
    )

def detect_object(image):
    results = model(image, conf=0.85)
    detected_objects = []

    for box in results[0].boxes:
        #get object's class name
        cls = int(box.cls[0])
        name = model.names[cls]

        #get object's polygon
        x1, y1, x2, y2 = box.xyxy[0]

        #calculate object's location
        cx = float(((x1 + x2) / 2).item())
        cy = float(((y1 + y2) / 2).item())
        #ignore if object is inside white bin
        if is_inside_bin(cx, cy):
            continue
        if cy > 470:
            continue

        print("Name: ", name)
        print("Location: ", int(cx), int(cy))

        detected_objects.append({
            "name": name,
            "id": object_id[name],
            "location": [cx, cy]
        })
    return detected_objects

def detected_to_obs(detected_objects, bin_weights):
    """
    obs structure (16 dims)
    [
        screwdriver_x,
        screwdriver_y,
        screwdriver_exist,

        battery_x,
        battery_y,
        battery_exist,

        black_tape_x,
        black_tape_y,
        black_tape_exist,

        cup_x,
        cup_y,
        cup_exist,

        first_drawer_weight,
        second_drawer_weight,
        gray_bin_weight,
        white_bin_weight
    ]
    """

    obs = np.zeros(16, dtype=np.float32)

    #object information
    for obj in detected_objects:
        obj_id = obj["id"]
        x, y = obj["location"]

        #get index for current object
        base_idx = obj_id * 3

        #normalize x,y -> obs
        obs[base_idx] = x
        obs[base_idx + 1] = y

        #object exists
        obs[base_idx + 2] = 1.0

    # bin weights
    obs[12] = bin_weights["first_drawer"]
    obs[13] = bin_weights["second_drawer"]
    obs[14] = bin_weights["gray_bin"]
    obs[15] = bin_weights["white_bin"]

    return obs

model = YOLO(os.path.join(BASE_DIR, "../vla_rl/models/best.pt"))