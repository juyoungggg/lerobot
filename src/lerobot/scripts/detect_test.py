from ultralytics import YOLO
import numpy as np

def find_zone(x, y, width=640, height=480):
    #define size of zone
    zone_h = height / 3
    zone_w = width / 3

    #find object's row,col
    row = int(y // zone_h)
    col = int(x // zone_w)

    #clip the result (to prevent being row/col 3 which does not exist)
    col = min(col, 2)
    row = min(row, 2)

    #calculate zone
    zone = row * 3 + col
    return zone

def detect_zone(image):
    results = model(image)
    detected_objects = []

    for box in results[0].boxes:
        #get object's class name
        cls = int(box.cls[0])
        name = model.names[cls]

        #get object's box locaion
        x1, y1, x2, y2 = box.xyxy[0]

        #calculate the zone
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        zone = find_zone(int(cx.item()), int(cy.item()))

        print("Name: ", name)
        print("Location: ", int(cx.item()), int(cy.item()))
        print("Zone: ",zone)
        detected_objects.append({
            "name": name,
            "zone": zone
        })
    return detected_objects

def detected_to_obs(detected_objects, num_zones, num_colors, left_total, right_total):
    obs_zone_counts = np.zeros((num_zones, num_colors), dtype=np.int32)

    for obj in detected_objects:
        zone = obj["zone"]
        if obj["name"] == "blue_box":
            color = 0
        elif obj["name"] == "green_box":
            color = 1

        obs_zone_counts[zone, color] += 1

    obs = []
    for zone in range(num_zones):
        for color in range(num_colors):
            obs.append(obs_zone_counts[zone, color])

    bin_diff = abs(left_total - right_total)
    obs.append(left_total)
    obs.append(right_total)
    obs.append(bin_diff)

    return np.array(obs, dtype=np.float32)

model = YOLO("../rltrain/models/best.pt")