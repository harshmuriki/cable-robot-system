import numpy as np
import json

with open("camera.json") as f:
    cam = json.load(f)

extrinsic = np.array(cam["extrinsic"]).reshape(4, 4)
R = extrinsic[:3, :3]
t = extrinsic[:3, 3]

# Camera position in world coordinates
eye = t
# Camera front (view direction): -Z axis of camera in world coordinates
front = -R[:, 2]
# Camera up: Y axis of camera in world coordinates
up = R[:, 1]
# Camera lookat: set to scene center or eye + front
lookat = eye + front

print("front:", front)
print("lookat:", lookat)
print("up:", up)
print("eye (camera position):", eye)