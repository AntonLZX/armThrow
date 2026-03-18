import os
import pybullet as p
import pybullet_data
import time
import numpy as np

# 1. Setup Physics Client
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 2. Load Models
# Get the directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))
urdf_path = os.path.join(dir_path, "arm.urdf")
plane_id = p.loadURDF("plane.urdf")
# Ensure 'arm.urdf' is in the same folder
robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)

# Add the ball (sphere) inside the cup
ball_id = p.loadURDF("sphere_small.urdf", [0, 0, 1.1]) 

# 3. Create "Magnetic" Constraint (The Grasp)
# Connects the last link (cup) to the ball
cid = p.createConstraint(robot_id, 2, ball_id, -1, p.JOINT_FIXED, 
                         [0,0,0], [0,0,0.1], [0,0,0])

# 4. Add Debug Sliders for Manual Testing
sliders = [
    p.addUserDebugParameter("Theta 1 (Base)", -3.14, 3.14, 0),
    p.addUserDebugParameter("Theta 2 (Shoulder)", -1.57, 1.57, 0),
    p.addUserDebugParameter("Theta 3 (Elbow)", -1.57, 1.57, 0),
    p.addUserDebugParameter("RELEASE (Trigger > 0.5)", 0, 1, 0)
]

print("\n--- Manual Control Started ---")
print("Move the sliders to test the arm. Slide 'RELEASE' to 1 to throw.")

released = False

try:
    while True:
        # Get slider values
        targets = [p.readUserDebugParameter(s) for s in sliders]
        
        # Apply positions to joints
        for i in range(3):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, 
                                    targetPosition=targets[i])
        
        # Check for Release Trigger
        if targets[3] > 0.5 and not released:
            p.removeConstraint(cid)
            released = True
            print("Ball Released!")
        
        p.stepSimulation()
        time.sleep(1./240.)

except KeyboardInterrupt:
    p.disconnect()