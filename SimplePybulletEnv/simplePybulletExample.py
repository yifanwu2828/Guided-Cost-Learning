import pybullet as p

#client = p.connect(p.DIRECT) # Can alternatively pass in p.DIRECT
client = p.connect(p.GUI)

import pybullet_data

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -10, physicsClientId=client)
planeId = p.loadURDF("plane.urdf")
huskyId = p.loadURDF("husky/husky.urdf", basePosition=[0,0,0.2])

for _ in range(10000): 
    position, orientation = p.getBasePositionAndOrientation(huskyId)
    p.applyExternalForce(huskyId, 0, [480, 0, 0], position, p.WORLD_FRAME)
    p.stepSimulation()
