# Simple PyBullet Example 
This guide assumes rudimentary knowledge of the structure of OpenAI Gym environments, along with proficiency in Python.

## Pybullet Installation
[PyBullet](https://github.com/bulletphysics/bullet3) Installation is simple:
```
pip3 install pybullet --upgrade --user
python3 -m pybullet_envs.examples.enjoy_TF_AntBulletEnv_v0_2017may
python3 -m pybullet_envs.examples.enjoy_TF_HumanoidFlagrunHarderBulletEnv_v1_2017jul
python3 -m pybullet_envs.deep_mimic.testrl --arg_file run_humanoid3d_backflip_args.txt
```
Make sure you checkout the [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3) for more detail.

Reading documentation will be crucial in becoming comfortable with PyBullet.

## Hello PyBullet
A physics simulation is established by calling p.connect(). With p.GUI, a new window will be created for visualization and debugging purpose. Alternatively, p.DIRECT provides the fastest connection without visulization.


Both GUI and DIRECT connections will execute the physics simulation and rendering in the same process as PyBullet.


Other connection modes such as `SHARED_MEMORY`, `UDP`, `TCP GUI_SERVER`, `SHARED_MEMORY_SERVER`, `SHARED_MEMORY_GUI` are described in [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3)


Notice that we have only a single physics simulation running. `p.connect` returns an integer defaults to zero. 

```
import pybullet as p

#client = p.connect(p.DIRECT) # Can alternatively pass in p.DIRECT
client = p.connect(p.GUI)
```

You can provide your own data files, or you can use the PyBullet_data package that ships with PyBullet.
For this, `import pybullet_data` and register the directory using `pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())`.

```
import pybullet_data

p.setAdditionalSearchPath(pybullet_data.getDataPath())
```

P.loadURDF allows you to create a robot as specified in a URDF file. It returns an integer which is the ID passed into other function to query the state of a robot and perform actions on it.


We created a plane and a racercar model in our physics simulation. The basePostion of racercar in 3D is [x,y,z] = [0,0,0.2]. By using `p.setGravity(0, 0, -10, physicsClientId=client)`, we initiate the gravity in physical simulation who's Id = client.
```
p.setGravity(0, 0, -10, physicsClientId=client)
planeId = p.loadURDF("plane.urdf")
racecarId = p.loadURDF("racecar/racecar.urdf", basePosition=[0,0,0.2])
```

Now we are able to create our first PyBullet simulation. A racer car will fall to the plane and the simulation will end shortly.
We will run the simulation for 10000 time steps. Each time steps is 1/240 of a second.

```
for _ in range(10000): 
    p.stepSimulation()
```
We can do some modification in for loop to move the racecar. `p.getBasePositionAndOrientation(racecarId)` will return the current position and orientation of the base (or root link) of the body in Cartesian world coordinates. The orientation is a quaternion in [x,y,z,w] format.


```
for _ in range(10000): 
    position, orientation = p.getBasePositionAndOrientation(racecarId)
    p.applyExternalForce(racecarId, 0, [28, 0, 0], position, p.WORLD_FRAME)
    p.stepSimulation()
```
The car should move a little bit forward in the simulation by applying `p.applyExternalForce()` Note that this method will only work when explicitly stepping the simulation using stepSimulation.


In general, applying an external force to move the robot is not a good practice. It is better to apply a force to the axle of the car.


