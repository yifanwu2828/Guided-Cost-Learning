import pybullet as p
import pybullet_data
import time

# start pybullet simulation
p.connect(p.GUI)

# reset the simulation to its original state
p.resetSimulation()

# load urdf file path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# import plane
plane = p.loadURDF("plane.urdf")

# import Kuka urdf and fix it to the ground
robot = p.loadURDF("kuka_experimental/kuka_kr210_support/urdf/kr210l150.urdf", [0, 0, 0], useFixedBase=1)

# request the position and orientation of the robot
position, orientation = p.getBasePositionAndOrientation(robot)
print("The robot position is {}".format(position))
print("The robot orientation (x, y, z, w) is {}".format(orientation))

# print the number of joints of the robot
nb_joints = p.getNumJoints(robot)
print("The robot is made of {} joints.".format(nb_joints))
print("The arm does not really have 8 joints. It has 6 revolute joints and 2 fixed joints.")

# print information about joint 2
joint_index = 2
joint_info = p.getJointInfo(robot, joint_index)
print("Joint index: {}".format(joint_info[0]))
print("Joint name: {}".format(joint_info[1]))
print("Joint type: {}".format(joint_info[2]))
print("First position index: {}".format(joint_info[3]))
print("First velocity index: {}".format(joint_info[4]))
print("flags: {}".format(joint_info[5]))
print("Joint damping value: {}".format(joint_info[6]))
print("Joint friction value: {}".format(joint_info[7]))
print("Joint positional lower limit: {}".format(joint_info[8]))
print("Joint positional upper limit: {}".format(joint_info[9]))
print("Joint max force: {}".format(joint_info[10]))
print("Joint max velocity {}".format(joint_info[11]))
print("Name of link: {}".format(joint_info[12]))
print("Joint axis in local frame: {}".format(joint_info[13]))
print("Joint position in parent frame: {}".format(joint_info[14]))
print("Joint orientation in parent frame: {}".format(joint_info[15]))
print("Parent link index: {}".format(joint_info[16]))

# print state of joint 2
joints_index_list = range(nb_joints)
joints_state_list = p.getJointStates(robot, joints_index_list)

print("Joint position: {}".format(joints_state_list[joint_index][0]))
print("Joint velocity: {}".format(joints_state_list[joint_index][1]))
print("Joint reaction forces (Fx, Fy, Fz, Mx, My, Mz): {}".format(joints_state_list[joint_index][2]))
print("Torque applied to joint: {}".format(joints_state_list[joint_index][3]))

# print state of link 2
link_state_list = p.getLinkState(robot, 2)
print("Link position (center of mass): {}".format(link_state_list[0]))
print("Link orientation (center of mass): {}".format(link_state_list[1]))
print("Local position offset of inertial frame: {}".format(link_state_list[2]))
print("Local orientation offset of inertial frame: {}".format(link_state_list[3]))
print("Link frame position: {}".format(link_state_list[4]))
print("Link frame orientation: {}".format(link_state_list[5]))

# Define gravity in x, y and z
p.setGravity(0, 0, -9.81)

# define a target angle position for each joint (note, you can also control by velocity or torque)
p.setJointMotorControlArray(robot, joints_index_list, p.POSITION_CONTROL, targetPositions=[-1, 0, -0.5, 1, 1, 0, 0, 0])  

joint_index = 2
# step through the simulation
for _ in range(1000000):
    p.stepSimulation()
    time.sleep(1./30.)  # slow down the simulation

    joints_state_list = p.getJointStates(robot, joints_index_list)
    print("Joint position: {}".format(joints_state_list[joint_index][0]))
    print("Joint velocity: {}".format(joints_state_list[joint_index][1]))
    print("Torque applied to joint: {}".format(joints_state_list[joint_index][3]))


p.disconnect()