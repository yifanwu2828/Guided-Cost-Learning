import gym

import pybullet as p
import pybullet_data
import pybullet_envs
import time




def main():
    env = gym.make("HalfCheetahBulletEnv-v0")
    env.render(mode="human")
 
    print(env.action_space)	    	#Box(3,)
    print(env.observation_space)    #Box(15,)

    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            #print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    print(observation)
    print(action)
    env.close()

if __name__ == "__main__":
    main()

