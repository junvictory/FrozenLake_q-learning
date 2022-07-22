import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from gym.envs.registration import register


def qGridPrint():
    print('------------------------------------------------------------------------------------')
    print('      {0:.4f}        |      {1:.4f}        |      {2:.4f}        |      {3:.4f}        |'
    .format(q_table[0,3],q_table[1,3],q_table[2,3],q_table[3,3]))
    print('{0:.4f}        {1:.4f}|{2:.4f}        {3:.4f}|{4:.4f}        {5:.4f}|{6:.4f}        {7:.4f}|'
    .format(q_table[0,0],q_table[0,2],q_table[1,0],q_table[1,2],q_table[2,0],q_table[2,2],q_table[3,0],q_table[3,2]))
    print('      {0:.4f}        |      {1:.4f}        |      {2:.4f}        |      {3:.4f}        |'
    .format(q_table[0,1],q_table[1,1],q_table[2,1],q_table[3,1]))
    print('------------------------------------------------------------------------------------')

    print('      {0:.4f}        |      {1:.4f}        |      {2:.4f}        |      {3:.4f}        |'
    .format(q_table[4,3],q_table[5,3],q_table[6,3],q_table[7,3]))
    print('{0:.4f}        {1:.4f}|{2:.4f}        {3:.4f}|{4:.4f}        {5:.4f}|{6:.4f}        {7:.4f}|'
    .format(q_table[4,0],q_table[4,2],q_table[5,0],q_table[5,2],q_table[6,0],q_table[6,2],q_table[7,0],q_table[7,2]))
    print('      {0:.4f}        |      {1:.4f}        |      {2:.4f}        |      {3:.4f}        |'
    .format(q_table[4,1],q_table[5,1],q_table[6,1],q_table[7,1]))
    print('------------------------------------------------------------------------------------')
    
    print('      {0:.4f}        |      {1:.4f}        |      {2:.4f}        |      {3:.4f}        |'
    .format(q_table[8,3],q_table[9,3],q_table[10,3],q_table[11,3]))
    print('{0:.4f}        {1:.4f}|{2:.4f}        {3:.4f}|{4:.4f}        {5:.4f}|{6:.4f}        {7:.4f}|'
    .format(q_table[8,0],q_table[8,2],q_table[9,0],q_table[9,2],q_table[10,0],q_table[10,2],q_table[11,0],q_table[11,2]))
    print('      {0:.4f}        |      {1:.4f}        |      {2:.4f}        |      {3:.4f}        |'
    .format(q_table[8,1],q_table[9,1],q_table[10,1],q_table[11,1]))
    print('------------------------------------------------------------------------------------')
    
    print('      {0:.4f}        |      {1:.4f}        |      {2:.4f}        |      {3:.4f}        |'
    .format(q_table[12,3],q_table[13,3],q_table[14,3],q_table[15,3]))
    print('{0:.4f}        {1:.4f}|{2:.4f}        {3:.4f}|{4:.4f}        {5:.4f}|{6:.4f}        {7:.4f}|'
    .format(q_table[12,0],q_table[12,2],q_table[13,0],q_table[13,2],q_table[14,0],q_table[14,2],q_table[15,0],q_table[15,2]))
    print('      {0:.4f}        |      {1:.4f}        |      {2:.4f}        |      {3:.4f}        |'
    .format(q_table[12,1],q_table[13,1],q_table[14,1],q_table[15,1]))
    print('------------------------------------------------------------------------------------')


# register(
#     id='FrozenLake-v3',
#     entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name':'4x4',
#            'is_slippery':False}
# ) 

# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3
actionArrow = ["←","↓","→","↑"]

num_episodes = 10000

exploration_decay_rate = 0.001
learning_rate = 0.2
discount_rate = 0.99
exploration_rate = 1.0
rList = []
episodeHistory = []

desc = ["SFFF", "FHFH", "FFFF", "HFFG"] # Frozen Lake 상태, [S : 시작, H : Hole, G : Goal]
env = gym.make('FrozenLake-v1', desc = desc,is_slippery=False)
# env = gym.make('Deterministic-4x4-FrozenLake-v0', desc = desc,is_slippery=False)

q_table = np.zeros([env.observation_space.n, env.action_space.n]) # Q Table을 모두 0으로 초기화 한다. : 2차원 (number of state, action space) = (16,4)

for i in range(num_episodes) : 
    state = env.reset()
    rAll = 0
    done = None

    episodeAction =[]
    count = 0
    while not done : 
        # env.render()
        # Explore vs Exploit
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, _ = env.step(action)        # 해당 Action을 했을 때 environment가 변하고, 새로운 state, reward, done 여부를 반환 받음
        
        q_table[state, action] =q_table[state, action]* (1-learning_rate) \
            + learning_rate*(reward + discount_rate * np.max(q_table[new_state, :]))
        # q_table[state, action] = q_table[state, action] + \
        #                         learning_rate * (reward + discount_rate * np.max(q_table[new_state]) - q_table[state, action])
        
        episodeAction.append(actionArrow[action])
        count +=1
        
        if done & (new_state != 15):
            episodeAction.clear()

        rAll += reward
        state = new_state
    if episodeAction != []:
        episodeHistory.append(episodeAction)
    # episodeHistory.append(episodeAction)
    rList.append(rAll)


    ## exploation -
    exploration_rate -= exploration_decay_rate


print("Episode History")
print(episodeHistory)
print("Success rate : "+str(sum(rList) / num_episodes))
print("Final Q-Table Values")
qGridPrint()
# print(q_table)


# plt.bar(range(len(rList)), rList, color="blue")
# plt.show()





for episode in range(3):

    state = env.reset()
    done = False
    print("*****에피소드 ", episode+1, "*****\n\n\n\n")
    for step in range(num_episodes):
        # 현재 상태를 그려 본다.
        # clear_output(wait=True)
        # print(env.render(mode="ansi"))
        env.render()
        # 현재 상태에서의 q값(보상)이 가장 큰 action을 취한다.
        action = np.argmax(q_table[state, :]) 
        # 새로운 action을 취한다
        new_state, reward, done, info = env.step(action)
        if done:
            if reward == 1:
                # 만약에 Goal에 도착하여 reward가 1이라면
                print("****목표에 도달하였습니다.!****")
            else:
                # Goal에 도달하지 못했다면
                print("****Hole에 빠지고 말았습니다.****")
            break
       
        # 새로운 상태를 설정한다.
        state = new_state

env.close()


