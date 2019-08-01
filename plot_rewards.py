import pickle
import numpy as np
from matplotlib import pyplot as plt

rewards = np.asarray(pickle.load(open('./train_7/rewards_20000.dump', 'rb')))
avg_rewards = []
cal_mean = 100

for i in range (1, len(rewards)):
	if i%cal_mean == 0:
		avg_rewards.append([i, np.mean(rewards[:i,1])])

avg_rewards = np.asarray(avg_rewards)

plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=15)

plt.plot(rewards[:,0],rewards[:,1], label = 'Instantaneous Episode reward')
plt.plot(avg_rewards[:,0], avg_rewards[:,1],'red',label = 'Average episode reward')
plt.legend(loc='best')
plt.xlim([0,21000])
plt.ylim([-300,0])
plt.xlabel('Episodes')
plt.ylabel('Avergage rewards')
plt.title('Average rewards DQN_CNN')

plt.show()