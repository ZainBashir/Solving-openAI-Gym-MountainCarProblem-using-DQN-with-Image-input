########################3
#  This is script to test your trained model. Load your trained model and its weights using the
# load_model function given in the Keras API and use it to make prediction of what actions to take
# in any given state so that the car can successfully reach its target

import gym
import tensorflow as tf

#important to use the same keras version for loading the model as used for training/saving the model. I used 2.0.8
#and tested on 2.2.4 which gave me an error of not being able to load the saved model correctly.
import keras
import numpy as np
import cv2
import collections
from time import sleep
import os




 # The following lines make sure that only CPU is used to run this script. If you have installed
 # tensorflow for GPU the python script will detect it automatically by default and use GPU. To avoid
 # that I have used this
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


# Since we trained our model with the image format as channels first we need to do the same for
# evaluation
keras.backend.set_image_data_format('channels_first')

# Process each frame before making any prediction: RGB to Gray, rescale, normalize
def process_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (150, 100))
    image = np.float32(np.true_divide(image,255))
    return image

# Function to use the the model to predict an action
def greedy_action(current_state):
    action_mask = np.ones((1, 3))
    q_values = learnt_model.predict_on_batch([current_state, action_mask])[0]
    action = np.argmax(q_values)

    return action


# Load the trained model. Change the path according to the location of your model file
learnt_model = keras.models.load_model('./train_7/DQN_CNN_model_20000.h5',
                 custom_objects = {"huber_loss":tf.losses.huber_loss})


# Initialize environment and set all the required parameters
env = gym.make('MountainCar-v0').env
seq_memory = collections.deque(maxlen=4)
done = False
steps = 0
frame_skip = 4
stack_depth = 4
time_steps = 300
episodes = 5

# Added this flag just for debugging. Can always remain True
evaluate = True


# The main loops: outer one for each episode and the inner one to loop over each step of the episode
# The same as the ones in the main script except that I do no take any random actions here
if evaluate:
    for episode in range(1,episodes):
        seq_memory.clear()
        initial_state = env.reset()
        current_image = env.render(mode = 'rgb_array')
        frame = process_image(current_image)
        frame = frame.reshape(1, frame.shape[0], frame.shape[1])
        current_state = np.repeat(frame, stack_depth, axis=0)
        seq_memory.extend(current_state)
        episode_reward = 0
        for time in range(time_steps):
            if time % frame_skip == 0:
                if np.random.rand() <= 0:
                    action = np.int(np.random.choice([0, 2], 1))
                else:
                    action = greedy_action(current_state.reshape(1, current_state.shape[0] \
                                         , current_state.shape[1], current_state.shape[2]))
                
            next_pos, reward, done, _ = env.step(action)
            next_frame = env.render(mode='rgb_array')
            next_frame = process_image(next_frame)
            seq_memory.append(next_frame)
            next_state = np.asarray(seq_memory)
    
            current_state = next_state

            # just to clearly visualize the car. Otherwise the car moves too fast
            sleep(0.01)
            episode_reward = episode_reward + reward
            if done:
                print(done)
                break
    
        print('Episode {} completed in  {} time steps'.format(episode, time))
env.close()
