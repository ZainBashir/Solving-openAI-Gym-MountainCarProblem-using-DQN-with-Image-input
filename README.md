#                        Solving-openAI-Gym-MountainCarProblem-using-DQN-with-Image-input
I have targeted to solve the a benchmark problem in Reinforcement learning literature using Deep Q-networks with images as the only input to the model. Keras was used to model the Convolutional neural network which predicts the best action to take in a given state. I was unable to find a comprehensive tutorial which implements the the DQN algorithm to solve the mountain car problem with images as the only input to the CNN model. I took help from a lot of different tutorials to get my code working and will reference all of them below.

## Problem setup
A car which starts at the bottom of the mountain has to climb up the mountain but doesn't have enough engine power to take it to the top without taking help from gravity. Therefore the car has to learn to go left and take help from gravity as well as its engine to reach its goal.

## CNN model
The following is a graphical description of the CNN model

