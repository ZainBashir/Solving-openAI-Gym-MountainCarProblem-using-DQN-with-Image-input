#                        Solving-openAI-Gym-MountainCarProblem-using-DQN-with-Image-input
I have targeted to solve the a benchmark problem in Reinforcement learning literature using Deep Q-networks with images as the only input to the model. Keras was used to model the Convolutional neural network which predicts the best action to take in a given state. I was unable to find a comprehensive tutorial which implements the the DQN algorithm to solve the mountain car problem with images as the only input to the CNN model. I took help from a lot of different tutorials to get my code working and will reference all of them below.

## Problem setup
A car which starts at the bottom of the mountain has to climb up the mountain but doesn't have enough engine power to take it to the top without taking help from gravity. Therefore the car has to learn to go left and take help from gravity as well as its engine to reach its goal. The observation space is continous with position values ranging from -1.2 to 0.5. The action space is discreet and the car can either move left (1), do nothing (0) or move right (2).

## CNN model
The following is a graphical description of the CNN model 


![alt](images-tutorial/model.png)

Our model takes in two inputs; a stack of 4 gray scale images and an action mask. The action mask multiples with the output of our model. This encodes our Q-values in a one-hot style with the hot value corresponidng to the action index.

```
        input_shape = (self.stack_depth, self.image_height, self.image_width)
        actions_input = layers.Input((self.num_actions,), name = 'action_mask')
```
Thus the input layer is defined as the size [None,4,100,150]. The None here signifies the unkown batch size we are going to
feed our model.

 ```
        frames_input = layers.Input(input_shape, name='input_layer')
 ```
 
In my case I have set the image format in Keras backend as 'Channels first' (no particular reason)

```
        from keras import backend as K
        keras.backend.set_image_data_format('channels_first')
```

Next we define the core layers of our model exactly the same parameters as defined in the original DQN paper. I took help from this very nice tutorial to define my model and also use the action mask technique to vectorize my code (https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26)

```
        conv_1 = layers.Conv2D(32, (8,8), strides=4, padding ='same'\
        ,activation = 'relu', name='conv_1',kernel_initializer='glorot_uniform',bias_initializer='zeros')(frames_input)

        conv_2 = layers.Conv2D(64, (4,4), strides=2, padding='same', activation='relu',name='conv_2'\
           ,kernel_initializer='glorot_uniform',bias_initializer='zeros')(conv_1)

        conv_3 = layers.Conv2D(64, (3,3), strides=1, padding='same',name='conv_3', activation='relu'\
           ,kernel_initializer='glorot_uniform',bias_initializer='zeros')(conv_2)

        flatten_1 = layers.Flatten()(conv_3)

        dense_1 = layers.Dense(512, activation='relu', name='dense_1',
            kernel_initializer='glorot_uniform',bias_initializer='zeros')(flatten_1)
            
        output = layers.Dense(self.num_actions, activation='linear', name='output',
            kernel_initializer='glorot_uniform',bias_initializer='zeros')(dense_1)
```
