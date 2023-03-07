
import tensorflow as tf
import numpy as np

class Policy_Estimator:
    def __init__(self,state_space = 10, hidden_size = 128,num_layers = 1, number_of_actions = 3,
                 learning_rate = 0.001):
        self.state_space = state_space # maximum number of nodes added to a a layer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.number_of_actions = number_of_actions


        # building the model
        # since we will have to generate a sequence by using the output of the current time step as the input of the next,
        # we will create to separate models:
        # 1) an lstm models that is comprised of stacked lstm cells (not the default keras layers.LSTM)
        # 2) a dense model which takes the lstm's output as input and turns into a probability vector using softmax
        # when inferring and training, we will make use of a loop and gradient tape
        cells = [tf.keras.layers.LSTMCell(self.hidden_size) for _ in range(self.num_layers)]
        self.lstm_cell = tf.keras.layers.StackedRNNCells(cells)
        self.dense_model = tf.keras.Sequential([tf.keras.Input(shape=(self.hidden_size,)),
                                            tf.keras.layers.Dense(state_space,activation=tf.nn.softmax)])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)


    def update(self,state,r_minus_v,actions):
        probabilities = state
        states = ([tf.keras.backend.variable(np.zeros(shape=(1,self.hidden_size))),tf.keras.backend.variable(np.zeros(shape=(1,self.hidden_size)))],)
        all_probs = []
        with tf.GradientTape() as tape:
            for time_step in range(self.number_of_actions):
                probabilities,states = self.lstm_cell(probabilities,states)
                probabilities = self.dense_model(probabilities)
                all_probs.append(probabilities)
            picked_action_prob = 0
            for time_step in range(self.number_of_actions):
                if time_step == 0:
                    picked_action_prob = all_probs[time_step][0,actions[time_step]]
                else:
                    picked_action_prob = picked_action_prob * all_probs[time_step][0,actions[time_step]]
            loss = -tf.math.log(picked_action_prob) * r_minus_v
        trainable_variables = []
        trainable_variables += self.lstm_cell.trainable_variables
        trainable_variables += self.dense_model.trainable_variables
        grads = tape.gradient(loss,trainable_variables)
        self.optimizer.apply_gradients(zip(grads,trainable_variables))

    def predict(self,state):
        probabilities = state
        states = ([tf.keras.backend.variable(np.zeros(shape=(1, self.hidden_size))),
                   tf.keras.backend.variable(np.zeros(shape=(1, self.hidden_size)))],)
        all_probs = []
        for time_step in range(self.number_of_actions):
            probabilities, states = self.lstm_cell(probabilities, states)
            probabilities = self.dense_model(probabilities)
            all_probs.append(probabilities)
        return all_probs



class Value_estimator:
    def __init__(self,state_space = 10,learning_rate = 0.0005 ):
        self.state_space = state_space
        self.dense_network = tf.keras.Sequential([tf.keras.layers.Input(shape=(state_space,)),
                                                  tf.keras.layers.Dense(1)])
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)



    def update(self,state,reward):
        state = tf.reshape(state,(1,self.state_space))
        with tf.GradientTape() as tape:
            estimated_value = self.dense_network(state)
            estimated_value = tf.squeeze(estimated_value)
            loss = tf.math.squared_difference(estimated_value,reward)
        grads = tape.gradient(loss, self.dense_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.dense_network.trainable_variables))
    def predict(self,state):
        state = tf.reshape(state, (1, self.state_space))
        estimated_value = self.dense_network(state)
        return tf.squeeze(estimated_value)




class Controller_Network:
    def __init__(self,state_space = 10, lstm_hidden_size = 128,lstm_num_layers = 1, number_of_actions = 3,
                 learning_rate_policy = 0.001,learning_rate_value = 0.0005):
        self.state_space = state_space
        self.number_of_actions = number_of_actions
        self.state = np.random.random(size=(1, state_space))
        self.state_space = state_space
        self.policy_estimator = Policy_Estimator(state_space = state_space, hidden_size = lstm_hidden_size,
                                                 num_layers = lstm_num_layers,number_of_actions = number_of_actions,learning_rate = learning_rate_policy)
        self.value_estimator = Value_estimator(state_space = state_space,learning_rate=learning_rate_value)

    def train(self,reward):
        v = self.value_estimator.predict(self.state)
        r_minus_v = reward - v
        self.value_estimator.update(state= self.state, reward = reward)
        self.policy_estimator.update(state= self.state,r_minus_v = r_minus_v,actions= self.actions)

    def get_actions(self):
        action_probs = self.policy_estimator.predict(self.state)
        self.actions = []
        for i in range(self.number_of_actions):
            prob = action_probs[i][0]
            prob = np.array(prob)
            prob /= prob.sum()
            action = np.random.choice(np.arange(self.state_space), p=prob)
            self.actions.append(action)
        return self.actions









