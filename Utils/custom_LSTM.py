# An LSTM network architecture that we designed specifically for this experiment. The feature of this network is that
# it is able to process the sequences one by one, were as in the lstm layer in tensorflow the all the sequences have to be fed
# to the network as the input
import tensorflow as tf
from Utils import config as config


class LSTM_Network(tf.keras.Model):
    def __init__(self,input_size,LSTM_units,dense_units, lstm_weights = None):
        super(LSTM_Network, self).__init__()
        self.input_size = input_size # size of the input_vector
        self.LSTM_units = LSTM_units # a list containing the size of hidden unit in each of the LSTMs
        self.num_of_LSTM_units = len(LSTM_units)
        self.dense_units = dense_units # a list containing the size of each of the dense layers
        self.num_of_dense_layers = len(dense_units)

        # building initial model
        cells = [tf.keras.layers.LSTMCell(self.LSTM_units[i]) for i in range(self.num_of_LSTM_units)]
        if lstm_weights is None:
            self.lstm = tf.keras.layers.StackedRNNCells(cells)
        else:
            self.lstm = tf.keras.layers.StackedRNNCells(cells, weights= lstm_weights)
        self.lstm_state = None

        dense_layers= [tf.keras.Input(shape=(self.LSTM_units[-1],))]
        for i in range(self.num_of_dense_layers - 1):
            dense_layers.append(tf.keras.layers.Dense(self.dense_units[i], activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1(
                config.l1_lambda),
                                                      bias_regularizer=tf.keras.regularizers.l1(config.l1_lambda)))
            
        dense_layers .append(tf.keras.layers.Dense(self.dense_units[-1], activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l1(
            config.l1_lambda),
                                                   bias_regularizer=tf.keras.regularizers.l1(config.l1_lambda)))
        self.dense_model = tf.keras.Sequential(dense_layers)

    def reset_state(self):
        self.lstm_state = None


    def call(self, inputs, training=None, mask=None, return_complete = False, reset_at_end = True):
        if self.lstm_state is None: # we want the current sequence to be independent of the last one
            self.lstm_state = self.lstm.get_initial_state(inputs)
        state = self.lstm_state
        finals = []
        outputs = None
        for i in range(inputs.shape[1]):
            # print(i)
            outputs, state = self.lstm(inputs[:, i], state)
            # print(outputs.shape)
            outputs = self.dense_model(outputs)
            finals.append(outputs)
        if reset_at_end:
            self.reset_state()
        else:
            self.lstm_state = state
        # todo fix the line below
        # return tf.convert_to_tensor(np.array(finals))
        if return_complete:
            return finals
        return outputs
    def copy(self):
        clone_model = LSTM_Network(input_size= self.input_size,
                                   LSTM_units= self.LSTM_units,
                                   dense_units= self.dense_units,
                                   lstm_weights= self.lstm.get_weights())
        #clone_model.lstm.set_weights(self.lstm.get_weights())
        clone_model.dense_model = tf.keras.models.clone_model(self.dense_model)
        clone_model.dense_model.set_weights(self.dense_model.get_weights())
        return clone_model

    def call_lstm(self, inputs, reset_at_end = True, return_complete = False): # runs only the lstm part of the model on given input
        if self.lstm_state is None: # we want the current sequence to be independent of the last one
            self.lstm_state = self.lstm.get_initial_state(inputs)
        state = self.lstm_state
        finals = []
        outputs = None
        for i in range(inputs.shape[1]):
            # print(i)
            outputs, state = self.lstm(inputs[:, i], state)
            # print(outputs.shape)
            finals.append(outputs)
        if reset_at_end:
            self.reset_state()
        else:
            self.lstm_state = state
        if return_complete:
            return finals
        return outputs
        
    def load(self, path):
        input_model = tf.keras.models.load_model(path)
        self.lstm = input_model.lstm
        self.dense_model = input_model.dense_model



