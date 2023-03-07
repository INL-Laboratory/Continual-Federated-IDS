import tensorflow as tf
import numpy as np
import gc
class Agent:
    def __init__(self,id,batch_size,loss_function,shared_model = None):
        self.id = id
        self.batch_size = batch_size # the batch size it should use for each time computing a gradient
        self.current_sample = 0
        self.X = None
        self.y = None
        # we have the premise that there will be 2 main methods that the agents and the controller will communicate:
        # 1) the agents all serve as threads. in this scenario, we will consider a shared model object between all agents
        #   and update the model and it's gradients with a lock.
        # 2) the agents and controller interact through the network. since in this case they might use
        # a variation of protocols, we will live it up to the user to implement their own communication system.

        self.model = shared_model
        self.loss_function = loss_function
        self.epochs = 0 # number of epochs the agent has been working

    def get_data(self,new_x,new_y): # gets the new data that this agent is supposed to train on
        self.X = np.array(new_x)
        self.y = np.array(new_y)

    def update_model(self,updated_model,change_weights = False):
        # when change weights is set to true, we only want to change the weights of our current model (in asynchronous version)
        if not change_weights:
            self.model = updated_model
        else:
            self.model.set_weights(updated_model.get_weights())
            gc.collect()
    def run(self): # computes the gradient
        X_grad = self.X[self.current_sample : self.current_sample + self.batch_size]
        y_grad = self.y[self.current_sample : self.current_sample + self.batch_size]

        with tf.GradientTape() as tape:
            predicted_output = self.model(X_grad)
            loss = self.loss_function(y_grad,predicted_output)
        self.current_sample = self.current_sample + self.batch_size
        if self.current_sample >= len(self.X):
            self.current_sample = 0
            self.epochs += 1
        grads = tape.gradient(loss,self.model.trainable_variables)
        return grads









