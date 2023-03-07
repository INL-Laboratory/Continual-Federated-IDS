# using keras there is no exact way to specify some weights of an layer to be untrainable. ( only layers are able to be defined untrainable)
# thus we put a reguralizer that tends to keep the weights ought to be untrainable close to their original values.

import tensorflow as tf
import numpy as np
from Utils import config


@tf.keras.utils.register_keras_serializable(package='Custom', name='weight stabilizer')
class WStabilizer(tf.keras.regularizers.Regularizer):
    def __init__(self, selected_upper_nodes=None, selected_lower_nodes=None,
                 num_of_all_lower_nodes = 0, num_of_all_upper_nodes = 0,
                 unselected_upper_nodes=None, unselected_lower_nodes=None,
                 coefficient=1, M= config.l1_lambda):

        self.selected_upper_nodes = selected_upper_nodes
        self.selected_lower_nodes = selected_lower_nodes
        self.unselected_upper_nodes = unselected_upper_nodes
        self.unselected_lower_nodes = unselected_lower_nodes
        self.num_of_all_lower_nodes = num_of_all_lower_nodes
        self.num_of_all_upper_nodes = num_of_all_upper_nodes



        self.coefficient = coefficient  # coefficient for stabilizer
        self.M = M  # coefficient for l2 norm

        # a tensor matrix that consists of original weights in their indices and 0 in the other parts
        self.original_weights = np.zeros(shape=[self.num_of_all_lower_nodes, self.num_of_all_upper_nodes])
        # a tensor matrix that will be multiplied with the inputs
        self.multiplication_matrix = np.zeros(shape=[self.num_of_all_lower_nodes, self.num_of_all_upper_nodes])
        self.multiplication_matrix[:] = self.M * self.M
        # the rest of the configuration of  these two matrices will be done in the function below





    def configure_weight_matrix(self,weights):
        # creating
        unselected_upper_nodes =  self.find_unselected_nodes(1)
        unselected_lower_nodes = self.find_unselected_nodes(0)
        print("&&&&&&&&&&&&&&&&&&")
        print(unselected_upper_nodes)
        
        print(self.selected_upper_nodes)
        print(self.selected_lower_nodes)
        print(unselected_lower_nodes)
        print("configuring reguralizer started")
        coef2 = self.coefficient * self.coefficient
        for upper_node in unselected_upper_nodes:
            for lower_node in range(self.num_of_all_lower_nodes):
                # since we want to calculate the l2 norm, we will multiply the square of coefficient with the original weight
                self.original_weights[lower_node,upper_node] = coef2 * weights[lower_node,upper_node]
                self.multiplication_matrix[lower_node,upper_node] = coef2
        for lower_node in unselected_lower_nodes:
            for upper_node in range(self.num_of_all_upper_nodes):
                # since we want to calculate the l2 norm, we will multiply the square of coefficient with the original weight
                self.original_weights[lower_node,upper_node] = coef2 * weights[lower_node,upper_node]
                self.multiplication_matrix[lower_node,upper_node] = coef2
        self.original_weights = tf.constant(self.original_weights,dtype= np.float)
        self.multiplication_matrix = tf.constant(self.multiplication_matrix,dtype= np.float)
        #print( self.original_weights)


    def find_unselected_nodes(self,which = 0):
        unselected_listed_nodes = []
        if which == 0:
            listed_nodes = self.selected_lower_nodes
            number_of_nodes = self.num_of_all_lower_nodes
        else:
            listed_nodes = self.selected_upper_nodes
            number_of_nodes = self.num_of_all_upper_nodes
        if listed_nodes[0] != 0:
            for missing in range(0, listed_nodes[0]):
                    unselected_listed_nodes.append(missing)
        for index in range(len(listed_nodes) - 1):
            if listed_nodes[index + 1] - listed_nodes[index] != 1:
                for missing in range(listed_nodes[index] + 1, listed_nodes[index + 1]):
                    unselected_listed_nodes.append(missing)
        if listed_nodes[-1] != number_of_nodes - 1:
            for missing in range(listed_nodes[-1] + 1, number_of_nodes ):
                unselected_listed_nodes.append(missing)
        return unselected_listed_nodes




    def __call__(self, x):
        return tf.nn.l2_loss(tf.multiply(self.multiplication_matrix,x) - self.original_weights)
        # print("call1")
        # unchanged_extracted_part = self.extract__sub_weight_matrix(weight_matrix=x, selected=False)
        # print("call2")
        # changed_extracted_part = self.extract__sub_weight_matrix(weight_matrix=x, selected=True)
        # print("call3")
        # return self.coefficient * tf.nn.l2_loss(unchanged_extracted_part - self.original_weights) + \
        #        self.M * tf.nn.l2_loss(changed_extracted_part)



    def get_config(self):
        return {'l2': float(self.coefficient)}




@tf.keras.utils.register_keras_serializable(package='Custom', name='bias stabilizer')
class BStabilizer(tf.keras.regularizers.Regularizer):
    def __init__(self, original_biases=None, selected_upper_nodes=None,num_of_all_biases = 0,unselected_upper_nodes=None, coefficient=0.5,
                 M=0.001):

        self.selected_upper_nodes = selected_upper_nodes
        self.unselected_upper_nodes = unselected_upper_nodes
        self.num_of_all_biases = num_of_all_biases
        self.coefficient = coefficient  # coefficient for stabilizer
        self.M = M  # coefficient for l2 norm
        # a row tensor with the original biases and 0 in other indices
        self.biases = np.zeros(shape=(self.num_of_all_biases,))
        self.coefficient_matrix = np.zeros(shape=(self.num_of_all_biases,))
        self.coefficient_matrix[:] = self.M * self.M

    def set_bias_configuration(self,weights):
        unselected_nodes = self.find_unselected_nodes()
        coef2 = self.coefficient * self.coefficient
        for node in unselected_nodes:
            self.biases[node] = coef2 * weights[node]
            self.coefficient_matrix [node] = coef2
        self.biases = tf.constant(self.biases, dtype= np.float)
        self.coefficient_matrix = tf.constant(self.coefficient_matrix, dtype= np.float)

    def find_unselected_nodes(self):
        unselected_listed_nodes = []
        listed_nodes = self.selected_upper_nodes
        number_of_nodes = self.num_of_all_biases
        for index in range(len(listed_nodes) - 1):
            if listed_nodes[index + 1] - listed_nodes[index] != 1:
                for missing in range(listed_nodes[index] + 1, listed_nodes[index + 1]):
                    unselected_listed_nodes.append(missing)
        if listed_nodes[-1] != number_of_nodes - 1:
            for missing in range(listed_nodes[-1] + 1, number_of_nodes ):
                unselected_listed_nodes.append(missing)
        return unselected_listed_nodes

    def __call__(self, x):
        return tf.nn.l2_loss(tf.multiply(x,self.coefficient_matrix) - self.biases)
        # unchanged_extracted_part = self.extract__sub_weight_matrix(weight_matrix=x, selected=False)
        # changed_extracted_part = self.extract__sub_weight_matrix(weight_matrix=x, selected=True)
        # return self.coefficient * tf.nn.l2_loss(unchanged_extracted_part - self.original_weights) + \
        #        self.M * tf.nn.l2_loss(changed_extracted_part)

    def get_config(self):
        return {'l2': float(self.coefficient)}











