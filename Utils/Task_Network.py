import tensorflow as tf
import numpy as np
from tqdm import tqdm
from time import time

from Utils import config
from Utils.Sregulizer import WStabilizer
from Utils.Control_NetWork import Controller_Network
import gc

# in addition to having the abillity to expand the network as explained in the paper, this class
# has the abillity to find the best expansion by using reinforcment learning (similar to RCL) using the with_reinforced arguemtn
# Note that in the paper, we didn't use the rcl version
# Furthermore, for only training the nodes that have been added to the network, we have used and Wstabilizer
# For more info, refer to the Sregulizer file

class Task_Nerwork:
    def __init__(self, number_of_conv2ds=2, conv2d_filters=[16, 32], number_of_dense_layers=3, kernel_size=3,
                 num_output_classes=2, nk=10, dynamic_coeff=0.001,
                 path_to_report='fully_trained.json', penalty=0.0001,
                 controller_epochs=20, num_lstm_layers=1, fisher_compress_coeff=0.2, max_fisher_samples=1000,
                 number_of_train_samples=0):
        self.weights = {}  # we will preserve our parameters in this dictionary of ndarrays through time
        self.T = 1
        self.number_of_conv2ds = number_of_conv2ds
        self.conv2d_filters = conv2d_filters
        self.number_of_dense_layers = number_of_dense_layers
        self.num_output_classes = num_output_classes
        self.cnn_network = None  # we will use a fix cnn network and alter the dense network with new data
        if num_output_classes == 2:
            self.mode = 'binary'
        else:
            self.mode = 'multi-class'
        self.kernel_size = kernel_size
        # weight below this value will be considered as zero
        self.nk = nk
        self.dynamic_coeff = dynamic_coeff
        self.reports = []  # array of jsons
        self.path_to_report = path_to_report
        self.penalty = penalty
        self.controller_epochs = controller_epochs
        self.num_lstam_layers = num_lstm_layers

        self.fisher_compress_coeff = fisher_compress_coeff  # the coefficient of the fisher term in our loss when compressing
        self.fisher_matrix = []
        self.max_fisher_samples = max_fisher_samples  # maximum number of samples we use when computing fisher matrix
        # self.max_fisher = 1
        self.number_of_train_samples = number_of_train_samples  # number of attack samples the model has been trained on

    def initialize_weights(self, path="base-cnn",
                           input_model=None):  # loads the weights from our base model to the weights
        if input_model is None:
            reconstructed_model = tf.keras.models.load_model(path)
        else:
            reconstructed_model = input_model
        # getting the weights for our conv2d layers
        for i in range(self.number_of_conv2ds):
            number_of_filters = reconstructed_model.layers[i].trainable_variables[0].shape[-1]
            number_of_channels = reconstructed_model.layers[i].trainable_variables[0].shape[-2]
            self.weights['conv' + str(i)] = reconstructed_model.layers[i].trainable_variables[0].numpy(). \
                reshape(number_of_filters, number_of_channels, self.kernel_size, self.kernel_size)
            self.weights['bconv' + str(i)] = reconstructed_model.layers[i].trainable_variables[1].numpy()  # biases

        self.cnn_network = self.build_cnn_model()

        # getting the weights for dense layers
        for i in range(self.number_of_conv2ds + 1,
                       self.number_of_conv2ds + 1 + self.number_of_dense_layers + 1):  # the + 1 is for flatten layer
            num_inputs = reconstructed_model.layers[i].trainable_variables[0].shape[-2]
            self.weights['dense' + str(i)] = reconstructed_model.layers[i].trainable_variables[0].numpy()
            self.weights['bdense' + str(i)] = reconstructed_model.layers[i].trainable_variables[1].numpy()  # biases

    def update_fisher_matrix(self, model, controller):
        if model is None:
            model = self.build_expanded_dense_netowrk(actions=[], fix_previous=False)
        self.fisher_matrix = self.compute_fisher_matrix(model, controller)

    def build_cnn_model(self):
        inputs = tf.keras.Input(shape=(config.flow_size - 1, config.pkt_size, 1))
        inputs_of_conv = inputs

        for i in range(self.number_of_conv2ds):
            number_of_inputs = self.weights['conv' + str(i)].shape[0]
            kernels = tf.keras.backend.variable(
                self.weights['conv' + str(i)].reshape(self.kernel_size, self.kernel_size, -1, number_of_inputs))
            biases = tf.keras.backend.variable(self.weights['bconv' + str(i)])

            conv_prev = tf.keras.layers.Conv2D(filters=self.weights['conv' + str(i)].shape[0],
                                               kernel_size=self.kernel_size, activation=tf.nn.relu,
                                               kernel_initializer=tf.keras.initializers.Constant(kernels),
                                               bias_initializer=tf.keras.initializers.Constant(biases)
                                               )
            conv_prev.trainable = False
            inputs_of_conv = conv_prev(inputs_of_conv)
        input_of_dense = tf.keras.layers.Flatten()(inputs_of_conv)
        return tf.keras.Model(inputs=inputs, outputs=input_of_dense, name='fix_cnn')

    def save_dense_weights(self, model):
        for i in range(self.number_of_conv2ds + 1,
                       self.number_of_conv2ds + 1 + self.number_of_dense_layers + 1):  # the + 1 is for flatten layer
            # index_of_weight = weight_layers_in_model[i - (self.number_of_conv2ds + 1)]
            print(i)
            index_of_weight = i - (
                self.number_of_conv2ds)  # start index should be from 1 instead of 0 because the first layer is input layer
            print(type(model.layers[index_of_weight]))
            print(type(model.layers[index_of_weight].weights[0]))
            self.weights['dense' + str(i)] = model.layers[index_of_weight].weights[0].numpy()
            self.weights['bdense' + str(i)] = model.layers[index_of_weight].weights[1].numpy()  # biases

    def save_expanded_weights(self, model):
        weight_layers_in_model = [3 * (i + 1) for i in range(self.number_of_dense_layers)]
        weight_layers_in_model.append(len(model.layers) - 1)
        for i in range(self.number_of_conv2ds + 1,
                       self.number_of_conv2ds + 1 + self.number_of_dense_layers + 1):  # the + 1 is for flatten layer
            index_of_weight = weight_layers_in_model[i - (self.number_of_conv2ds + 1)]
            if i == self.number_of_conv2ds + 1 + self.number_of_dense_layers:

                self.weights['dense' + str(i)] = model.layers[index_of_weight].weights[0].numpy()
                self.weights['bdense' + str(i)] = model.layers[index_of_weight].weights[1].numpy()  # biases
            else:
                self.weights['dense' + str(i)] = np.concatenate((model.layers[index_of_weight - 2].weights[0].numpy(),
                                                                 model.layers[index_of_weight - 1].weights[0].numpy()),
                                                                axis=1)
                self.weights['bdense' + str(i)] = np.concatenate((model.layers[index_of_weight - 2].weights[1].numpy(),
                                                                  model.layers[index_of_weight - 1].weights[1].numpy()),
                                                                 axis=0)  # biases

    def build_complete_model(self, trainable = False):
        inputs = tf.keras.Input(shape=(config.flow_size - 1, config.pkt_size, 1))
        inputs_of_conv = inputs

        for i in range(self.number_of_conv2ds):
            number_of_inputs = self.weights['conv' + str(i)].shape[0]
            number_of_channels = self.weights['conv' + str(i)].shape[1]
            kernels = tf.keras.backend.variable(
                self.weights['conv' + str(i)].reshape(self.kernel_size, self.kernel_size, -1, number_of_inputs))
            biases = tf.keras.backend.variable(self.weights['bconv' + str(i)])
            conv_prev = tf.keras.layers.Conv2D(filters=self.weights['conv' + str(i)].shape[0],
                                               kernel_size=self.kernel_size, activation=tf.nn.relu,
                                               kernel_initializer=tf.keras.initializers.Constant(kernels),
                                               bias_initializer=tf.keras.initializers.Constant(biases)
                                               )
            conv_prev.trainable = trainable
            inputs_of_conv = conv_prev(inputs_of_conv)
        # flatten layer
        input_of_dense = tf.keras.layers.Flatten()(inputs_of_conv)

        for i in range(self.number_of_conv2ds + 1, self.number_of_conv2ds + 1 + self.number_of_dense_layers):
            number_of_nodes = self.weights['dense' + str(i)].shape[1]
            numer_of_prev_layer_nodes = self.weights['dense' + str(i)].shape[0]
            kernels = tf.keras.backend.variable(self.weights['dense' + str(i)])
            biases = tf.keras.backend.variable(self.weights['bdense' + str(i)])
            dense_prev = tf.keras.layers.Dense(self.weights['dense' + str(i)].shape[1], activation=tf.nn.relu,
                                               kernel_initializer=tf.keras.initializers.Constant(kernels),
                                               bias_initializer=tf.keras.initializers.Constant(biases)
                                               )
            dense_prev.trainable = trainable
            input_of_dense = dense_prev(input_of_dense)
        # output layer
        index_of_last = self.number_of_conv2ds + 1 + self.number_of_dense_layers
        numer_of_prev_layer_nodes = self.weights['dense' + str(index_of_last)].shape[0]
        number_of_nodes = self.weights['dense' + str(index_of_last)].shape[1]
        kernel = tf.keras.backend.variable(
            self.weights['dense' + str(index_of_last)])
        bias = tf.keras.backend.variable(self.weights['bdense' + str(index_of_last)])
        output_layer_prev = tf.keras.layers.Dense(self.num_output_classes, activation=tf.nn.softmax,
                                                  kernel_initializer=tf.keras.initializers.Constant(kernel),
                                                  bias_initializer=tf.keras.initializers.Constant(bias))(input_of_dense)

        return tf.keras.Model(inputs=inputs, outputs=output_layer_prev, name='complete_model' + str(self.T))

    def build_expanded_dense_netowrk(self, actions=[], fix_previous=True, with_softmax=True):
        number_of_flatten_nodes = (config.pkt_size - 2 * (self.kernel_size - 1)) * (
                config.flow_size - 1 - 2 * (self.kernel_size - 1)) * self.conv2d_filters[-1]
        inputs = tf.keras.Input(shape=(number_of_flatten_nodes,))
        input_of_dense = inputs
        for i in range(self.number_of_conv2ds + 1, self.number_of_conv2ds + 1 + self.number_of_dense_layers):
            number_of_nodes = self.weights['dense' + str(i)].shape[1]
            if i == self.number_of_conv2ds + 1 or len(actions) == 0:
                kernels = tf.keras.backend.variable(self.weights['dense' + str(i)])
            else:
                kernels = tf.keras.backend.variable(self.add_dense_weight(self.weights['dense' + str(i)], actions[
                    i - (self.number_of_conv2ds + 1) - 1]))
            biases = tf.keras.backend.variable(self.weights['bdense' + str(i)])
            dense_prev = tf.keras.layers.Dense(self.weights['dense' + str(i)].shape[1],
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.keras.initializers.Constant(kernels),
                                               bias_initializer=tf.keras.initializers.Constant(biases)
                                               )
            dense_prev.trainable = not fix_previous
            if len(actions) != 0:
                dense_new = tf.keras.layers.Dense(actions[i - (self.number_of_conv2ds + 1)], activation=tf.nn.relu,
                                                  )
                input_of_dense = tf.keras.layers.Concatenate(axis=-1)(
                    [dense_prev(input_of_dense), dense_new(input_of_dense)])
            else:
                input_of_dense = dense_prev(input_of_dense)
        if with_softmax:
            final_activation = tf.nn.softmax
        else:
            final_activation = None

        index_of_last = self.number_of_conv2ds + 1 + self.number_of_dense_layers
        # if verbose:
        #     print("build " + str(index_of_last))
        number_of_nodes = self.weights['dense' + str(index_of_last)].shape[1]
        number_of_lower_nodes = self.weights['dense' + str(index_of_last)].shape[0]
        if len(actions) != 0 and actions[-1] != 0:
            kernel = tf.keras.backend.variable(
                self.add_dense_weight(self.weights['dense' + str(index_of_last)], actions[-1]))
            bias = tf.keras.backend.variable(self.weights['bdense' + str(index_of_last)])
            wstabilzer = WStabilizer(
                selected_lower_nodes=[i for i in range(number_of_lower_nodes, number_of_lower_nodes + actions[-1])],
                selected_upper_nodes=[0, 1],
                num_of_all_lower_nodes=number_of_lower_nodes + actions[-1],
                num_of_all_upper_nodes=self.weights['dense' + str(index_of_last)].shape[1]
            )
            wstabilzer.configure_weight_matrix(
                self.add_dense_weight(self.weights['dense' + str(index_of_last)], actions[-1]))

            output_layer_prev = tf.keras.layers.Dense(self.num_output_classes, activation=final_activation,
                                                      kernel_regularizer=wstabilzer)(
                input_of_dense)
        else:
            kernel = tf.keras.backend.variable(self.weights['dense' + str(index_of_last)])
            bias = tf.keras.backend.variable(self.weights['bdense' + str(index_of_last)])
            output_layer_prev = tf.keras.layers.Dense(self.num_output_classes, activation=final_activation,
                                                      kernel_initializer=
                                                      tf.keras.initializers.Constant(kernel),
                                                      bias_initializer=tf.keras.initializers.Constant(bias))
            output_layer_prev.trainable = not fix_previous
            output_layer_prev = output_layer_prev(input_of_dense)

        return tf.keras.Model(inputs=inputs, outputs=output_layer_prev, name='expanded_model' + str(self.T))

    def add_dense_weight(self, layer_weights, number_of_new_nodes):
        if number_of_new_nodes == 0:
            return layer_weights
        new_weights = np.array([[0 for i in range(layer_weights.shape[1])] for j in range(number_of_new_nodes)])
        return np.concatenate((layer_weights, new_weights))

    def evaluate(self, test_flows, test_labels, model=None, with_cnn=False):
        number_of_corrects = 0
        number_of_samples = 0
        flow_count = 0
        cnn_model = None
        if with_cnn:
            if model is None:
                model = self.build_complete_model()
            test_flows = tf.convert_to_tensor(np.expand_dims(test_flows, axis=-1), dtype=tf.float32)
        else:
            # if model is None:
            #     model = self.create_dense_network()
            cnn_model = self.cnn_network
            test_flows = tf.convert_to_tensor(np.expand_dims(test_flows, axis=-1), dtype=tf.float32)
        while flow_count < len(test_flows):
            end_index = min(flow_count + 100, len(test_flows) - 1)

            if not with_cnn:
                current_test_flows = cnn_model.predict(test_flows[flow_count:end_index])
            else:
                current_test_flows = test_flows[flow_count: end_index]

            labels = test_labels[flow_count:end_index]
            output = model.predict(current_test_flows)
            predicted_outputs = tf.argmax(output, axis=1)
            # number_of_corrects = 0
            number_of_samples += len(labels)
            for i in range(len(predicted_outputs)):
                if predicted_outputs[i] == labels[i]:
                    number_of_corrects += 1
            del current_test_flows
            del output
            gc.collect()
            flow_count += 100
        try:
            precision = number_of_corrects / number_of_samples
        except:
            precision = 0
        del test_flows
        gc.collect()
        return precision

    def reward(self, accuracy, actions):
        return accuracy - self.penalty * sum(actions)

    def compute_fisher_matrix(self, model, controller, train_data=None):
        if model is None:
            model = self.build_expanded_dense_netowrk(actions=[], fix_previous=False)
        max_f = 0.0
        F_matrix = []
        for layer in model.trainable_variables:
            F_matrix.append(np.zeros(shape=layer.shape))
        counter = 0
        if train_data is None:
            max_number_of_data = self.max_fisher_samples
            controller.reset()
        else:
            max_number_of_data = len(train_data)
        while counter < max_number_of_data:
            if counter % 100 == 0:
                print("fisher ", counter)
            if train_data is None:
                data = controller.generate('train')
                if data is False:
                    break
                flow = data['x']
            else:
                flow = np.expand_dims(train_data[counter], axis=0)

            x_input = self.cnn_network(np.expand_dims(flow, axis=-1))
            with tf.GradientTape() as tape:
                output = model(x_input)
                prob = tf.math.log(output[0][tf.argmax(output, axis=1)[0]])
            grads = tape.gradient(prob, model.trainable_variables)
            for layer in range(len(F_matrix)):
                squares = np.square(grads[layer])
                max_squares = np.amax(squares)
                if max_f < max_squares:
                    max_f = max_squares
                # if max_squares > self.max_fisher:
                # self.max_fisher = max_squares
                F_matrix[layer] += squares
            del x_input
            gc.collect()
            counter += 1
        s = counter
        for layer in range(len(F_matrix)):
            F_matrix[layer] /= s
        print(max_f)
        return F_matrix

    def memory_safe_sum_mul(self, student_model, original_model, layer_idx, step_size):
        # computing the fisher regularization tern for the flatten layer of a cnn would result in a memory error since the layer was too big.
        print('before num_rows')
        num_rows = self.fisher_matrix[layer_idx].shape[0]
        print(num_rows)
        computation_size = int(num_rows / step_size)
        output = 0
        for i in range(step_size):
            fisher_tensor = tf.convert_to_tensor(
                self.fisher_matrix[layer_idx][i * computation_size: (i + 1) * computation_size], dtype=np.float32)
            output += tf.math.reduce_sum(tf.math.multiply(fisher_tensor,
                                                          tf.math.square(student_model.trainable_variables[layer_idx][
                                                                         i * computation_size: (
                                                                                                       i + 1) * computation_size] -
                                                                         original_model.trainable_variables[layer_idx][
                                                                         i * computation_size: (
                                                                                                       i + 1) * computation_size])))
            del fisher_tensor
            gc.collect()

        return output

    #  @tf.function
    def compression_train_step(self, data, logits, labels, student_model, original_model, optimizer,
                               fisher_coefficient):
        print('before tape')
        with tf.GradientTape() as tape:
            print('before predict')
            predicted_outputs = student_model(data)
            # regularizer for preventing catastrophic forgetting with fisher matrix
            ewc_reg = 0
            fisher_coefficient = fisher_coefficient * self.max_fisher
            print('before layer')
            for layer in range(len(student_model.trainable_variables)):
                print('layer')
                # ewc_reg += fisher_coefficient * tf.math.reduce_sum(tf.math.multiply(self.fisher_matrix[layer],
                #                                                                     tf.math.square(
                #                                                                         student_model.trainable_variables[
                #                                                                             layer] -
                #                                                                         original_model.trainable_variables[
                #                                                                             layer])))
                if self.fisher_matrix[layer].shape[0] < 10000:
                    fisher_tensor = tf.convert_to_tensor(self.fisher_matrix[layer], dtype=np.float32)
                    ewc_reg += fisher_coefficient * tf.math.reduce_sum(tf.math.multiply(fisher_tensor,
                                                                                        tf.math.square(
                                                                                            student_model.trainable_variables[
                                                                                                layer] -
                                                                                            original_model.trainable_variables[
                                                                                                layer])))
                else:
                    ewc_reg += fisher_coefficient * self.memory_safe_sum_mul(student_model, original_model, layer, 8)

            # loss with regards to knowledge distillation
            kd_loss = tf.math.square(predicted_outputs - logits)
            print('kd')
            # loss with respect to the actual label
            predicted_outputs_softmax = tf.nn.softmax(predicted_outputs)
            entropy_loss = tf.keras.losses.BinaryCrossentropy()(labels, predicted_outputs_softmax)
            print('entropy')
            loss = entropy_loss + kd_loss + ewc_reg
        grads = tape.gradient(loss, student_model.trainable_variables)
        print('grads')
        del tape
        del logits
        del predicted_outputs
        del loss
        gc.collect()
        print('before optimize')
        optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
        print('optimize')
        del grads
        print('optimize - del')
        gc.collect()
        # return loss

    def compress(self, expanded_dense_model, train_flows, train_labels, epochs, batch_size, learning_rate,
                 fisher_coefficient):
        student_model = self.build_expanded_dense_netowrk(actions=[], fix_previous=False, with_softmax=False)
        original_model = self.build_expanded_dense_netowrk(actions=[], fix_previous=False, with_softmax=False)
        expanded_dense_model.layers[-1].activation = None  # changing softmax to none to obtain the logits
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # define our optimizer
        count = 0
        for epoch in range(epochs):
            print(epoch, ")")
            for idx in tqdm(range(0, len(train_flows), batch_size)):
                # print('for start')
                data = train_flows[idx: idx + batch_size]
                logits = expanded_dense_model(data)
                if count == 0:
                    print(logits)
                    count += 1
                # print('before label')
                labels = train_labels[idx: idx + batch_size].reshape(-1, 1)
                # print('before step')
                with tf.GradientTape() as tape:
                    # print('before predict')
                    predicted_outputs = student_model(data)
                    # regularizer for preventing catastrophic forgetting with fisher matrix
                    ewc_reg = 0
                    # fisher_coefficient = fisher_coefficient * self.max_fisher
                    # print('before layer')
                    for layer in range(len(student_model.trainable_variables)):
                        # print('layer')
                        # ewc_reg += fisher_coefficient * tf.math.reduce_sum(tf.math.multiply(self.fisher_matrix[layer],
                        #                                                                     tf.math.square(
                        #                                                                         student_model.trainable_variables[
                        #                                                                             layer] -
                        #                                                                         original_model.trainable_variables[
                        #                                                                             layer])))
                        # if self.fisher_matrix[layer].shape[0] < 10000:
                        fisher_tensor = tf.convert_to_tensor(self.fisher_matrix[layer], dtype=np.float32)
                        ewc_reg += fisher_coefficient * tf.math.reduce_sum(tf.math.multiply(fisher_tensor,
                                                                                            tf.math.square(
                                                                                                student_model.trainable_variables[
                                                                                                    layer] -
                                                                                                original_model.trainable_variables[
                                                                                                    layer])))
                        # else:
                        #     ewc_reg += fisher_coefficient * self.memory_safe_sum_mul(student_model, original_model,
                        #                                                              layer, 8)

                    # loss with regards to knowledge distillation
                    kd_loss = tf.math.square(predicted_outputs - logits)
                    # print('kd')
                    # loss with respect to the actual label
                    predicted_outputs_softmax = tf.nn.softmax(predicted_outputs)
                    entropy_loss = tf.keras.losses.BinaryCrossentropy()(labels, predicted_outputs_softmax)
                    # print('entropy')
                    loss = entropy_loss + kd_loss + ewc_reg
                grads = tape.gradient(loss, student_model.trainable_variables)
                # print('grads')
                del tape
                del logits
                del predicted_outputs
                del loss
                gc.collect()
                # print('before optimize')
                optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
                # print('optimize')
                del grads
                # print('optimize - del')
                gc.collect()

        student_model.layers[-1].activation = tf.nn.softmax
        del original_model
        gc.collect()
        return student_model

    def add_flows(self, train_flows, train_labels, validation_flows, validation_labels, new_test_flows, new_test_labels,
                  prev_test_flows, prev_test_labels,
                  benign_test_flows, benign_test_labels,
                  train_attacks, target_attacks, with_reinforeced=False, path='', save_compressed=True, save_fisher = True,
                  save_expanded=False, return_performances = False):
        train_flows_before_cnn = train_flows
        train_flows = tf.convert_to_tensor(np.expand_dims(train_flows, axis=-1), dtype=tf.float32)
        train_flows = self.cnn_network.predict(train_flows)
        best_actions = ''
        best_model = ''
        best_reward = 0
        best_validation_acc = 0
        rewards = []
        all_actions = []

        new_test_flows = np.concatenate((new_test_flows, benign_test_flows))
        new_test_labels = np.concatenate((new_test_labels, benign_test_labels))
        prev_test_flows = np.concatenate((prev_test_flows, benign_test_flows))
        prev_test_labels = np.concatenate((prev_test_labels, benign_test_labels))
        initial_accuracy = self.evaluate(new_test_flows, new_test_labels, model=None, with_cnn=True)
        
        expansion_time = 0
        compression_time = 0
        if not with_reinforeced:
            for iter in range(3):
                if best_validation_acc > 0.9:
                    break
                for batch_size in config.task_batch_sizes:
                    if best_validation_acc > 0.9:
                        break
                    for learning_rate in [1e-6, 1e-3, 1]:
                        actions = [self.nk for i in range(self.number_of_dense_layers)]
                        expanded_dense_network = self.build_expanded_dense_netowrk(actions=actions)
                        start_time = time()
                        expanded_dense_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                                       loss='sparse_categorical_crossentropy',
                                                       metrics=['accuracy'])
                        expanded_dense_network.fit(train_flows, train_labels, batch_size=batch_size,
                                                   epochs=config.task_epochs)
                        end_time = time()
                        seconds_elapsed = end_time - start_time
                        expansion_time += seconds_elapsed
                        print('cnn update time : ', seconds_elapsed)
                        accuracy_val = self.evaluate(validation_flows, validation_labels, expanded_dense_network)
                        all_actions.append(actions)
                        if best_validation_acc < accuracy_val:
                            best_model = expanded_dense_network
                            best_validation_acc = accuracy_val
                            best_actions = actions
                        else:
                            del expanded_dense_network
                            gc.collect()
                        if best_validation_acc > 0.9:
                            break

        else:
            for iter in range(3):
                if best_validation_acc > 0.9:
                    break
                for batch_size in config.task_batch_sizes:
                    if best_validation_acc > 0.9:
                        break
                    for learning_rate in [1e-6, 1e-3, 1]:
                        controller_network = Controller_Network(state_space=self.nk,
                                                                number_of_actions=self.number_of_dense_layers,
                                                                lstm_num_layers=self.num_lstam_layers)
                        if best_validation_acc > 0.9:
                            break
                        for reinforcement_epoch in range(self.controller_epochs):
                            print('reinforcement epoch : ', reinforcement_epoch)
                            actions = controller_network.get_actions()
                            expanded_dense_network = self.build_expanded_dense_netowrk(actions)
                            expanded_dense_network.compile(
                                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
                            expanded_dense_network.fit(train_flows, train_labels, batch_size=batch_size,
                                                       epochs=config.task_epochs)
                            accuracy_val = self.evaluate(validation_flows, validation_labels, expanded_dense_network)
                            reward = self.reward(accuracy_val, actions)
                            rewards.append(reward)
                            all_actions.append(actions)
                            if best_reward < reward:
                                best_reward = reward
                                best_model = expanded_dense_network
                                best_validation_acc = accuracy_val
                                best_actions = actions
                            else:
                                del expanded_dense_network
                                gc.collect()

                            controller_network.train(reward)
                            if best_validation_acc > 0.9:
                                break

        new_acc_exp = self.evaluate(new_test_flows, new_test_labels, best_model)
        prev_acc_exp = self.evaluate(prev_test_flows, prev_test_labels, model=best_model)

        if save_expanded:
            print("validation accuracy of best expanded model: ", best_validation_acc)
            print("accuracy of best expanded model on new:", new_acc_exp)
            print("accuracy of best expanded model on previous: ", prev_acc_exp)
            self.save_expanded_weights(best_model)
            return self.build_complete_model()

        best_compressed_model = None
        best_validation_compress = 0
        for iter in range(3):
            if best_validation_compress > 0.8:
                break
            for fisher_coefficient in config.fisher_coeffs:
                gc.collect()
                if best_validation_compress > 0.8:
                    break
                for batch_size in config.task_batch_sizes:
                    gc.collect()
                    if best_validation_compress > 0.8:
                        break
                    for learning_rate in [1e-3, 1e-6, 1]:
                        gc.collect()
                        start_time = time()
                        print('compression - iter,', iter, ' learning rate ', learning_rate, ' || batch_size ',
                              batch_size, ' || fisher coeff ', fisher_coefficient)
                        compressed_model = self.compress(expanded_dense_model=best_model, train_flows=train_flows,
                                                         train_labels=train_labels, batch_size=batch_size,
                                                         learning_rate=learning_rate, epochs=self.controller_epochs,
                                                         fisher_coefficient=fisher_coefficient)
                        end_time = time()
                        seconds_elapsed = end_time - start_time
                        compression_time += seconds_elapsed
                        validation = self.evaluate(validation_flows, validation_labels, compressed_model)
                        print("validation : ", validation)
                        if validation > best_validation_compress:
                            best_compressed_model = compressed_model
                            best_validation_compress = validation
                        else:
                            del compressed_model
                            gc.collect()
                        if best_validation_compress > 0.8:
                            break
        print(len(benign_test_flows))
        print(len(new_test_flows))
        print(len(prev_test_flows))
        print("validation accuracy of best expanded model: ", best_validation_acc)
        print("accuracy of best expanded model on new:", new_acc_exp)
        print("accuracy of best expanded model on previous: ", prev_acc_exp)
        # print("accuracy of best expanded model on benign: ", ben_acc1)
        print(actions)
        # print(rewards)
        print("##################")
        print("validation accuracy of best compressed model: ", best_validation_compress)
        acc_new_comp = self.evaluate(new_test_flows, new_test_labels, best_compressed_model)
        acc_prev_comp = self.evaluate(prev_test_flows, prev_test_labels, best_compressed_model)
        print("accuracy of best compressed model on new:", acc_new_comp)
        print("accuracy of best compressed model on previous: ", acc_prev_comp)

        number_of_samples = len(train_flows)
        output = 'Type : ' + self.mode + '\n'
        classes_trained = ''
        for classes in train_attacks:
            classes_trained += classes + ' '
        output += 'Flows trained on : ' + classes_trained + '\n'

        classes_tested = ''
        for classes in target_attacks:
            classes_tested += classes + ' '

        output += 'Flows targeted : ' + classes_tested + '\n'
        output += 'Number of Flows tested on : ' + str(number_of_samples) + '\n'
        output += "accuracy on original data after training : " + str(acc_prev_comp) + '\n'
        output += "accuracy on new attack before training : " + str(initial_accuracy) + '\n'
        nodes_added = ''
        for action in actions:
            nodes_added += ' ' + str(action)
        output += 'Number of nodes added to each layer : ' + nodes_added + '\n'
        output += "accuracy on new attack after training : " + str(acc_new_comp) + '\n'

        # output += '  Precision     Recall    F1-score\n'
        # output += '#####################################\n'
        # output += '  ' + str(precision) + '     ' + str(recall) + '     ' + str(f1_score)

        with open(path, 'w') as results:
            results.write(output)
        if return_performances:
            return expansion_time, compression_time
        if save_compressed:
            # best_compressed_model.summary()
            # print(len(best_compressed_model.layers))
            # tf.saved_model.save(best_compressed_model,'compress1')
            # print(len(best_compressed_model.layers[0].weights))

            # best_compressed_model.save('compress1')
            self.save_dense_weights(best_compressed_model)
            if save_fisher:
              new_fisher = self.compute_fisher_matrix(model=best_compressed_model, controller=None,
                                                      train_data=train_flows_before_cnn)
              with open('new_fisher.npy', 'wb') as f:
                  np.save(f, new_fisher)
              coeff = len(train_flows) / self.number_of_train_samples
              for layer in range(len(new_fisher)):
                  self.fisher_matrix[layer] += coeff * new_fisher[layer]
              self.number_of_train_samples += len(train_flows)

        # self.save_dense_weights(best_model)
        # test_acc = self.evaluate(test_flows,test_labels,best_model)
        # print("validation accuracy of best model: ",best_validation_acc)
        # print("test accuracy of best model: ", test_acc)
        # print(actions)
        # print(rewards)
        # accuracy_on_prev = self.evaluate(prev_flows,prev_labels,with_cnn = True)
        # print("accuracy on prev: ",  accuracy_on_prev)
        # self.report_model(test_flows,test_labels,best_actions,train_attacks,target_attacks,path = path,initial_accuracy=initial_accuracy,later_accuracy_on_previous = accuracy_on_prev)
        return acc_prev_comp, acc_new_comp

    def merge_model(self,new_model, new_n_samples = 0, coefficient = None):
        if coefficient is None:
            all_samples = self.number_of_train_samples + new_n_samples
            coef1 = self.number_of_train_samples / (all_samples * 1.0)
            coef2 =  new_n_samples/(all_samples * 1.0)
        else:
            coef1 = 1 - coefficient
            coef2 = coefficient
        new_model.summary()
        print(len(new_model.layers))
        for i in range(self.number_of_conv2ds + 1, self.number_of_conv2ds + 1 + self.number_of_dense_layers + 1):  # the + 1 is for flatten layer
            print(i)
            self.weights['dense' + str(i)] = coef1 * self.weights['dense' + str(i)] + coef2 * new_model.layers[i+1].trainable_variables[0].numpy()
            self.weights['bdense' + str(i)] = coef1 * self.weights['bdense' + str(i)] + coef2 * new_model.layers[i+1].trainable_variables[1].numpy()
    def report_model(self, test_flows, test_labels, actions, train_attacks, target_attacks, path, initial_accuracy,
                     later_accuracy_on_previous):
        model = self.build_complete_model()
        test_flows = tf.convert_to_tensor(np.expand_dims(test_flows, axis=-1), dtype=tf.float32)
        labels = test_labels
        output = model.predict(test_flows)
        predicted_outputs = tf.argmax(output, axis=1)
        number_of_samples = len(labels)
        output = ''
        if self.mode == 'binary':
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i in range(len(predicted_outputs)):
                if predicted_outputs[i] == labels[i]:
                    if labels[i] == 1:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if labels[i] == 1:
                        fn += 1
                    else:
                        fp += 1
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            try:
                precision = tp / (tp + fp)
            except:
                precision = 0
            try:
                recall = tp / (tp + fn)
            except:
                recall = 0
            try:
                f1_score = (2 * recall * precision) / (recall + precision)
            except:
                f1_score = 0
            accuracy = round(accuracy, 2)
            precision = round(precision, 2)
            recall = round(recall, 2)
            f1_score = round(f1_score, 2)
            output += 'Type : ' + self.mode + '\n'
            classes_trained = ''
            for classes in train_attacks:
                classes_trained += classes + ' '
            output += 'Flows trained on : ' + classes_trained + '\n'

            classes_tested = ''
            for classes in target_attacks:
                classes_tested += classes + ' '

            output += 'Flows targeted : ' + classes_tested + '\n'
            output += 'Number of Flows tested on : ' + str(number_of_samples) + '\n'
            output += "accuracy on original data after training : " + str(later_accuracy_on_previous) + '\n'
            output += "accuracy before training : " + str(initial_accuracy) + '\n'
            nodes_added = ''
            for action in actions:
                nodes_added += ' ' + str(action)
            output += 'Number of nodes added to each layer : ' + nodes_added + '\n'
            output += '  Precision     Recall    F1-score\n'
            output += '#####################################\n'
            output += '  ' + str(precision) + '     ' + str(recall) + '     ' + str(f1_score)
        with open(path, 'w') as results:
            results.write(output)










