from Distributed.Clients import AgentClient, Broadcaster
from Distributed.Agent import Agent
from Distributed.Controller import AsyncController
import threading
import tensorflow as tf
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class AgentClient_Thread(AgentClient):
    def __init__(self, controller_broadcaster, agent, lstm=False, thread_pool = None):
        self.controller_broadcaster = controller_broadcaster
        self.agent = agent
        self.thread = None
        self.lstm = lstm  # if in lstm mode, the messeages differ slightly
        self.thread_pool = thread_pool

    def send_grads(self, msg):
        self.controller_broadcaster.receive(msg)

    def compute_gradient(self, msg):
        if self.thread_pool is not None:
          #print('submitting to pool')
          self.thread_pool.submit(self.get_gradient_from_agent, msg)
        else:
          
          self.thread = threading.Thread(target=self.get_gradient_from_agent, args=(msg,))
          self.thread.start()
        

    def get_gradient_from_agent(self, msg):  # this function will actually compute the gradient and send it
        with tf.device('/cpu:0'):
          print('initialize getting gradient from agent')
          updated_model = msg['model']
          self.agent.update_model(updated_model=updated_model, change_weights=True, lstm=self.lstm)
          print('getting grads')
          grads = self.agent.run(lstm=self.lstm)
          print('grads received')
          msg = {'grads': grads, 'id': self.agent.id, 'epoch': self.agent.epochs, 'batch': self.agent.current_batch,
                 'total': self.agent.total_batch}
          self.send_grads(msg)


class BroadCaster_Thread(Broadcaster):

    def __init__(self, controller, agent_clients, epochs, data_controller, original_data_controller, main_attack,
                 initial_accuracy, save_path, number_of_agents=5, lstm=False):
        self.controller = controller
        self.agent_clients = agent_clients
        self.number_of_agents = number_of_agents
        self.epochs = epochs  # number of epochs we expect all agents to compute
        self.number_of_agents_completed = 0
        self.data_controller = data_controller
        self.original_data_controller = original_data_controller
        self.lock = threading.Lock()
        self.main_attack = main_attack
        self.initial_accuracy = initial_accuracy
        self.save_path = save_path
        self.completed = False
        self.lstm = lstm

    def broadcast_to_all(self, msg):
        if self.lstm:
            msg = {'model': self.controller.model.dense_model}
        else:
            msg = {'model': self.controller.model}
        #print('msg created')
        for agent_client in self.agent_clients:
            self.agent_clients[agent_client].compute_gradient(msg)

    def broadcast_to_agent(self, id, msg):
        if self.lstm:
            msg = {'model': self.controller.model.dense_model}
        else:
            msg = {'model': self.controller.model}
        self.agent_clients[id].compute_gradient(msg)

    def receive(self, msg):
        grads = msg['grads']
        id = msg['id']
        epochs = msg['epoch']
        batch = msg['batch']
        total = msg['total']
        new_model = self.controller.update_model(grads, id, batch, total)
        sending_message = {'model': new_model}
        if epochs < self.epochs:
            self.broadcast_to_agent(id=id, msg=sending_message)
        else:
            self.lock.acquire()
            try:
                self.number_of_agents_completed += 1
                if self.number_of_agents_completed >= self.number_of_agents:
                    # all of the agents are finished
                    # evaluating the model on new data
                    # evaluate model on all validation data
                    print('all threads are done')
                    if self.lstm:
                        self.controller.model.dense_model.layers[-1].activation = tf.nn.softmax
                    else: 
                        self.controller.model.layers[-1].activation = tf.nn.softmax
                    print('activation set')
                    self.data_controller.reset()
                    number_of_corrects = 0
                    number_of_samples = 0
                    counter = 0
                    while 1:
                        print('evaluating new ',counter)
                        counter += 1
                        data = self.data_controller.generate('validation')
                        if data is False:
                            break
                        if self.lstm:
                            test_flows = tf.convert_to_tensor(data['x'], dtype=tf.float32)
                            labels = data['y']
                            output = self.controller.model(test_flows)
                            self.controller.model.reset_state()
                        else:
                            test_flows = tf.convert_to_tensor(np.expand_dims(data['x'], axis=-1), dtype=tf.float32)
                            labels = data['y']
                            output = self.controller.model.predict(test_flows)
                        predicted_outputs = tf.argmax(output, axis=1)

                        number_of_samples += len(labels)
                        for i in range(len(predicted_outputs)):
                            if predicted_outputs[i] == labels[i]:
                                number_of_corrects += 1
                    accuracy_on_new_data = number_of_corrects / number_of_samples
                    # evaluating the model on previous data
                    # evaluate model on all validation data
                    self.data_controller.reset()
                    original_number_of_corrects = 0
                    original_number_of_samples = 0
                    counter = 0
                    while 1:
                        print('evaluating old ',counter)
                        counter += 1
                        data = self.original_data_controller.generate('validation')
                        if data is False:
                            break
                        if self.lstm:
                            test_flows = data['x']
                            labels = data['y']
                            output = self.controller.model(test_flows)
                        else:
                            test_flows = tf.convert_to_tensor(np.expand_dims(data['x'], axis=-1), dtype=tf.float32)
                            labels = data['y']
                            output = self.controller.model.predict(test_flows)
                        predicted_outputs = tf.argmax(output, axis=1)

                        original_number_of_samples += len(labels)
                        for i in range(len(predicted_outputs)):
                            if predicted_outputs[i] == labels[i]:
                                original_number_of_corrects += 1

                    accuracy_on_old_data = original_number_of_corrects / original_number_of_samples
                    output = ""
                    output += "Initial Attack: " + str(self.main_attack) + '\n'
                    output += "accuracy on original data after training : " + str(accuracy_on_old_data) + '\n'
                    output += "accuracy on new attack before training : " + str(self.initial_accuracy) + '\n'
                    output += "accuracy on new attack after training : " + str(accuracy_on_new_data) + '\n'
                    print(output)
                    with open(self.save_path, 'w') as results:
                        results.write(output)
                    self.completed = True

            except Exception as e:
                print(e)
            finally:
                self.lock.release()