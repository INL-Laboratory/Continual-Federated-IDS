from Clients import AgentClient,Broadcaster
from Agent import  Agent
from Controller import AsyncController
import threading
from Utils import config
from Utils.data import DataController
import tensorflow as tf
import numpy as np



class AgentClient_Thread(AgentClient):
    def __init__(self, controller_broadcaster,agent):
        self.controller_broadcaster = controller_broadcaster
        self.agent = agent
        self.thread = None
    def send_grads(self, msg):
        self.controller_broadcaster.receive(msg)

    def compute_gradient(self, msg):
        self.thread = threading.Thread(target=self.get_gradient_from_agent,args=(msg,))
        self.thread.start()


    def get_gradient_from_agent(self,msg): # this function will actually compute the gradient and send it
        updated_model = msg['model']
        self.agent.update_model(updated_model=  updated_model,change_weights=True)
        grads = self.agent.run()
        msg = {'grads': grads , 'id' : self.agent.id , 'epoch' : self.agent.epochs}
        self.send_grads(msg)


class BroadCaster_Thread(Broadcaster):

    def __init__(self,controller,agent_clients,epochs,data_controller,number_of_agents = 4):
        self.controller  = controller
        self.agent_clients = agent_clients
        self.number_of_agents = number_of_agents
        self.epochs = epochs # number of epochs we expect all agents to compute
        self.number_of_agents_completed = 0
        self.data_controller = data_controller
        self.lock = threading.Lock()

    def broadcast_to_all(self, msg):
        msg = {'model' : self.controller.model}
        for agent_client in self.agent_clients:
            agent_client.compute_gradient(msg)

    def broadcast_to_agent(self, id, msg):
        msg = {'model': self.controller.model}
        self.agent_clients[id].compute_gradient(msg)


    def receive(self, msg):
        grads = msg['grads']
        id = msg['id']
        epochs = msg['epoch']
        new_model = self.controller.update_model(grads,id)
        sending_message = {'model' : new_model}
        if epochs < self.epochs:
            self.broadcast_to_agent(id = id,msg=sending_message)
        else :
            self.lock.acquire()
            try:
                self.number_of_agents_completed += 1
                if self.number_of_agents_completed >= self.number_of_agents:
                    # all of the agents are finished
                    # evaluating the model
                    # evaluate model on all validation data
                    self.data_controller.reset()
                    number_of_corrects = 0
                    number_of_samples = 0
                    while 1:
                        data = self.data_controller.generate('validation')
                        if data is False:
                            break
                        test_flows = tf.convert_to_tensor(np.expand_dims(data['x'], axis=-1), dtype=tf.float32)
                        labels = data['y']
                        output = self.controller.model.predict(test_flows)
                        predicted_outputs = tf.argmax(output, axis=1)

                        number_of_samples += len(labels)
                        for i in range(len(predicted_outputs)):
                            if predicted_outputs[i] == labels[i]:
                                number_of_corrects += 1

                    print("number of test samples: ", number_of_samples)
                    print("precision : ", number_of_corrects / number_of_samples)
                    
            finally:
                self.lock.release()











learning_rate = 1e-3
controller = DataController(batch_size=128, data_list=config.divided2, mode='binary',
                            should_report= True, report_path = 'exp-sync1.txt', flatten = False, max_attack_flow = 1000)

base_cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters = 16, kernel_size=3, activation=tf.nn.relu,
                           kernel_regularizer= tf.keras.regularizers.l1(config.l1_lambda),
                           bias_regularizer=tf.keras.regularizers.l1(config.l1_lambda)
                           ),
    tf.keras.layers.Conv2D(filters = 32, kernel_size=3, activation=tf.nn.relu,
                           kernel_regularizer= tf.keras.regularizers.l1(config.l1_lambda),
                           bias_regularizer=tf.keras.regularizers.l1(config.l1_lambda)
                           ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu,
                          kernel_regularizer=tf.keras.regularizers.l1(config.l1_lambda),
                          bias_regularizer=tf.keras.regularizers.l1(config.l1_lambda)
                          ),
    tf.keras.layers.Dense(128, activation=tf.nn.relu,
                          kernel_regularizer=tf.keras.regularizers.l1(config.l1_lambda),
                          bias_regularizer=tf.keras.regularizers.l1(config.l1_lambda)
                          ),
    tf.keras.layers.Dense(64, activation=tf.nn.relu,
                          kernel_regularizer=tf.keras.regularizers.l1(config.l1_lambda),
                          bias_regularizer=tf.keras.regularizers.l1(config.l1_lambda)
                          ),


    tf.keras.layers.Dense(2, activation=tf.nn.softmax,
                          kernel_regularizer=tf.keras.regularizers.l1(config.l1_lambda),
                          bias_regularizer=tf.keras.regularizers.l1(config.l1_lambda)
                          )]

)

# code for building model
data = controller.generate('validation')
flows = data['x']
train_flows = tf.convert_to_tensor(np.expand_dims(flows, axis=-1), dtype=tf.float32)
base_cnn_model(train_flows)

optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate) # define our optimizer

agent_num = 4

# dividing our training data to 4 parts to give to the agents
criteria = controller.train_n_batches / agent_num
all_data = [[] for i in range(agent_num)]
all_labels = [[] for i in range(agent_num)]
for i in range(controller.train_n_batches):
    index = int(i/criteria)
    data = controller.generate('train')
    if data is False:
        break
    flows = data['x']
    labels = data['y'].reshape(-1, 1)
    train_flows = tf.convert_to_tensor(np.expand_dims(flows, axis=-1), dtype=tf.float32)
    all_data[index] += list(train_flows)
    all_labels[index] += list(labels)


for i in range(agent_num):
    print(len(all_data[i]))



async_controller = AsyncController(optimizer = optimizer)
async_controller.initialize_model(model=base_cnn_model)
broadcaster = BroadCaster_Thread(controller =async_controller,agent_clients=[],data_controller= controller,epochs=10)
agents = []
for i in range(agent_num):
    unique_model = tf.keras.models.clone_model(base_cnn_model)
    unique_model.set_weights(base_cnn_model.get_weights())
    agent = Agent(id = i,batch_size=32,shared_model=unique_model,loss_function=tf.keras.backend.sparse_categorical_crossentropy)
    agent.get_data(new_x=np.array(all_data[i]),new_y=np.array(all_labels[i]))
    client = AgentClient_Thread(agent=agent,controller_broadcaster= broadcaster)
    agents.append(agent)
    broadcaster.agent_clients.append(client)



msg = {'model' : base_cnn_model}
broadcaster.broadcast_to_all(msg)