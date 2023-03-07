from Clients import AgentClient,Broadcaster
from Agent import  SyncAgent
from Controller import SyncController
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
        self.thread = threading.Thread(target=self.get_gradient_from_agent)
        self.thread.start()


    def get_gradient_from_agent(self): # this function will actually compute the gradient and send it
        grads = self.agent.run()
        msg = {'grads': grads , 'id' : self.agent.id}
        self.send_grads(msg)


class BroadCaster_Thread(Broadcaster):

    def __init__(self,controller,agent_clients):
        self.controller  = controller
        self.agent_clients = agent_clients

    def broadcast_to_all(self, msg):
        for agent_client in self.agent_clients:
            agent_client.compute_gradient(None)


    def receive(self, msg):
        grads = msg['grads']
        id = msg['id']
        self.controller.append_to_grads(grads,id)







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

optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate) # define our optimizer

data = controller.generate('train')

sync_controller = SyncController(optimizer = optimizer)
sync_controller.initialize_model(model=base_cnn_model)
broadcaster = BroadCaster_Thread(controller =sync_controller,agent_clients=[])
agent_num = 4
agents = []
for i in range(agent_num):
    agent = SyncAgent(id = i,batch_size=8,shared_model=base_cnn_model,loss_function=tf.keras.backend.sparse_categorical_crossentropy)
    client = AgentClient_Thread(agent=agent,controller_broadcaster= broadcaster)
    agents.append(agent)
    broadcaster.agent_clients.append(client)


# if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
for epoch in range(config.epochs):
    print("epoch : ",epoch + 1)
    controller.reset()
    while True:
        data = controller.generate('train')
        if data is False:
            break
        flows = data['x']
        labels = data['y'].reshape(-1,1)
        train_flows = tf.convert_to_tensor(np.expand_dims(flows, axis=-1), dtype=tf.float32)
        start_interval = 0
        for i in range(agent_num):
            agents[i].get_data(train_flows[start_interval : start_interval + 32],labels[start_interval : start_interval + 32])
            start_interval += 32
        for time in range(4):
            broadcaster.broadcast_to_all(None)

            for i in range(agent_num):
                broadcaster.agent_clients[i].thread.join()

            sync_controller.update_model()


# evaluate model on all validation data
controller.reset()
number_of_corrects = 0
number_of_samples = 0
while 1:
    data = controller.generate('validation')
    if data is False:
        break
    test_flows = tf.convert_to_tensor(np.expand_dims(data['x'],axis=-1), dtype=tf.float32)
    labels = data['y']
    output = base_cnn_model.predict(test_flows)
    predicted_outputs = tf.argmax(output, axis=1)

    number_of_samples += len(labels)
    for i in range(len(predicted_outputs)):
        if predicted_outputs[i] == labels[i]:
            number_of_corrects += 1

print("number of test samples: ",number_of_samples )
print("precision : ", number_of_corrects/number_of_samples)



#code for loading
# reconstructed_model = tf.keras.models.load_model("base-cnn")

