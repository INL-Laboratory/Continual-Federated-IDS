from Utils.Task_Network import Task_Nerwork
from Utils.data2018 import DataController
from Utils import config
from Distributed.AsyncModules import *
from Distributed.Agent import Agent
from Distributed.Controller import AsyncController

import tensorflow as tf
from tqdm import tqdm
import random
import gc
attacks = ['attack_bot',
               'DOS_SlowLoris','DOS_GoldenEye', 'FTPBruteForce',
               'SSHPBruteForce' ]
normals = ['Benign_Friday_02','Benign_Friday_16',
           "Benign_Thursday_01","Benign_Thursday_15",
           "Benign_Thursday_22", "Benign_Wednesday_14" , "Benign_Tuesday_20"]

def data_sampler(prev_attacks, new_attack, normals, new_attack_size=128, min_prev_size=20,
                 max_benign=2000, validation_size=1000, test_size=1000):
    new_attack_controller = DataController(batch_size=new_attack_size, data_list=[new_attack], mode='binary',
                                           should_report=False, report_path='', flatten=False, max_attack_flow=2000)
    prev_attacks_controllers = []
    prev_batch_size = int(max(min_prev_size, new_attack_size / len(prev_attacks)))
    for prev_attack in prev_attacks:
        prev_attacks_controllers.append(
            DataController(batch_size=prev_batch_size, data_list=[prev_attack], mode='binary',
                           should_report=False, report_path='', flatten=False, max_attack_flow=2000))
    benign_controller = DataController(batch_size=int(min(new_attack_size * 2, max_benign)), data_list=normals,
                                       mode='binary',
                                       should_report=False, report_path='', flatten=False, max_attack_flow=2000)

    # generating new flows for continual training:
    new_flows = []
    new_labels = []
    data_benign = benign_controller.generate('train')
    new_flows += list(data_benign['x'])
    new_labels += list(data_benign['y'])
    data_new_attack = new_attack_controller.generate('train')
    new_flows += list(data_new_attack['x'])
    new_labels += list(data_new_attack['y'])
    for i in range(len(prev_attacks)):
        data_prev_attack = prev_attacks_controllers[i].generate('train')
        new_flows += list(data_prev_attack['x'])
        new_labels += list(data_prev_attack['y'])
    temp = list(zip(new_flows, new_labels))
    random.shuffle(temp)
    new_flows, new_labels = zip(*temp)
    new_flows = np.array(new_flows)
    new_labels = np.array(new_labels)

    # generating validation flows
    validation_list = [new_attack]
    for normal in normals:
        validation_list.append(normal)
    for prev_attack in prev_attacks:
        validation_list.append(prev_attack)
    validation_controller = DataController(batch_size=1, data_list=validation_list, mode='binary',
                                           should_report=False, report_path='', flatten=False, max_attack_flow=2000)
    validation_flows = []
    validation_labels = []
    for i in range(validation_size):
        data = validation_controller.generate('validation')
        if data is False:
            break
        validation_flows += list(data['x'])
        validation_labels += list(data['y'])
    validation_flows = np.array(validation_flows)
    validation_labels = np.array(validation_labels)

    # generating flows for testing the new attack
    new_test_flows = []
    new_test_labels = []
    new_attack_test_controller = DataController(batch_size=1, data_list=[new_attack], mode='binary',
                                                should_report=False, report_path='', flatten=False,
                                                max_attack_flow=2000)
    for i in range(test_size):
        data = new_attack_test_controller.generate('test')
        if data is False:
            break
        new_test_flows += list(data['x'])
        new_test_labels += list(data['y'])
    new_test_flows = np.array(new_test_flows)
    new_test_labels = np.array(new_test_labels)

    # generating flows for testing the previous attacks
    prev_test_flows = []
    prev_test_labels = []
    prev_attack_test_controller = DataController(batch_size=1, data_list=prev_attacks, mode='binary',
                                                 should_report=False, report_path='', flatten=False,
                                                 max_attack_flow=2000)
    for i in range(test_size):
        data = prev_attack_test_controller.generate('test')
        if data is False:
            break
        prev_test_flows += list(data['x'])
        prev_test_labels += list(data['y'])
    prev_test_flows = np.array(prev_test_flows)
    prev_test_labels = np.array(prev_test_labels)

    # generating flows for testing the beinigns
    benign_test_flows = []
    benign_test_labels = []
    benign_test_controller = DataController(batch_size=1, data_list=normals, mode='binary',
                                            should_report=False, report_path='', flatten=False,
                                            max_attack_flow=2000)
    for i in range(test_size):
        data = benign_test_controller.generate('test')
        if data is False:
            break
        benign_test_flows += list(data['x'])
        benign_test_labels += list(data['y'])
    benign_test_flows = np.array(benign_test_flows)
    benign_test_labels = np.array(benign_test_labels)

    return new_flows, new_labels, validation_flows, validation_labels, new_test_flows, new_test_labels, prev_test_flows, prev_test_labels, benign_test_flows, benign_test_labels


def create_base_cnn(data_list, path, max_attack_flows=3000):
    best_validation = 0
    best_model = None
    best_controller = None
    number_of_train_samples = 0
    number_of_test_samples = 0
    learning_rates = [1e-3, 1e-2]
    found = False
    middle = 256
    for iter in range(3):
        if found is True:
            break
        for learning_rate in learning_rates:
            if found is True:
                break
            for batch_size in config.base_batch_sizes:
                controller = DataController(batch_size=batch_size, data_list=data_list, mode='binary',
                                            should_report=True, report_path=path, flatten=False,
                                            max_attack_flow=max_attack_flows)

                base_cnn_model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation=tf.nn.relu,
                                           kernel_regularizer=tf.keras.regularizers.l1(config.l1_lambda),
                                           bias_regularizer=tf.keras.regularizers.l1(config.l1_lambda)
                                           ),
                    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation=tf.nn.relu,
                                           kernel_regularizer=tf.keras.regularizers.l1(config.l1_lambda),
                                           bias_regularizer=tf.keras.regularizers.l1(config.l1_lambda)
                                           ),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(256, activation=tf.nn.relu,
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

                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # define our optimizer

                if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists
                for epoch in range(config.epochs):
                    print("epoch : ", epoch + 1)
                    controller.reset()
                    for idx in tqdm(range(0, controller.train_n_batches, config.base_batch_size)):
                        data = controller.generate('train')
                        if data is False:
                            break
                        flows = data['x']
                        labels = data['y'].reshape(-1, 1)
                        train_flows = tf.convert_to_tensor(np.expand_dims(flows, axis=-1), dtype=tf.float32)
                        with tf.GradientTape() as tape:
                            predicted_outputs = base_cnn_model(train_flows)
                            loss_value = tf.keras.backend.sparse_categorical_crossentropy(labels, predicted_outputs)
                        grads = tape.gradient(loss_value, base_cnn_model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, base_cnn_model.trainable_variables))

                # evaluate model on all validation data
                controller.reset()
                number_of_corrects = 0
                number_of_samples = 0
                while 1:
                    data = controller.generate('validation')
                    if data is False:
                        break
                    test_flows = tf.convert_to_tensor(np.expand_dims(data['x'], axis=-1), dtype=tf.float32)
                    labels = data['y']
                    output = base_cnn_model.predict(test_flows)
                    predicted_outputs = tf.argmax(output, axis=1)

                    number_of_samples += len(labels)
                    for i in range(len(predicted_outputs)):
                        if predicted_outputs[i] == labels[i]:
                            number_of_corrects += 1

                print("number of test samples: ", number_of_samples)
                validation_acc = number_of_corrects / number_of_samples
                print("precision : ", validation_acc)
                if validation_acc > best_validation:
                    best_validation = validation_acc
                    best_model = base_cnn_model
                    number_of_train_samples = controller.test_n_batches
                    best_controller = controller
                number_of_test_samples = number_of_samples
                if validation_acc > 0.9:
                    found = True
                    break
    best_controller.reset()
    number_of_corrects = 0
    number_of_samples = 0
    while 1:
        data = best_controller.generate('test')
        if data is False:
            break
        test_flows = tf.convert_to_tensor(np.expand_dims(data['x'], axis=-1), dtype=tf.float32)
        labels = data['y']
        output = best_model(test_flows)
        predicted_outputs = tf.argmax(output, axis=1)

        number_of_samples += len(labels)
        for i in range(len(predicted_outputs)):
            if predicted_outputs[i] == labels[i]:
                number_of_corrects += 1
    test_acc = number_of_corrects / number_of_samples
    print("number of train samples: ", len(best_controller.trainIDs))
    print("number of test samples: ", number_of_test_samples)
    print("best validation precision : ", best_validation)
    print("best test precision : ", test_acc)
    return best_model, best_controller


def experiment(base_index=0, start_index=0, build_base=True, path_base='base-memory-cnn', path_fisher='fisher.npy'):
    global attacks
    global normals
    portions_size = int(len(attacks) / 2)
    first_attacks = []
    first_original_labels = {}
    second_attacks = []
    second_original_labels = {}
    index = 0
    while len(first_attacks) < portions_size:
        if index != base_index:
            first_attacks.append(attacks[index])
        index += 1
    while len(second_attacks) < portions_size:
        if index != base_index:
            second_attacks.append(attacks[index])
        index += 1
    for i in range(portions_size):
        first_original_labels[i] = attacks.index(first_attacks[i])
        second_original_labels[i] = attacks.index(second_attacks[i])

    flow_labels = [normal for normal in normals]
    flow_labels.append(attacks[base_index])
    fisher_controller = DataController(batch_size=1, data_list=flow_labels, mode='binary',
                                       should_report=False, report_path='', flatten=False,
                                       max_attack_flow=2000)

    if build_base:
        base_cnn, chosen_controller = create_base_cnn(flow_labels, path='base1-cnn.txt')
        base_cnn.save(path_base)
        number_of_train_samples = len(chosen_controller.trainIDs)
    else:
        base_cnn = tf.keras.models.load_model(path_base)
        number_of_train_samples = 1562
    new_flows = []
    new_flows += normals
    new_flows += first_attacks
    new_flows += second_attacks
    new_data_controller = DataController(batch_size=100, data_list=new_flows, mode='binary',
                                         should_report=False, report_path='', flatten=False, max_attack_flow=2000)
    # computing initial accuracy on new data
    number_of_corrects = 0
    number_of_samples = 0
    while 1:
        data = new_data_controller.generate('validation')
        if data is False:
            break
        test_flows = tf.convert_to_tensor(np.expand_dims(data['x'], axis=-1), dtype=tf.float32)
        labels = data['y']
        output = base_cnn.predict(test_flows)
        predicted_outputs = tf.argmax(output, axis=1)

        number_of_samples += len(labels)
        for i in range(len(predicted_outputs)):
            if predicted_outputs[i] == labels[i]:
                number_of_corrects += 1
    initial_accuracy = number_of_corrects / number_of_samples

    flow_labels = []
    flow_labels += normals
    flow_labels.append(attacks[base_index])
    fisher_controller = DataController(batch_size=1, data_list=flow_labels, mode='binary',
                                       should_report=False, report_path='', flatten=False,
                                       max_attack_flow=2000)
    fisher_network = Task_Nerwork(penalty=0.001, conv2d_filters=[8, 16],
                                  number_of_train_samples=number_of_train_samples)
    fisher_network.initialize_weights(input_model=base_cnn)
    fisher_network.update_fisher_matrix(model=None, controller=fisher_controller)
    fisher_matrix = fisher_network.fisher_matrix

    distillation_train_data, distillation_train_logits, distillation_train_labels = expand_models(base_cnn=base_cnn,
                                                                                                  number_of_train_samples=number_of_train_samples,
                                                                                                  attacks=first_attacks,
                                                                                                  original_attacks=
                                                                                                  [attacks[base_index]],
                                                                                                  original_labels=first_original_labels)
    half_trained_cnn = distill_models(base_index = base_index,base_cnn=base_cnn, distillation_train_data = distillation_train_data,
                   distillation_train_labels = distillation_train_labels,
                   distillation_train_logits = distillation_train_logits, number_of_train_samples=number_of_train_samples, attacks=first_attacks,
                   original_attacks= [attacks[base_index]], original_labels=first_original_labels,
                   fisher_matrix=fisher_matrix, initial_accuracy=initial_accuracy, save_number=0)
    for layer in range(2):
       half_trained_cnn.layers[layer].trainable = True
    temp = Task_Nerwork(penalty=0.001, conv2d_filters=[8, 16], number_of_train_samples=number_of_train_samples)
    temp.initialize_weights(input_model= half_trained_cnn)
    
    flow_labels = [normal for normal in normals]
    flow_labels += first_attacks
    new_fisher_controller =  DataController(batch_size=1, data_list=flow_labels, mode='binary',
                                       should_report=False, report_path='', flatten=False,
                                       max_attack_flow=2000)
    temp.update_fisher_matrix(model=None, controller=new_fisher_controller)
    number_of_agents_training_samples = 0
    for key in distillation_train_data:
      number_of_agents_training_samples += len(distillation_train_data[key])
    number_of_all_data = number_of_agents_training_samples + number_of_train_samples
    for layer in range(len(fisher_matrix)):
      fisher_matrix[layer] = (number_of_agents_training_samples/ number_of_all_data) * temp.fisher_matrix[layer] +  (number_of_train_samples/ number_of_all_data) * fisher_matrix[layer]
    
    original_attacks = [attacks[base_index]]
    original_attacks += first_attacks
    distillation_train_data, distillation_train_logits, distillation_train_labels = expand_models(base_cnn=half_trained_cnn,
                                                                                                  number_of_train_samples=number_of_all_data,
                                                                                                  attacks=second_attacks,
                                                                                                  original_attacks=
                                                                                                  original_attacks,
                                                                                                  original_labels=second_original_labels)
    distill_models(base_index = base_index,base_cnn= half_trained_cnn, distillation_train_data = distillation_train_data,
                   distillation_train_labels = distillation_train_labels,
                   distillation_train_logits = distillation_train_logits, number_of_train_samples= number_of_all_data, attacks=second_attacks,
                   original_attacks= original_attacks, original_labels= second_original_labels,
                   fisher_matrix=fisher_matrix, initial_accuracy=initial_accuracy, save_number=1, initial_attack=attacks[base_index])
    
    
    
    
    
    
    

    # print(inst.max_fisher)


def expand_models(base_cnn, number_of_train_samples, attacks, original_attacks, original_labels):
    global normals

    distillation_train_data = {}
    distillation_train_logits = {}
    distillation_train_labels = {}
    prev_attacks = original_attacks
    flow_labels = []
    flow_labels += normals
    flow_labels += original_attacks

    # with open('fisher.npy', 'wb') as f:
    # np.save(f, fisher_matrix)
    for i in range(len(attacks)):
        new_flows, new_labels, validation_flows, validation_labels, new_test_flows, new_test_labels, prev_test_flows, prev_test_labels, benign_test_flows, benign_test_labels = data_sampler(
            prev_attacks=prev_attacks, new_attack=attacks[i], normals=normals)
        inst = Task_Nerwork(penalty=0.001, conv2d_filters=[8, 16], number_of_train_samples=number_of_train_samples)
        inst.initialize_weights(input_model=base_cnn)

        expanded_model = inst.add_flows(new_flows, new_labels, validation_flows, validation_labels, new_test_flows,
                                        new_test_labels, prev_test_flows, prev_test_labels, benign_test_flows,
                                        benign_test_labels, train_attacks=flow_labels, target_attacks=[attacks[i]],

                                        path='', save_expanded=True)
        expanded_model.layers[-1].activation = None  # for obtaining logits
        distillation_train_flows = tf.convert_to_tensor(np.expand_dims(new_flows, axis=-1), dtype=tf.float32)
        train_logits = expanded_model.predict(distillation_train_flows)
        del expanded_model
        del inst
        gc.collect()
        distillation_train_data[original_labels[i]] = new_flows
        distillation_train_logits[original_labels[i]] = train_logits
        distillation_train_labels[original_labels[i]] = new_labels

    return distillation_train_data, distillation_train_logits, distillation_train_labels


def distill_models(base_index,base_cnn, distillation_train_data, distillation_train_logits, distillation_train_labels, number_of_train_samples, attacks, original_attacks, original_labels, fisher_matrix,
                   initial_accuracy, save_number, initial_attack=None):
    global normals
    all_attacks = ['attack_bot',
               'DOS_SlowLoris','DOS_GoldenEye', 'FTPBruteForce',
               'SSHPBruteForce' ]
    print("Starting Distilation")
    learning_rate = 1e-3
    batch_size = 32
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    epochs = 20
    new_flows = []
    new_flows += normals
    for i in range(len(attacks)):
        new_flows.append(attacks[i])
    
    old_flows = original_attacks
    old_flows += normals
    
    if initial_attack is None:
        old_data_controller = DataController(batch_size=100, data_list=old_flows, mode='binary',
                                             should_report=False, report_path='', flatten=False, max_attack_flow=2000)
        new_data_controller = DataController(batch_size=100, data_list=new_flows, mode='binary',
                                             should_report=False, report_path='', flatten=False, max_attack_flow=2000)
    else:  # old will be our initial attack and new will be the rest of the attacks
        all_new_attacks = normals.copy()
        all_new_attacks += [attack for attack in all_attacks if attack != initial_attack]

        initial_flows = normals.copy()
        initial_flows.append(initial_attack)
        old_data_controller = DataController(batch_size=100, data_list=initial_flows, mode='binary',
                                             should_report=False, report_path='', flatten=False, max_attack_flow=2000)
        new_data_controller = DataController(batch_size=100, data_list=all_new_attacks, mode='binary',
                                             should_report=False, report_path='', flatten=False, max_attack_flow=2000)
                                             
                                             
    base_cnn.layers[-1].activation = None
    for layer in range(2):
        base_cnn.layers[layer].trainable = False
    async_controller = AsyncController(optimizer=optimizer)
    async_controller.initialize_model(model=base_cnn)
    broadcaster = BroadCaster_Thread(controller=async_controller, agent_clients={}, data_controller=new_data_controller,
                                     original_data_controller=old_data_controller, main_attack= attacks[0],
                                     initial_accuracy=initial_accuracy,
                                     save_path='../../Results/results-cnn-distributed2018/exp' + str(base_index) + 'part' + str(
                                         save_number) + '.txt', epochs=20, number_of_agents=len(attacks))
    agents = []

    for i in range(len(attacks)):
        initial_model = tf.keras.models.clone_model(base_cnn)
        initial_model.set_weights(base_cnn.get_weights())
        original_model = tf.keras.models.clone_model(base_cnn)
        original_model.set_weights(base_cnn.get_weights())
        agent = Agent(id=i, batch_size=batch_size, fisher_matrix=fisher_matrix, shared_model=initial_model,
                      original_model=original_model)
        agent.get_data(new_x=np.array(distillation_train_data[original_labels[i]]),
                       new_labels=np.array(distillation_train_labels[original_labels[i]]),
                       new_logits=distillation_train_logits[original_labels[i]])
        client = AgentClient_Thread(agent=agent, controller_broadcaster=broadcaster)
        agents.append(agent)
        broadcaster.agent_clients[i] = client

    msg = {'model': base_cnn}
    print('threads initialized')
    broadcaster.broadcast_to_all(msg)
    while broadcaster.completed is False:
        # wait
        continue
    
    half_trained_cnn = broadcaster.controller.model
    for layer in range(2):
       half_trained_cnn.layers[layer].trainable = True
    return half_trained_cnn
    


for i in range(len(attacks)):
    experiment(i, build_base = True)


