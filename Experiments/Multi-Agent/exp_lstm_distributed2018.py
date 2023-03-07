from Utils.LSTM_Task_Network import LSTM_Task_Nerwork
from Utils.custom_LSTM import LSTM_Network
from Utils.data2018 import DataController
from Utils import config
from Distributed.AsyncModules import *
from Distributed.Agent import Agent
from Distributed.Controller import AsyncController
from time import time
import tensorflow as tf
from tqdm import tqdm
import random
import gc

#attacks = ['attack_bot', 'SSHPBruteForce', 'FTPBruteForce',
#           'DOS_SlowLoris', 'DOS_GoldenEye', 'Infilteration'
#           ]

attacks = ['attack_bot', 'DOS_SlowLoris','DOS_GoldenEye', 'FTPBruteForce', 'SSHPBruteForce' ,'Infilteration']
#attacks = [ 'DOS_SlowLoris','DOS_GoldenEye', 'FTPBruteForce', 'SSHPBruteForce' ]
#attacks = ['DOS_SlowLoris', 'DOS_GoldenEye']
normals = ['Benign_Friday_02', 'Benign_Friday_16',
           "Benign_Thursday_01", "Benign_Thursday_15",
           "Benign_Thursday_22", "Benign_Wednesday_14", "Benign_Tuesday_20"]
with_meta_data = False
LSTM_units = [1024]
dense_units = [512, 256, 128, 64, 2]


def create_base_lstm(data_list, path, max_attack_flows=4000, index=0):
    global LSTM_units
    global dense_units
    global with_meta_data
    if data_list[-1] == 'attack_DDOS':
        learning_rates = [1e-3, 1e-6, 1e-2, 1]
    else:
        learning_rates = [1e-3, 1e-2, 1e-6, 1]
    batch_sizes_1 = [ 32, 16, 64, 128]
    best_model, best_controller, best_validation, best_batch_size, best_learning_rate, number_of_train_samples, number_of_test_samples, found = train_lstm(
        data_list=data_list, path=path,
        max_attack_flows=max_attack_flows,
        learning_rates=learning_rates,
        batch_sizes=batch_sizes_1)

    best_controller.reset()
    number_of_corrects = 0
    number_of_samples = 0
    while 1:
        data = best_controller.generate('test')
        if data is False:
            break
        test_flows = tf.convert_to_tensor(data['x'], dtype=tf.float32)
        labels = data['y']
        output = best_model(test_flows)
        best_model.reset_state()
        predicted_outputs = tf.argmax(output, axis=1)

        number_of_samples += len(labels)
        for i in range(len(predicted_outputs)):
            if predicted_outputs[i] == labels[i]:
                number_of_corrects += 1
    test_acc = number_of_corrects / number_of_samples
    with open('lstm-base.txt', 'w') as results:
        output = "best test precision : " + str(test_acc)
        results.write(output)
    with open('lstm-paramaters/' + str(index) + '.txt', 'w') as results:
        output = "best learning rate: " + str(best_learning_rate) + '\n'
        output += "best batch size: " + str(best_batch_size)
        results.write(output)
    print("number of train samples: ", len(best_controller.trainIDs))
    print("number of test samples: ", number_of_test_samples)
    print("best validatation precision : ", best_validation)
    print("best test precision : ", test_acc)

    return best_model, best_controller


def train_lstm(data_list, path, max_attack_flows, learning_rates, batch_sizes):
    global with_meta_data
    best_validation = 0
    best_model = None
    best_controller = None
    best_batch_size = 0
    best_learning_rate = 0
    number_of_train_samples = 0
    number_of_test_samples = 0
    found = False
    for iter in range(3):
        if found is True:
            break
        for learning_rate in learning_rates:
            if found is True:
                break
            for batch_size in batch_sizes:
                controller = DataController(batch_size=batch_size, data_list=data_list, mode='binary',
                                            should_report=True, report_path=path, flatten=False,
                                            max_attack_flow=max_attack_flows, with_meta_data=with_meta_data)
                # dense_units=[128, 64, 32, 2]
                # base_lstm_model = LSTM_Network(input_size=config.pkt_size, LSTM_units=[256,128], dense_units=[ 64, 32, 2])
                # base_lstm_model = LSTM_Network(input_size=config.pkt_size, LSTM_units=[512,512], dense_units=[256,128,64,32, 2]
                # base_lstm_model = LSTM_Network(input_size=config.pkt_size, LSTM_units=[128], dense_units=[ 64, 32, 2])
                # base_lstm_model = LSTM_Network(input_size=config.pkt_size, LSTM_units=[1024,512], dense_units=[256,128,64,32, 2])
                # base_lstm_model = LSTM_Network(input_size=config.pkt_size, LSTM_units=[512,256], dense_units=[256,128,64, 2])
                # base_lstm_model = LSTM_Network(input_size=config.pkt_size, LSTM_units=[1024],
                # dense_units=[512, 256, 128, 64, 2])
                base_lstm_model = LSTM_Network(input_size=config.pkt_size, LSTM_units=LSTM_units,
                                               dense_units=dense_units)

                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # define our optimizer

                if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists
                start_time = time()
                for epoch in range(config.epochs):
                    print("epoch : ", epoch + 1)
                    controller.reset()
                    for idx in tqdm(range(0, controller.train_n_batches, batch_size)):
                        data = controller.generate('train')
                        if data is False:
                            break
                        flows = data['x']
                        labels = data['y'].reshape(-1, 1)
                        train_flows = tf.convert_to_tensor(flows, dtype=tf.float32)
                        with tf.GradientTape() as tape:
                            predicted_outputs = base_lstm_model(train_flows)
                            loss_value = tf.keras.backend.sparse_categorical_crossentropy(labels, predicted_outputs)
                        grads = tape.gradient(loss_value, base_lstm_model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, base_lstm_model.trainable_variables))
                        base_lstm_model.reset_state()
                end_time = time()
                seconds_elapsed = end_time - start_time
                print('cnn initial training time: ', seconds_elapsed)
                # evaluate model on all validation data
                controller.reset()
                number_of_corrects = 0
                number_of_samples = 0
                while 1:
                    data = controller.generate('validation')
                    if data is False:
                        break
                    test_flows = tf.convert_to_tensor(data['x'], dtype=tf.float32)
                    labels = data['y']
                    output = base_lstm_model(test_flows)
                    base_lstm_model.reset_state()
                    predicted_outputs = tf.argmax(output, axis=1)

                    number_of_samples += len(labels)
                    for i in range(len(predicted_outputs)):
                        if predicted_outputs[i] == labels[i]:
                            number_of_corrects += 1
                validation_acc = number_of_corrects / number_of_samples
                print("iter:", iter, " batch size: ", batch_size, " learning rate : ", learning_rate)
                print("precision : ", validation_acc)
                if validation_acc > best_validation:
                    best_validation = validation_acc
                    best_model = base_lstm_model
                    number_of_train_samples = controller.train_n_batches
                    best_controller = controller
                    best_batch_size = batch_size
                    best_learning_rate = learning_rate
                if validation_acc > 0.8:
                    found = True
                    break
                number_of_test_samples = number_of_samples

    return best_model, best_controller, best_validation, best_batch_size, best_learning_rate, number_of_train_samples, number_of_test_samples, found


def data_sampler(prev_attacks, new_attack, normals, new_attack_size=128, min_prev_size=20,
                 max_benign=2000, validation_size=1000, test_size=1000):
    global with_meta_data
    new_attack_controller = DataController(batch_size=new_attack_size, data_list=[new_attack], mode='binary',
                                           should_report=False, report_path='', flatten=False, max_attack_flow=2000,
                                           with_meta_data=with_meta_data)
    prev_attacks_controllers = []
    prev_batch_size = int(max(min_prev_size, new_attack_size / len(prev_attacks)))
    for prev_attack in prev_attacks:
        prev_attacks_controllers.append(
            DataController(batch_size=prev_batch_size, data_list=[prev_attack], mode='binary',
                           should_report=False, report_path='', flatten=False, max_attack_flow=2000,
                           with_meta_data=with_meta_data))
    benign_controller = DataController(batch_size=int(min(new_attack_size * 2, max_benign)), data_list=normals,
                                       mode='binary',
                                       should_report=False, report_path='', flatten=False, max_attack_flow=2000,
                                       with_meta_data=with_meta_data)

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
                                           should_report=False, report_path='', flatten=False, max_attack_flow=2000,
                                           with_meta_data=with_meta_data)
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
                                                max_attack_flow=2000, with_meta_data=with_meta_data)
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
                                                 max_attack_flow=2000, with_meta_data=with_meta_data)
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
                                            max_attack_flow=2000, with_meta_data=with_meta_data)
    for i in range(test_size):
        data = benign_test_controller.generate('test')
        if data is False:
            break
        benign_test_flows += list(data['x'])
        benign_test_labels += list(data['y'])
    benign_test_flows = np.array(benign_test_flows)
    benign_test_labels = np.array(benign_test_labels)

    return new_flows, new_labels, validation_flows, validation_labels, new_test_flows, new_test_labels, prev_test_flows, prev_test_labels, benign_test_flows, benign_test_labels


def experiment(base_index=0, start_index=0, build_base=True, path_base='base-lstm-1024', path_fisher='fisher.npy', compute_fisher = False):
    global with_meta_data
    global LSTM_units
    global dense_units
    global attacks
    global normals
    first_attacks = [attacks[i] for i in range(len(attacks)) if i != base_index]
    first_original_labels ={}
    for i in range(len(first_attacks)):
        first_original_labels[i] = attacks.index(first_attacks[i])



    flow_labels = [normal for normal in normals]
    flow_labels.append(attacks[base_index])
    fisher_controller = DataController(batch_size=1, data_list=flow_labels, mode='binary',
                                       should_report=False, report_path='', flatten=False,
                                       max_attack_flow=2000, with_meta_data=with_meta_data)
    if build_base:
        base_lstm, chosen_controller = create_base_lstm(flow_labels, path='base1-lstm.txt')
        if with_meta_data:
            base_lstm.dense_model.save_weights(path_base + '/dense/', save_format='tf')
            tf.keras.models.save_model(base_lstm.lstm, path_base +'/lstm/',
                                       save_format='tf')
        elif base_index == 0 or base_index == 1 or base_index == 4:
            base_lstm.dense_model.save_weights('lstm-models2018/lstm' + str(base_index) + '/dense/', save_format='tf')
            tf.keras.models.save_model(base_lstm.lstm, 'lstm-models2018/lstm' + str(base_index) + '/lstm/',
                                       save_format='tf')

        else:
            base_lstm.dense_model.save_weights('lstm-models2018/lstm/dense/', save_format='tf')
            tf.keras.models.save_model(base_lstm.lstm, 'lstm-models2018/lstm/lstm/', save_format='tf')

        # base_lstm.save('base-memory-lstm')
        number_of_train_samples = len(chosen_controller.trainIDs)
    else:
        # base_lstm = tf.keras.models.load_model(path_base)
        base_lstm = LSTM_Network(input_size=config.pkt_size, LSTM_units=LSTM_units,
                                 dense_units=dense_units)
        if with_meta_data:
            flows = np.zeros([1, config.flow_size, config.pkt_size])
        else:
            flows = np.zeros([1, config.flow_size - 1, config.pkt_size])
        output = base_lstm(flows)
        base_lstm.lstm = tf.keras.models.load_model(path_base + '/lstm/')
        base_lstm.dense_model.load_weights(path_base + '/dense/')
        number_of_train_samples = 1562

    new_flows = []
    new_flows += normals
    new_flows += first_attacks
    new_data_controller = DataController(batch_size=100, data_list=new_flows, mode='binary',
                                         should_report=False, report_path='', flatten=False, max_attack_flow=2000,
                                         with_meta_data=with_meta_data)
    # computing initial accuracy on new data
    number_of_corrects = 0
    number_of_samples = 0

    while 1:
        data = new_data_controller.generate('validation')
        if data is False:
            break
        test_flows = tf.convert_to_tensor(data['x'], dtype=tf.float32)
        # test_flows = data['x']
        labels = data['y']
        output = base_lstm(test_flows)
        base_lstm.reset_state()
        predicted_outputs = tf.argmax(output, axis=1)

        number_of_samples += len(labels)
        for i in range(len(predicted_outputs)):
            if predicted_outputs[i] == labels[i]:
                number_of_corrects += 1
    initial_accuracy = number_of_corrects / number_of_samples
    new_data_controller.reset()
    flow_labels = []
    flow_labels += normals
    flow_labels.append(attacks[base_index])

    if build_base or compute_fisher:
        fisher_network = LSTM_Task_Nerwork(penalty=0.001, lstm_layers=LSTM_units, dense_layers=dense_units, nk=10)
        fisher_network.initialize_weights(input_model=base_lstm)
        fisher_network.update_fisher_matrix(model=None, controller=fisher_controller)
        fisher_matrix = fisher_network.fisher_matrix
        with open(path_fisher, 'wb') as f:
            np.save(f, fisher_matrix)
    else:
        fisher_matrix = np.load(path_fisher, allow_pickle=True)

    distillation_train_data, distillation_train_logits, distillation_train_labels = expand_models(base_index=base_index,
                                                                                                  base_lstm=base_lstm,
                                                                                                  number_of_train_samples=number_of_train_samples,
                                                                                                  attacks=first_attacks,
                                                                                                  original_attacks=
                                                                                                  [attacks[base_index]],
                                                                                                  original_labels=first_original_labels)

    lstm_distillation_train_data = {}
    for key in distillation_train_data.keys():
        lstm_distillation_train_data[key] = base_lstm.call_lstm(distillation_train_data[key].astype('float32'))

    half_trained_lstm = distill_models(base_index=base_index, base_lstm=base_lstm,
                                       distillation_train_data=lstm_distillation_train_data,
                                       distillation_train_labels=distillation_train_labels,
                                       distillation_train_logits=distillation_train_logits,
                                       number_of_train_samples=number_of_train_samples, attacks=first_attacks,
                                       original_attacks=[attacks[base_index]], original_labels=first_original_labels,
                                       fisher_matrix=fisher_matrix, initial_accuracy=initial_accuracy, save_number=0)
    # for layer in range(2):
    # half_trained_lstm.layers[layer].trainable = True



def expand_models(base_lstm, number_of_train_samples, attacks, original_attacks, original_labels, base_index=0,
                  more_data_indices=None):
    global normals
    global with_meta_data
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
        # if i in more_data_indices:
        # new_attack_size = 256
        # else:
        # new_attack_size = 128
        new_attack_size = 512
        new_flows, new_labels, validation_flows, validation_labels, new_test_flows, new_test_labels, prev_test_flows, prev_test_labels, benign_test_flows, benign_test_labels = data_sampler(
            prev_attacks=prev_attacks, new_attack=attacks[i], normals=normals, new_attack_size=new_attack_size)
        inst = LSTM_Task_Nerwork(penalty=0.001, lstm_layers=LSTM_units, dense_layers=dense_units, nk=10)
        inst.initialize_weights(input_model=base_lstm)

        expanded_model, _, _ = inst.add_flows(new_flows, new_labels, validation_flows,
                                                                             validation_labels, new_test_flows,
                                                                             new_test_labels, prev_test_flows,
                                                                             prev_test_labels, benign_test_flows,
                                                                             benign_test_labels,
                                                                             train_attacks=flow_labels,
                                                                             target_attacks=[attacks[i]],
                                                                             path=''
                                                                             , save_expanded=True,
                                                                             save_expanded_to_file=True)

        new_flows, new_labels, validation_flows, validation_labels, new_test_flows, new_test_labels, prev_test_flows, prev_test_labels, benign_test_flows, benign_test_labels = data_sampler(
            prev_attacks=prev_attacks, new_attack=attacks[i], normals=normals)
        # expanded_model.summary()
        expanded_model.dense_model.layers[-1].activation = None  # for obtaining logits
        distillation_train_flows = tf.convert_to_tensor(new_flows, dtype=tf.float32)
        train_logits = expanded_model(distillation_train_flows)
        del expanded_model
        del inst
        gc.collect()
        distillation_train_data[original_labels[i]] = new_flows
        distillation_train_logits[original_labels[i]] = train_logits
        distillation_train_labels[original_labels[i]] = new_labels

    return distillation_train_data, distillation_train_logits, distillation_train_labels


def distill_models(base_index, base_lstm, distillation_train_data, distillation_train_logits, distillation_train_labels,
                   number_of_train_samples, attacks, original_attacks, original_labels, fisher_matrix,
                   initial_accuracy, save_number, full_lstm=False, initial_attack=None):
    global normals
    global with_meta_data
    all_attacks = ['attack_bot', 'SSHPBruteForce', 'FTPBruteForce',
                   'DOS_SlowLoris', 'DOS_GoldenEye'
                   ]
    print("Starting Distilation")
    learning_rate = 1e-3
    batch_size = 16
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
                                             should_report=False, report_path='', flatten=False, max_attack_flow=2000,
                                             with_meta_data=with_meta_data)
        new_data_controller = DataController(batch_size=100, data_list=new_flows, mode='binary',
                                             should_report=False, report_path='', flatten=False, max_attack_flow=2000,
                                             with_meta_data=with_meta_data)
    else:  # old will be our initial attack and new will be the rest of the attacks
        all_new_attacks = normals.copy()
        all_new_attacks += [attack for attack in all_attacks if attack != initial_attack]

        initial_flows = normals.copy()
        initial_flows.append(initial_attack)
        old_data_controller = DataController(batch_size=100, data_list=initial_flows, mode='binary',
                                             should_report=False, report_path='', flatten=False, max_attack_flow=2000,
                                             with_meta_data=with_meta_data)
        new_data_controller = DataController(batch_size=100, data_list=all_new_attacks, mode='binary',
                                             should_report=False, report_path='', flatten=False, max_attack_flow=2000,
                                             with_meta_data=with_meta_data)

    base_lstm.dense_model.layers[-1].activation = None
    base_lstm.lstm.trainable = False
    flows = np.zeros([1, config.flow_size - 1, config.pkt_size])
    output = base_lstm(flows)
    # for layer in range(2):
    # base_lstm.layers[layer].trainable = False
    threadPool = ThreadPoolExecutor(max_workers=len(attacks))
    async_controller = AsyncController(optimizer=optimizer, type='LSTM')
    async_controller.initialize_model(model=base_lstm)
    broadcaster = BroadCaster_Thread(controller=async_controller, agent_clients={}, data_controller=new_data_controller,
                                     original_data_controller=old_data_controller, main_attack = original_attacks[0],
                                     initial_accuracy=initial_accuracy,
                                     save_path='../../Results/results-lstm-distributed2018/exp' + str(base_index) + 'part' + str(
                                         save_number) + '.txt', epochs=20, number_of_agents=len(attacks), lstm=True)
    agents = []

    for i in range(len(attacks)):
        initial_model = tf.keras.models.clone_model(base_lstm.dense_model)
        initial_model.set_weights(base_lstm.dense_model.get_weights())
        original_model = tf.keras.models.clone_model(base_lstm.dense_model)
        original_model.set_weights(base_lstm.dense_model.get_weights())
        # initial_model = base_lstm.copy()
        # original_model = base_lstm.copy()
        # flows = np.zeros([1,config.flow_size - 1,config.pkt_size])
        # output = initial_model(flows)
        # output = original_model(flows)
        # initial_model.lstm.trainable = False
        # original_model.lstm.trainable = False
        agent = Agent(id=i, batch_size=batch_size, fisher_matrix=fisher_matrix, shared_model=initial_model,
                      original_model=original_model)
        agent.get_data(new_x=np.array(distillation_train_data[original_labels[i]]),
                       new_labels=np.array(distillation_train_labels[original_labels[i]]),
                       new_logits=distillation_train_logits[original_labels[i]])
        client = AgentClient_Thread(agent=agent, controller_broadcaster=broadcaster, lstm=True, thread_pool=threadPool)
        agents.append(agent)
        broadcaster.agent_clients[i] = client

    msg = {'model': base_lstm}
    print('threads initialized')
    broadcaster.broadcast_to_all(msg)
    while broadcaster.completed is False:
        # wait
        continue

    half_trained_lstm = broadcaster.controller.model
    half_trained_lstm.lstm.trainable = True
    return half_trained_lstm

for i in range(len(attacks)):
    experiment(i, build_base=True, path_base='lstm-models2018/lstm',
               path_fisher='fishers-lstm2018/' + str(i) + '.npy', compute_fisher = False)
