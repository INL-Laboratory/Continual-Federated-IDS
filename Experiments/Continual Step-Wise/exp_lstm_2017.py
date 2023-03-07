from Utils.LSTM_Task_Network import LSTM_Task_Nerwork
from Utils.custom_LSTM import LSTM_Network
from Utils.data import DataController
from Utils import config
import numpy as np

import tensorflow as tf
from tqdm import tqdm
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print('GPU configured Successfully!')
    except RuntimeError as e:
        print('GPU could not be configured correctly! Error:')
        print(e)

# for the lstm 2017 step wise evaluations, we used more random permutations (saved in permutations folder) to demonstrate
# the effectiveness of our proposed approach

initial_batch_sizes = []
initial_learning_rates = []

continual_batch_sizes = []
continual_learning_rates =[]
def create_base_lstm(data_list, path, max_attack_flows=3000):
    global initial_batch_sizes
    global initial_learning_rates
    best_validation = 0
    best_model = None
    best_controller = None
    best_batch_size = 0
    best_learning_rate = 0
    number_of_train_samples = 0
    number_of_test_samples = 0
    learning_rates = [1e-3, 1e-2, 1e-6, 1]
    found = False
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
                # dense_units=[128, 64, 32, 2]
                # base_lstm_model = LSTM_Network(input_size=config.pkt_size, LSTM_units=[256,128], dense_units=[ 64, 32, 2])
                # base_lstm_model = LSTM_Network(input_size=config.pkt_size, LSTM_units=[512,512], dense_units=[256,128,64,32, 2]
                # base_lstm_model = LSTM_Network(input_size=config.pkt_size, LSTM_units=[128], dense_units=[ 64, 32, 2])
                # base_lstm_model = LSTM_Network(input_size=config.pkt_size, LSTM_units=[1024,512], dense_units=[256,128,64,32, 2])
                # base_lstm_model = LSTM_Network(input_size=config.pkt_size, LSTM_units=[512,256], dense_units=[256,128,64, 2])
                base_lstm_model = LSTM_Network(input_size=config.pkt_size, LSTM_units=[1024],
                                               dense_units=[512, 256, 128, 64, 2])

                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # define our optimizer

                if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists
                for epoch in range(config.epochs):
                    print("epoch : ", epoch + 1)
                    controller.reset()
                    for idx in tqdm(range(0, controller.train_n_batches, 1)):
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
                    best_batch_size = batch_size
                    best_learning_rate = learning_rate
                    number_of_train_samples = controller.train_n_batches
                    best_controller = controller
                if validation_acc > 0.9:
                    found = True
                    break
                number_of_test_samples = number_of_samples

    best_controller.reset()
    initial_batch_sizes.append(best_batch_size)
    initial_learning_rates.append(best_learning_rate)
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
    print("number of train samples: ", len(best_controller.trainIDs))
    print("number of test samples: ", number_of_test_samples)
    print("best validatation precision : ", best_validation)
    print("best test precision : ", test_acc)
    return best_model, best_controller


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

attacks = ['attack_bot', 'attack_DDOS', \
               'attack_portscan', 'DOS_SlowHttpTest', \
               'DOS_SlowLoris', 'DOS_Hulk', 'DOS_GoldenEye', 'FTPPatator', \
               'SSHPatator', 'Web_BruteForce', 'Web_XSS']

experiment_counter = 0
number_of_training_samples = [1552,3840, 3840, 3840, 3840, 3840, 3840, 3840, 3232, 1728, 864]
def experiment(base_index=0, rest_of_the_attacks = [], build_base=True,
               path_base='base-memory-lstm', path_fisher='fisher.npy',
               path_save_model = '', path_save_fisher = ''):
    # normals = ['vectorize_friday/benign','Benign_Wednesday','Benign_Tuesday','Benign_Thursday']
    global experiment_counter
    global continual_batch_sizes
    global continual_learning_rates
    normals = ['vectorize_friday/benign', 'Benign_Wednesday']
    flow_labels = [normal for normal in normals]
    flow_labels.append(attacks[base_index])
    if build_base:
        fisher_controller = DataController(batch_size=1, data_list=flow_labels, mode='binary',
                                           should_report=False, report_path='', flatten=False,
                                           max_attack_flow=2000)
        base_lstm, chosen_controller = create_base_lstm(flow_labels, path='base1-cnn.txt')
        #base_lstm.save('base-memory-lstm')
        base_lstm.save(path_save_model)

        inst = LSTM_Task_Nerwork(penalty=0.001, lstm_layers=[1024], dense_layers=[512, 256, 128, 64, 2], nk=10,
                                 number_of_train_samples=len(chosen_controller.trainIDs))
        inst.initialize_weights(input_model=base_lstm)
        inst.update_fisher_matrix(model=None, controller=fisher_controller)
        #with open('fisher.npy', 'wb') as f:
        with open(path_save_fisher, 'wb') as f:
            np.save(f, inst.fisher_matrix)
    else:
        inst = LSTM_Task_Nerwork(penalty=0.001, lstm_layers=[1024], dense_layers=[512, 256, 128, 64, 2], nk=10)
        inst.initialize_weights(path=path_base)
        inst.fisher_matrix = np.load(path_fisher, allow_pickle=True)
        inst.number_of_train_samples = number_of_training_samples[base_index]

    # print(inst.max_fisher)

    prev_attacks = [attacks[base_index]]
    for i in range(len(rest_of_the_attacks)):
        new_flows, new_labels, validation_flows, validation_labels, new_test_flows, new_test_labels, prev_test_flows, prev_test_labels, benign_test_flows, benign_test_labels = data_sampler(
            prev_attacks=prev_attacks, new_attack=rest_of_the_attacks[i], normals=normals)

        acc_prev, acc_new, best_batch_size, best_learning_rate = inst.add_flows(new_flows, new_labels, validation_flows, validation_labels, new_test_flows,
                                           new_test_labels, prev_test_flows, prev_test_labels, benign_test_flows,
                                           benign_test_labels, train_attacks=flow_labels, target_attacks=[rest_of_the_attacks[i]],

                                           path='../../Results/results-lstm-stepwise2017/results/exp' + str(experiment_counter) + str(i) + '.txt', save_compressed =True)
        continual_batch_sizes.append(best_batch_size)
        continual_learning_rates.append(best_learning_rate)
        prev_attacks.append(rest_of_the_attacks[i])
    experiment_counter += 1


model_exists = {}

for attack in attacks:
    model_exists[attack] = False
number_of_experiments = 40
for i in range(number_of_experiments):
    
    base_index = i % len(attacks)
    rest_of_the_attacks = [attack for attack in attacks if attack != attacks[base_index]]
    random.shuffle(rest_of_the_attacks)
    #if i < 30:
        #continue
    with open('permutations/' + str(i) + '.txt', 'w') as fp:
        for item in rest_of_the_attacks:
            # write each item on a new line
            fp.write("%s\n" % item)
    if model_exists[attacks[base_index]] is False:
        experiment(base_index= base_index, rest_of_the_attacks = rest_of_the_attacks,
                   build_base= True, path_save_model= 'lstm_models/' + attacks[base_index],
                   path_save_fisher= 'lstm_fishers/' + attacks[base_index] + '.npy')
        #model_exists[attacks[base_index]] = True
    else:
        experiment(base_index=base_index, rest_of_the_attacks=rest_of_the_attacks,
                   build_base=False, path_base= 'lstm_models/' + attacks[base_index],
                  path_fisher= 'lstm_fishers/' + attacks[base_index] + '.npy')


