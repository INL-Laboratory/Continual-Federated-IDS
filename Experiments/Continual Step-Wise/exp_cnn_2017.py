from Utils.Task_Network import Task_Nerwork
from Utils.data import DataController
from Utils import config
import numpy as np

import tensorflow as tf
from tqdm import tqdm
import random


def data_sampler(prev_attacks, new_attack, normals ,new_attack_size = 128, min_prev_size = 20,
                  max_benign = 2000, validation_size = 1000, test_size = 1000):
    new_attack_controller = DataController(batch_size=new_attack_size, data_list=[new_attack], mode='binary',
                                          should_report=False, report_path='', flatten=False, max_attack_flow=2000)
    prev_attacks_controllers = []
    prev_batch_size = int(max(min_prev_size, new_attack_size/len(prev_attacks)))
    for prev_attack in prev_attacks:
        prev_attacks_controllers.append(DataController(batch_size=prev_batch_size, data_list= [prev_attack], mode='binary',
                                should_report=False, report_path='', flatten=False, max_attack_flow=2000))
    benign_controller = DataController(batch_size= int(min(new_attack_size * 2 , max_benign)), data_list= normals, mode='binary',
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
                                          should_report=False, report_path='', flatten=False, max_attack_flow=2000)
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






def create_base_cnn(data_list,path,max_attack_flows = 3000):
    best_validation = 0
    best_model = None
    best_controller = None
    number_of_train_samples = 0
    number_of_test_samples = 0
    learning_rates = [1e-3,1e-2]
    found = False
    middle = 256
    for iter in range(3):
      if found is True :
        break
      for learning_rate in learning_rates:
        if found is True:
          break
        for batch_size in config.base_batch_sizes:
          controller = DataController(batch_size=batch_size, data_list=data_list, mode='binary',
                                      should_report=True,report_path= path, flatten=False, max_attack_flow=max_attack_flows)
          
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
      
          optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate) # define our optimizer
      
          
          
          
          if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
          for epoch in range(config.epochs):
              print("epoch : ",epoch + 1)
              controller.reset()
              for idx in tqdm(range(0, controller.train_n_batches, 1)):
                  data = controller.generate('train')
                  if data is False:
                      break
                  flows = data['x']
                  labels = data['y'].reshape(-1,1)
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
              test_flows = tf.convert_to_tensor(np.expand_dims(data['x'],axis=-1), dtype=tf.float32)
              labels = data['y']
              output = base_cnn_model.predict(test_flows)
              predicted_outputs = tf.argmax(output, axis=1)
          
              number_of_samples += len(labels)
              for i in range(len(predicted_outputs)):
                  if predicted_outputs[i] == labels[i]:
                      number_of_corrects += 1
          
          print("number of test samples: ",number_of_samples )
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
        test_flows = tf.convert_to_tensor(np.expand_dims(data['x'],axis=-1), dtype=tf.float32)
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
    return best_model,best_controller




                                   



def experiment(base_index = 0, start_index = 0, build_base = True , path_base = 'base-memory-cnn', path_fisher = 'fisher.npy'):
    attacks = [ 'attack_bot', 'attack_DDOS',\
                'attack_portscan', 'DOS_SlowHttpTest',\
                'DOS_SlowLoris', 'DOS_Hulk', 'DOS_GoldenEye', 'FTPPatator',\
                'SSHPatator', 'Web_BruteForce', 'Web_XSS']
    #normals = ['vectorize_friday/benign','Benign_Wednesday','Benign_Tuesday','Benign_Thursday']
    normals = ['vectorize_friday/benign','Benign_Wednesday']
    flow_labels = [normal for normal in normals]
    flow_labels.append(attacks[base_index])
    if build_base:
      fisher_controller = DataController(batch_size=1, data_list= flow_labels, mode='binary',
                                      should_report=False, report_path='', flatten=False,
                                      max_attack_flow=2000)
      base_cnn,chosen_controller = create_base_cnn(flow_labels, path='base1-cnn.txt')
      base_cnn.save(path_base)
      
      inst = Task_Nerwork(penalty = 0.001, conv2d_filters = [8,16], number_of_train_samples = len(chosen_controller.trainIDs))
      inst.initialize_weights(input_model = base_cnn)
      inst.update_fisher_matrix(model = None, controller = fisher_controller)
      with open('fisher.npy', 'wb') as f:
        np.save(f, inst.fisher_matrix)
    else:
      inst = Task_Nerwork(penalty = 0.001, conv2d_filters = [8,16], number_of_train_samples = 1562)
      inst.initialize_weights(path = path_base)
      inst.fisher_matrix = np.load('fisher.npy', allow_pickle= True)
    
    
    #print(inst.max_fisher)
    
    
    
    prev_attacks = [attacks[base_index]]
    for i in range(start_index,len(attacks)):
      if i == base_index:
        continue
      new_flows, new_labels, validation_flows, validation_labels, new_test_flows, new_test_labels, prev_test_flows, prev_test_labels, benign_test_flows, benign_test_labels = data_sampler(prev_attacks= prev_attacks, new_attack=attacks[i], normals=normals)
      
      acc_prev, acc_new = inst.add_flows(new_flows, new_labels, validation_flows, validation_labels, new_test_flows,
                                         new_test_labels, prev_test_flows, prev_test_labels, benign_test_flows,
                                         benign_test_labels, train_attacks=flow_labels, target_attacks=[attacks[i]],
 
                                         path='../../Results/results-cnn-stepwise/exp' + str(base_index) + str(i) + '.txt', save = True)
      prev_attacks.append(attacks[i])     
                                         
                                         

for i in range(11):
  experiment(i)
