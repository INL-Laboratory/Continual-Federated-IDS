import os, socket
import numpy as np
from Utils import config
import json
from Utils.reporter2018 import Reporter

seed = 7001
# this file has been written to access the cic-ids 2018 dataset located on our local server


with open('./classes2018.json') as classes_file:
    classes_config = json.load(classes_file)

path = '/home/behzad/University/sample_data/'
# path = '/home/behzad/University/pcap_data/'

if socket.gethostname() != 'HomeX':
    path = '/mnt/inl-backup/bkp-Home/home-old/soltani/vectorize/'  # 2018


if socket.gethostname() == 'soltani-server':
    path = '/backup3/iscxIDS2017/pcap/'

if socket.gethostname() == 'fateme-X456UQK':
    path = '/home/fateme/study/Thesis/data/'
data_str = ''

flow_size = config.flow_size
pkt_size = config.pkt_size


class DataController(object):
    def __init__(self, data_str=data_str, batch_size=20, data_list=[], mode='binary', reporter=None,
                 should_report=False, report_path='data_report.txt', flatten=True, max_attack_flow=1000,
                 with_meta_data=False):
        # mode will be checked to adjust the number of samples taken from normal flows vs abnormal flows (to avoid bias)
        # reporter will report the amount of data allotted to each class
        # flatten will show if we should flatten the batches to one vector or not
        super(DataController, self).__init__()
        self.batch_size = batch_size
        self.mode = mode
        self.flatten = flatten
        self.train_counter = 0
        self.validation_counter = 0
        self.test_counter = 0
        self.full_counter = 0
        self.number_of_classes = 1
        self.with_meta_data = with_meta_data
        if reporter is None:
            reporter = Reporter()
        data_dirs = []
        if len(data_list) == 0:
            data_dirs = data_str.split('?')
        else:
            if self.mode == 'binary':
                self.number_of_classes = 2
            for i, d in enumerate(data_list):
                data_dirs.append(path + classes_config[d]['path'])
                if 'benign' in str(classes_config[d]['path']).lower():
                    classes_config[d]['label'] = 0
                elif self.mode == 'binary':
                    classes_config[d]['label'] = 1
                elif i != 0:
                    classes_config[d]['label'] = self.number_of_classes
                    self.number_of_classes += 1
                else:
                    classes_config[d]['label'] = len(data_list)
        files = list()
        for directory in data_dirs:
            files.append(os.listdir(directory))

        # n_per_label = 10000
        # for file in files:
        #     n_per_label = min(n_per_label, len(file))

        # we will iterate on the files twice
        # first we will obtain data of the attacks
        # then if on binary mode, we will use size(attacks)  benign flows
        # if we are on multi-class mode, we will use 10000 flows obtaibengin flowsned from the
        data_files = list()
        number_of_attack_samples = 0
        number_of_attacks = 0
        number_of_benign = 0
        indexes_of_benigns = []
        self.max_attack_flow = max_attack_flow
        for k in range(0, files.__len__()):
            if 'benign' not in data_dirs[k].lower():
                # print(files[k].__len__())
                for j in range(0, min(max_attack_flow, files[k].__len__())):
                    data_files.append(data_dirs[k] + files[k][j])
                number_of_attack_samples += min(max_attack_flow, files[k].__len__())
                number_of_attacks += 1
                reporter.add_attack(file_path=data_dirs[k], number_of_flows=min(max_attack_flow, files[k].__len__()))
            else:
                number_of_benign += 1
                indexes_of_benigns.append(k)
        print("****")
        if number_of_benign != 0:
            if self.mode == 'binary':
                # the number of normal flows must be equal to the number of abnormal flows
                # in case of no attacks, the number of benign flows will be max_attack_flow
                # we try to absorb an equal amount of normal flows from each of our normal flow resources
                # e.g. newwedensday, vectorize friday and etc
                if number_of_attacks != 0:
                    size_of_each_benign = int(number_of_attack_samples / number_of_benign)
                else:
                    size_of_each_benign = int(max_attack_flow / number_of_benign)
            else:  # multiclass
                # we want our normal class to have equal samples with other classes
                actual_size_of_benign = int(number_of_attack_samples / number_of_attacks)
                size_of_each_benign = int(actual_size_of_benign / number_of_benign)
            number_of_benign_samples = 0
            list_of_sizes = [files[index].__len__() for index in indexes_of_benigns]
            # sorting the indexes based on their number of flows
            indexes_of_benigns = [x for _, x in sorted(zip(list_of_sizes, indexes_of_benigns))]
            for i, index in enumerate(indexes_of_benigns):
                for j in range(0, min(size_of_each_benign, files[index].__len__())):
                    data_files.append(data_dirs[index] + files[index][j])
                number_of_benign_samples += min(size_of_each_benign, files[index].__len__())
                reporter.add_normals(file_path=data_dirs[index],
                                     number_of_flows=min(size_of_each_benign, files[index].__len__()))
                # if this resource had less flows than we wanted, we'll take more flows from the next resources
                if i != len(indexes_of_benigns) - 1:
                    if number_of_attacks != 0:
                        if self.mode == 'binary':
                            size_of_each_benign = int(
                                (number_of_attack_samples - number_of_benign_samples) / (
                                            len(indexes_of_benigns) - (i + 1)))
                        else:
                            size_of_each_benign = int(
                                (actual_size_of_benign - number_of_benign_samples) / (
                                            len(indexes_of_benigns) - (i + 1)))
                    else:
                        size_of_each_benign = int(
                            (max_attack_flow - number_of_benign_samples) / (len(indexes_of_benigns) - (i + 1)))

        if should_report:
            reporter.report_data(save_to_file=True, file_path=report_path)

        np.random.seed(seed)
        np.random.shuffle(data_files)
        self.data_files = data_files
        cut = int(len(data_files) * 0.8)
        nonTestIDs = data_files[:cut]
        self.testIDs = data_files[cut:]
        cut2 = int(len(nonTestIDs) * 0.8)
        self.trainIDs = nonTestIDs[:cut2]
        self.validIDs = nonTestIDs[cut2:]

        self.trainIDs = self.trainIDs[: len(self.trainIDs) - len(self.trainIDs) % batch_size]
        self.validIDs = self.validIDs[: len(self.validIDs) - len(self.validIDs) % batch_size]
        self.testIDs = self.testIDs[: len(self.testIDs) - len(self.testIDs) % batch_size]

        self.train_n_batches = len(self.trainIDs) // batch_size
        self.validation_n_batches = len(self.validIDs) // batch_size
        self.test_n_batches = len(self.testIDs) // batch_size
        self.full_n_batches = len(self.data_files) // batch_size

    # print ('train size:' , len(self.trainIDs), 'batches:', self.train_n_batches)
    # print ('validation size', len(self.validIDs), 'batches:', self.validation_n_batches)
    # print ('test size:' , len(self.testIDs), 'batches:', self.test_n_batches)

    def generate(self, mode='full', output_model='label'):
        # output model shows if we want our y's to be just labels or one hot vectors showing probability
        num = 1
        if mode == 'full':
            target_set = self.data_files
            counter = self.full_counter
            n_batches = self.full_n_batches
            self.full_counter += 1
        elif mode == 'validation':
            target_set = self.validIDs
            counter = self.validation_counter
            n_batches = self.validation_n_batches
            self.validation_counter += 1
        elif mode == 'test':
            target_set = self.testIDs
            counter = self.test_counter
            n_batches = self.test_n_batches
            self.test_counter += 1
        elif mode == 'train':
            target_set = self.trainIDs
            counter = self.train_counter
            n_batches = self.train_n_batches
            self.train_counter += 1
        else:
            raise ValueError('DataGenerator mode not defined')

        start_index = counter * num * self.batch_size
        end_index = (counter + 1) * num * self.batch_size

        if counter < n_batches:

            # print('data: {}/{}'.format(counter+1, n_batches))
            files = target_set[start_index: end_index]
        else:
            # print('No more data ...')
            return False

        set_x, set_y, filenames = [], [], []
        for file in files:
            X, Y = self.parse_flow(file)
            set_x.append(X)
            set_y.append(Y)
            filenames.append(file)

        set_x = np.array(set_x)
        set_y = np.array(set_y)
        if output_model != 'label' and len(set_y) != 0:
            y_probs = []
            if self.mode == 'binary':
                for label in set_y:
                    if label == 0:
                        y_probs.append([1, 0])
                    else:
                        y_probs.append([0, 1])
            else:
                for label in set_y:
                    new_y = [0 for i in range(self.number_of_classes)]
                    new_y[label] = 1
                    y_probs.append(new_y[label])
            set_y = np.array(y_probs)
        filenames = np.array(filenames)
        return {
            'counter': counter,
            'x': set_x,
            'y': set_y,
            'filenames': filenames,
            'start_index': start_index,
            'end_index': end_index}

    def parse_flow(self, filename):
        X = list()
        Y = list()
        with open(filename) as flowfile:
            label = self.get_label_from_name(filename)
            Y.append(label)
            flowmatrix = list()
            counter = 0

            for line in flowfile:
                if counter == flow_size:
                    break
                line = line[:-1].split(',')
                for i in range(len(line)):
                    line[i] = float(line[i])
                if line[0] < 0:
                    line[0] = 0.0
                if line[0] > 0.5:
                    line[0] = 1
                else:
                    line[0] = line[0] / 0.5
                for i in range(pkt_size - len(line)):  # append 0 to packets which are smaller than packetsize(max=1500)
                    line.append(0.0)
                line = line[:pkt_size]
                for i in range(11,
                               21):  # masking src/dst ip to 0 (12,20)+1(for timediff) ,,, maksing checksum to 0 (10,12)+1=> (11,21) -- (1,41): just payload
                    line[i] = 0.0
                flowmatrix.append(line)
                counter += 1
            for i in range(flow_size - len(flowmatrix)):
                flowmatrix.append([0] * pkt_size)
            flowmatrix = np.array(flowmatrix)
        if self.flatten:
            X_train = np.reshape(flowmatrix, (-1, flowmatrix.shape[0] * flowmatrix.shape[1]))
        else:
            if self.with_meta_data:
                X_train = flowmatrix
            else:  # in this case, since we only want the packets, we exclude the first row which contains meta data
                X_train = flowmatrix[1:]
        Y_train = np.array(Y)
        # print(X_train.shape)
        # print(Y_train.shape)
        if len(X_train) != 0 and len(Y_train) != 0:
            if self.flatten:
                data_x, data_y = (X_train[0], Y_train[0])
            else:
                data_x = X_train
                data_y = Y_train[0]
        else:
            data_x, data_y = [], []
        # data_x = np.reshape(data_x, (flow_size, pkt_size))
        return data_x, data_y

    def reset(self):
        self.train_counter = 0
        self.validation_counter = 0
        self.test_counter = 0
        self.full_counter = 0

    def get_n_samples(self, mode):
        if mode == 'validation':
            target_set = self.validIDs
            counter = self.validation_counter
            n_batches = self.validation_n_batches
            self.validation_counter += 1
        elif mode == 'test':
            target_set = self.testIDs
            counter = self.test_counter
            n_batches = self.test_n_batches
            self.test_counter += 1
        else:
            target_set = self.trainIDs
            counter = self.train_counter
            n_batches = self.train_n_batches
            self.train_counter += 1
        return len(self.trainIDs)

    def get_label_from_name(self, filename):
        for clas in classes_config:
            if filename.__contains__(classes_config[clas]['path']):
                return classes_config[clas]['label']