class Reporter(object):
    # eventually an instance of this class will save a report file regarding our model, containing the flows used and other parameters
    def __init__(self):
        self.attacks = {} # attack_name: [[day,number of flow samples]]
        self.normals = {} # source of flows : number of flows
        self.total_number_of_attack_flows = 0
        self.total_number_of_normal_flows = 0
    def add_attack(self,file_path,number_of_flows):
        # store an attack and the number of its flows
        elements = file_path.split('/')
        if 'friday' in file_path:
            day = 'vectorize_friday'
        else:
            day = elements[-4]
        attack = elements[-2]
        if attack in self.attacks.keys():
            self.attacks[attack].append([day,number_of_flows])
        else:
            self.attacks[attack] = [[day, number_of_flows]]


        self.total_number_of_attack_flows += number_of_flows
    def add_normals(self,file_path,number_of_flows):
        # store a normal source and the number of its flows
        elements = file_path.split('/')
        if 'friday' in file_path:
            day = 'vectorize_friday'
        else:
            day = elements[6]
        self.normals[day] = number_of_flows
        #print(file_path)
        #print(day)
        self.total_number_of_normal_flows += number_of_flows
    def report_data(self,verbose = True,save_to_file = False,file_path = 'data_report.txt'):
        # reports the distribution of data we will be using for our training
        print(self.attacks)
        text =''
        text += "############################\n"
        text += "Attacks : \n"
        for i,attack in enumerate(self.attacks.keys()):
            text += attack + ' : ' + '\n'
            for data in self.attacks[attack]:

                text += 'day : ' + data[0] + '\n'
                text += 'number of samples : ' + str(data[1]) + '\n'
            if i != len(self.attacks) - 1:
                text += "--------\n"
        text += "############################\n"
        text += "Normals : \n"
        for i,normal in enumerate(self.normals.keys()):
            text += 'day : ' + normal +'\n'
            text += 'number of samples : ' + str(self.normals[normal]) + '\n'
            if i != len(self.normals.keys()) - 1:
                text += "--------\n"
        text += "############################\n"
        total_number_of_flows = self.total_number_of_attack_flows + self.total_number_of_normal_flows
        text += 'Number of attack flows : ' + str(self.total_number_of_attack_flows) + '(' +str(int(100 *self.total_number_of_attack_flows/total_number_of_flows)) +'%)' + '\n'
        text += 'Number of normal flows : ' + str(self.total_number_of_normal_flows) +  '(' +str(int(100 *self.total_number_of_normal_flows/total_number_of_flows)) +'%)' + '\n'
        text += 'Total number flows : ' + str(total_number_of_flows) + '\n'
        if verbose:
            print(text)
        if save_to_file:
            with open(file_path, 'w') as output:
                output.write(text)


