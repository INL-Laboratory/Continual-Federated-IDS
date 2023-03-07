flow_size = 100
pkt_size = 200

epochs = 50
task_epochs = 20
base_batch_size = 32
new_task_batch_size = 16
l1_lambda = 0.001
t_labels = ['benign', 'attack_bot', 'attack_DDOS',\
            'attack_portscan']
all_labels = ['vectorize_friday/benign', 'attack_bot', 'attack_DDOS',\
            'attack_portscan', 'Benign_Wednesday', 'DOS_SlowHttpTest',\
            'DOS_SlowLoris', 'DOS_Hulk', 'DOS_GoldenEye', 'FTPPatator',\
            'SSHPatator', 'Web_BruteForce', 'Web_XSS']
            

