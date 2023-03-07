flow_size = 100
pkt_size = 200

epochs = 50
task_epochs = 20
base_batch_sizes = [32,16,64,128,8]
task_batch_sizes =  [8,16,32]
task_compress_batch_sizes =  [8,16,32]
#task_batch_sizes = [16,4,64,128]
fisher_coeffs = [1,10,0.001,1e-6]
base_batch_size = 32
#base_batch_sizes = [32]
new_task_batch_size = 16
l1_lambda = 0.001
t_labels = ['benign', 'attack_bot', 'attack_DDOS',\
            'attack_portscan']
all_labels = ['vectorize_friday/benign', 'attack_bot', 'attack_DDOS',\
            'attack_portscan', 'Benign_Wednesday', 'DOS_SlowHttpTest',\
            'DOS_SlowLoris', 'DOS_Hulk', 'DOS_GoldenEye', 'FTPPatator',\
            'SSHPatator', 'Web_BruteForce', 'Web_XSS']
            
