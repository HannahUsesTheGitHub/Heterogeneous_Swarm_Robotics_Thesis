from experiment import main
import random
import os
import numpy as np

#random.seed(1) # uncomment to set random seed 

#there are different ways to set weights, in the final thesis only the uniform weight per seed were used

settings = [{'baseline': False, 'alignment': True, 'communication': False, 'folder_name': 'aTcF'}]

swarms = [10, 20, 30]
seeds = [i for i in range(30)]

weights_list = []
for i in range(30):
    incoming_weights = {}
    for j in range(30):
        if i == j: incoming_weights.setdefault(str(j), 1)
        else: incoming_weights.setdefault(str(j), random.uniform(0,1))
    weights_list.append(incoming_weights)
    

try: os.mkdir('experiment_folder')
except: pass

for setup in settings:
    try:
        foldername = 'experiment_folder' + '/' + setup['folder_name']
        os.mkdir(foldername)
    except: pass
    for swarm_size in swarms:


        # individual weights per swarm:

        #uniform weights:
        # weights_list = []
        # for i in range(swarm_size):
        #     incoming_weights = {}
        #     for j in range(swarm_size):
        #         if i == j: incoming_weights.setdefault(str(j), 1)
        #         else: incoming_weights.setdefault(str(j), random.uniform(0,1))
        #     # for j in range(swarm_size):
        #     #     if i == j: 
        #     #         incoming_weights.setdefault(str(j), 1)
        #     #     else: 
        #     #         incoming_weights.setdefault(str(j), X[index])
        #     #         index += 1
        #     weights_list.append(incoming_weights)


        # bimodal weights
        # N=swarm_size
        # mu, sigma = 0.3, 0.05
        # mu2, sigma2 = 0.7, 0.05
        # X1 = np.random.normal(mu, sigma, N)
        # X2 = np.random.normal(mu2, sigma2, N)
        # X = np.concatenate([X1, X2])
        # print(X)
        # plt.hist(X)
        # index = 0
        # weights_list = []
        # for i in range(swarm_size):
        #     incoming_weights = {}
        #     for j in range(swarm_size):
        #         if i == j: 
        #             incoming_weights.setdefault(str(j), 1)
        #         else: 
        #             incoming_weights.setdefault(str(j), X[index])
        #             index += 1
        #     weights_list.append(incoming_weights)
        # print(weights_list)
        
        dir_name = 'experiment_folder' + '/' + setup['folder_name'] + '/' + str(swarm_size)
        try: os.mkdir(dir_name)
        except: pass

        for seed in seeds:
            # random.seed(seed)
            #create weights
            N=int((swarm_size*swarm_size)/2)
            mu, sigma = 0.3, 0.05
            mu2, sigma2 = 0.7, 0.05
            X1 = np.random.normal(mu, sigma, N)
            X2 = np.random.normal(mu2, sigma2, N)
            X = np.concatenate([X1, X2])
            #print(X)
            #plt.hist(X)

            index = 0
            weights_list = []
            for i in range(swarm_size):
                incoming_weights = {}

                #bimodal weights
                # for j in range(swarm_size):
                #     if i == j: incoming_weights.setdefault(str(j), 1)
                #     else: incoming_weights.setdefault(str(j), random.uniform(0,1))

                #uniform weights
                for j in range(swarm_size):
                    if i == j: 
                        incoming_weights.setdefault(str(j), 1)
                    else: 
                        incoming_weights.setdefault(str(j), X[index])
                        index += 1


                weights_list.append(incoming_weights)
            #print(weights_list)

            dir_name = 'experiment_folder' + '/' + setup['folder_name'] + '/' + str(swarm_size) + '/' + str(seed)
            try: os.mkdir(dir_name)
            except: pass

            print(f'RUNNING SEED {seed}, SIZE {swarm_size}, SETTINGS {setup['folder_name']}')
            main(seed=seed, baseline=setup['baseline'], alignment=setup['alignment'], communication=setup['communication'], weights=weights_list, swarm_size=swarm_size, folder_name=dir_name)



