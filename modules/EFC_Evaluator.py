import pandas as pd
import os
import yaml
from EFC_Simulation import *
import sys

from enum import Enum

EVAL_TYPE = Enum('EVAL_TYPE', 'normal stepped')

eval_folder = 'evaluation'

mode = EVAL_TYPE.stepped

algorithms = ['random_walk', 'aco', 'mlp']
num_algorithm = len(algorithms)
num_pods_scenarios = [100, 250, 500, 1000, 2500]
num_scenarios = len(num_pods_scenarios)
mu = [0.5, 0.8]
num_mu = len(mu)
num_steps = 500

def start_evaluation(config, filename):
    averaging_steps_num = 10
    print('num evals: ', num_algorithm*num_scenarios*num_mu)
    average_fog_dependency = [0.0 for _ in range(num_algorithm*num_scenarios*num_mu)]
    average_cloud_dependency = [0.0 for _ in range(num_algorithm*num_scenarios*num_mu)]
    average_edge_cpu_utilization = [0.0 for _ in range(num_algorithm*num_scenarios*num_mu)]
    average_edge_memory_utilization = [0.0 for _ in range(num_algorithm*num_scenarios*num_mu)]
    average_executed_pods_ratio = [0.0 for _ in range(num_algorithm*num_scenarios*num_mu)]
    algorithms_ = ['' for _ in range(num_algorithm*num_scenarios*num_mu)]
    mu_ = [0.0 for _ in range(num_algorithm*num_scenarios*num_mu)]
    num_pods_scenarios_ = [0 for _ in range(num_algorithm*num_scenarios*num_mu)]

    config['randomized_pod_interval'] = True
    config['behaviour_type']
    config['import_graph'] = True
    config['graph_sample'] = 'EFC1'
    config['nn_sample'] = 'mlp_config_20241212_1452.yaml'
    config['plot_all'] = False
    config['mode'] = 'normal'
    config['path_to_dataset'] = 'generated/dataset'

    # start the evaluation
    for a in range(0, len(algorithms)):
        for p in range(0, len(num_pods_scenarios)):
            for m in range(0, len(mu)):
                config['behaviour_type'] = algorithms[a]
                config['sim_steps'] = 10 #int(num_pods_scenarios[p] / 10)
                config['num_pods'] = num_pods_scenarios[p]
                config['mu'] = mu[m]
                for i in range(averaging_steps_num):
                    sim = EdgeFogCloud_Simulation(config)
                    sim.start()
                    [executed_ratio, edge_cpu_utilization, edge_memory_utilization, fog_dependency, cloud_dependency] = sim.get_statitics()
                    average_fog_dependency[a*len(num_pods_scenarios)*len(mu)+p*len(mu)+m] += fog_dependency
                    average_cloud_dependency[a*len(num_pods_scenarios)*len(mu)+p*len(mu)+m] += cloud_dependency
                    average_edge_cpu_utilization[a*len(num_pods_scenarios)*len(mu)+p*len(mu)+m] += edge_cpu_utilization
                    average_edge_memory_utilization[a*len(num_pods_scenarios)*len(mu)+p*len(mu)+m] += edge_memory_utilization
                    average_executed_pods_ratio[a*len(num_pods_scenarios)*len(mu)+p*len(mu)+m] += executed_ratio
                    del sim
                average_fog_dependency[a*len(num_pods_scenarios)*len(mu)+p*len(mu)+m] /= averaging_steps_num
                average_cloud_dependency[a*len(num_pods_scenarios)*len(mu)+p*len(mu)+m] /= averaging_steps_num
                average_edge_cpu_utilization[a*len(num_pods_scenarios)*len(mu)+p*len(mu)+m] /= averaging_steps_num
                average_edge_memory_utilization[a*len(num_pods_scenarios)*len(mu)+p*len(mu)+m] /= averaging_steps_num
                algorithms_[a*len(num_pods_scenarios)*len(mu)+p*len(mu)+m] = algorithms[a]
                mu_[a*len(num_pods_scenarios)*len(mu)+p*len(mu)+m] = mu[m]
                num_pods_scenarios_[a*len(num_pods_scenarios)*len(mu)+p*len(mu)+m] = num_pods_scenarios[p]

    # print('algorithms len: ', len(algorithms_))
    # print('mu len: ', len(mu_))
    # print('num_pods_scenarios len: ', len(num_pods_scenarios_))
    # print('average_fog_dependency len: ', len(average_fog_dependency))
    # print('average_cloud_dependency len: ', len(average_cloud_dependency))
    # print('average_edge_cpu_utilization len: ', len(average_edge_cpu_utilization))
    # print('average_edge_memory_utilization len: ', len(average_edge_memory_utilization))
    # create a dictionary
    data = {
        'Algorithm': algorithms_,
        'Num Pods': num_pods_scenarios_,
        'mu': mu_,
        # 'Steps': [i for i in range(1, num_steps + 1)],
        'Fog Dependency': average_fog_dependency,
        'Cloud Dependency': average_cloud_dependency,
        'Edge CPU Utilization': average_edge_cpu_utilization,
        'Edge Memory Utilization': average_edge_memory_utilization,
        'Executed Pods Ratio': average_executed_pods_ratio
    }

    # create a dataframe from the dictionary
    df = pd.DataFrame(data)
    # create a folder if it does not exist
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)
    file = os.path.join(eval_folder, filename)
    # write dataframe to csv file
    df.to_csv(file, index=False)

def start_stepped_evaluation(config, filename):
    print('num evals: ', num_algorithm*num_scenarios*num_mu)
    fog_dependency_ = [0.0 for _ in range(num_algorithm*num_scenarios*num_mu*num_steps)]
    cloud_dependency_ = [0.0 for _ in range(num_algorithm*num_scenarios*num_mu*num_steps)]
    edge_cpu_utilization_ = [0.0 for _ in range(num_algorithm*num_scenarios*num_mu*num_steps)]
    edge_memory_utilization_ = [0.0 for _ in range(num_algorithm*num_scenarios*num_mu*num_steps)]
    executed_pods_ratio_ = [0.0 for _ in range(num_algorithm*num_scenarios*num_mu*num_steps)]
    steps_ = [0 for _ in range(num_algorithm*num_scenarios*num_mu*num_steps)]
    algorithms_ = ['' for _ in range(num_algorithm*num_scenarios*num_mu*num_steps)]
    mu_ = [0.0 for _ in range(num_algorithm*num_scenarios*num_mu*num_steps)]
    num_pods_scenarios_ = [0 for _ in range(num_algorithm*num_scenarios*num_mu*num_steps)]

    config['randomized_pod_interval'] = True
    config['behaviour_type']
    config['import_graph'] = True
    config['graph_sample'] = 'EFC1'
    config['nn_sample'] = 'mlp_config_20241212_1452.yaml' # 'mlp_config20:05:26.085489.yaml'
    config['plot_all'] = False
    config['mode'] = 'normal'
    config['path_to_dataset'] = 'generated/dataset'
    config['sim_steps'] = num_steps

    # start the evaluation
    for a in range(0, len(algorithms)):
        for p in range(0, len(num_pods_scenarios)):
            for m in range(0, len(mu)):
                config['behaviour_type'] = algorithms[a]
                config['num_pods'] = num_pods_scenarios[p]
                config['pods_per_step'] = int(num_pods_scenarios[p] / 100)
                config['mu'] = mu[m]
                sim = EdgeFogCloud_Simulation(config)
                for s in range(0, num_steps):
                    sim._step()
                    [executed_ratio, edge_cpu_utilization, edge_memory_utilization, fog_dependency, cloud_dependency] = sim.get_statitics()
                    fog_dependency_[a*len(num_pods_scenarios)*len(mu)*num_steps+p*len(mu)*num_steps+m*num_steps+s] = fog_dependency
                    cloud_dependency_[a*len(num_pods_scenarios)*len(mu)*num_steps+p*len(mu)*num_steps+m*num_steps+s] = cloud_dependency
                    edge_cpu_utilization_[a*len(num_pods_scenarios)*len(mu)*num_steps+p*len(mu)*num_steps+m*num_steps+s] = edge_cpu_utilization
                    edge_memory_utilization_[a*len(num_pods_scenarios)*len(mu)*num_steps+p*len(mu)*num_steps+m*num_steps+s] = edge_memory_utilization
                    executed_pods_ratio_[a*len(num_pods_scenarios)*len(mu)*num_steps+p*len(mu)*num_steps+m*num_steps+s] = executed_ratio*100
                    
                    algorithms_[a*len(num_pods_scenarios)*len(mu)*num_steps+p*len(mu)*num_steps+m*num_steps+s] = algorithms[a]
                    mu_[a*len(num_pods_scenarios)*len(mu)*num_steps+p*len(mu)*num_steps+m*num_steps+s] = mu[m]
                    num_pods_scenarios_[a*len(num_pods_scenarios)*len(mu)*num_steps+p*len(mu)*num_steps+m*num_steps+s] = num_pods_scenarios[p]
                    steps_[a*len(num_pods_scenarios)*len(mu)*num_steps+p*len(mu)*num_steps+m*num_steps+s] = s
                del sim

    # create a dictionary
    data = {
        'Algorithm': algorithms_,
        'Num Pods': num_pods_scenarios_,
        'mu': mu_,
        'Step': steps_,
        'Fog Dependency': fog_dependency_,
        'Cloud Dependency': cloud_dependency_,
        'Edge CPU Utilization': edge_cpu_utilization_,
        'Edge Memory Utilization': edge_memory_utilization_,
        'Executed Pods Ratio': executed_pods_ratio_
    }

    # create a dataframe from the dictionary
    df = pd.DataFrame(data)
    # create a folder if it does not exist
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)
    file = os.path.join(eval_folder, filename)
    # write dataframe to csv file
    df.to_csv(file, index=False)

if __name__ == '__main__':
    # receive file name as argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'eval.csv'
    sim_config = yaml.safe_load(open("config/config.yml"))
    
    if mode == EVAL_TYPE.normal:
        start_evaluation(sim_config, filename)
    else:
        start_stepped_evaluation(sim_config, filename)