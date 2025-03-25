import os
import sys
from mesa import Agent, Model
from mesa.time import RandomActivation
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from EFC_Plotting import *
from EFC_FrevoIPC import FrevoInterface
from interfaces import config_pb2, candidate_pb2, stats_pb2

from EFC_Agents import *
from EFC_Algorithms import *
from EFC_Model import *
from EFC_FrevoIPC import FrevoInterface

# test
import gc
# from memory_profiler import profile
import traceback

class EdgeFogCloud_Simulation():
    def __init__(self, config):
        self._init_from_config(config)
        self.pods_created = 0  # Track the number of pods created
        self.status = False
        self.arrive_time_point = 0
        if self.pods_per_step == 0:
            self.pods_per_step = self.num_pods / self.sim_steps # TMP - doesn't work when num_pods < sim_steps
        self.pod_counter = 0  # Initialize pod counter
        self.pods = []  # List to store created pods in order
        self.schedule = RandomActivation(self.model)
        self._init_model()

    def _init_model(self):
        self.model.init_model(schedule=self.schedule)
    
    def _init_from_config(self, config):
        self.sim_steps = config['sim_steps']
        self.num_pods = config['num_pods']
        self.inter_arrival_mu = config['mu']
        self.randomized_interval = config['randomized_pod_interval']
        self.frevo_training = False
        self.generate_samples = False
        self.ipc = None
        self.candidate_evals = 0
        self.training_results_path = None
        self.pods_per_step = config['pods_per_step']

        behaviour_type = config['behaviour_type']
        import_sample = config['import_graph']
        name_sample = None
        if import_sample:
            name_sample = config['graph_sample']

        mode = config['mode']
        self.plot_all = config['plot_all']
        if mode == 'train':
            self.frevo_training = True
            self.candidate_evals = config['candidate_evals']
            self.training_results_path = config['training_results_path']
            self.evaluate_best_candidate = config['evaluate_best_candidate']
        elif mode == 'gen':
            self.generate_samples = True
            self.gen_samples_num = config['gen_samples_num']
            import_sample = False
            name_sample = None
    
        path_to_dataset = config['path_to_dataset']
        if len(path_to_dataset) > 0 and not os.path.exists(path_to_dataset): # optional protection
            try: 
                os.makedirs(path_to_dataset)
            except OSError:
                print("Could not create directory for the dataset generation: " + path_to_dataset)
                sys.exit()
        # create model
        num_edge_devices = config['num_edge_devices']
        num_fog_nodes = config['num_fog_nodes']
        num_cloud_servers = config['num_cloud_servers']
        datacollector_enabled = False if self.frevo_training else self.plot_all # collecting data only if we plot and not in training mode
        self.model = EdgeFogCloud_Continuum(behaviour_type, num_edge_devices, num_fog_nodes, num_cloud_servers, import_sample, name_sample, path_to_dataset, datacollector_enabled)
        # create IPC
        if self.frevo_training:
            if behaviour_type != 'mlp':
                print("Wrong mode chosen. Frevo training is only supported for MLP behaviour type. Exiting.")
                sys.exit()
            self.ipc_port = config['frevo_port']
            self._create_frevo_session()
        
        # create behaviour profile
        self.behaviour_profile = PodBehaviourProfile(behaviour_type=behaviour_type, frevo_training=self.frevo_training, ipc=self.ipc)
        if behaviour_type == 'mlp' and not self.frevo_training:
            nn_sample = config['nn_sample']
            self.behaviour_profile.load_nn_from_sample(nn_sample)
    
    def _create_frevo_session(self):
        session_completed = False
        # print("Creating Frevo session")
        try:
            self.ipc = FrevoInterface(port=self.ipc_port)
        except Exception as e:
            print(f"Frevo training finished: connection was closed.")
            session_completed = True
            # sys.exit()
        return session_completed
    
    def _init_frevo_session(self):
        # print("Initiating Frevo session")
        ipc_config = config_pb2.Config()
        ipc_config.steps = self.sim_steps
        self.ipc.send_over_socket(ipc_config.SerializeToString())
            
    def _end_frevo_session(self):
        # print("Ending Frevo session")
        request = request_pb2.Request()
        request.last = True
        self.ipc.send_over_socket(request.SerializeToString())
    
    def _normalize_metrics(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)
    
    def _send_frevo_training_stats(self, utilization, fog_count, cloud_count, num_pods):
        # print("Sending Frevo training stats")
        stats = stats_pb2.Stats()
        # TMP: Manual scaling because of not enough sim steps
        if utilization.get("overconsumption_penalty", True):
            stats.edge_cpu_utilization = 0
            stats.edge_memory_utilization = 0
            stats.fog_cpu_utilization = 0
            stats.fog_memory_utilization = 0
            stats.fog_dependency = 10000000 # will be converted to a very low value
            stats.cloud_dependency = 0
            stats.executed_pods_ratio = 0
        else:
            edge_cpu_utilization = self._normalize_metrics(utilization["edge_cpu_utilization"], 0, 1)
            edge_memory_utilization = self._normalize_metrics(utilization["edge_memory_utilization"], 0, 1)
            fog_cpu_utilization = self._normalize_metrics(utilization["fog_cpu_utilization"], 0, 1)
            fog_memory_utilization = self._normalize_metrics(utilization["fog_memory_utilization"], 0, 1)
            cloud_cpu_utilization = self._normalize_metrics(utilization["cloud_cpu_utilization"], 0, 1)
            cloud_memory_utilization = self._normalize_metrics(utilization["cloud_memory_utilization"], 0, 1)
            fog_dependency = self._normalize_metrics(fog_count / num_pods, 0, 1)
            cloud_dependency = self._normalize_metrics(cloud_count / num_pods, 0, 1)
            executed_pods_ratio = self._normalize_metrics(utilization["executed_pods_ratio"], 0, 1)
            
            stats.edge_cpu_utilization = edge_cpu_utilization
            stats.edge_memory_utilization = edge_memory_utilization
            stats.fog_cpu_utilization = fog_cpu_utilization
            stats.fog_memory_utilization = fog_memory_utilization
            stats.fog_dependency = fog_dependency
            stats.cloud_dependency = cloud_dependency
            stats.executed_pods_ratio = executed_pods_ratio
        self.ipc.send_over_socket(stats.SerializeToString())
        self.ipc.close_channel()
    
    # this code is preferred because we are removing the executed agents from the schedule
    def _get_executed_pods(self):
        return self.model.executed_pods
    
    def get_statitics(self):
        num_pods = self.num_pods
        # executed_pods = self.get_executed()
        # print("Executed / Issued Pods:", executed_pods, " / ", self.pod_counter)
        final_pod_positions, fog_count, cloud_count = self._get_final_pod_positions_and_fog_count()
        print("Number of Pods on Fog Nodes:", fog_count)
        print("Number of Pods on Cloud Nodes:", cloud_count)

        # Get and print the paths of all pods
        pod_paths = self._get_pod_paths()
        print("Pod Paths and Transport Costs:")
        print("Edge Paths:", pod_paths['edge'])
        print("Fog Paths:", pod_paths['fog'])
        print("Cloud Paths:", pod_paths['cloud'])
        print("Edge to Fog Transitions:", pod_paths['edge_to_fog'])
        print("Fog to Cloud Transitions:", pod_paths['fog_to_cloud'])
        print("Fog to Edge Transitions:", pod_paths['fog_to_edge'])
        print("Cloud to Fog Transitions:", pod_paths['cloud_to_fog'])
        print("Cloud to Edge Transitions:", pod_paths['cloud_to_edge'])
        print("Edge to Cloud Transitions:", pod_paths['edge_to_cloud'])
        
        # Calculate and print utilization
        utilization = self._calculate_utilization()
        if self.frevo_training:
            self._send_frevo_training_stats(utilization, fog_count, cloud_count, num_pods)
            if utilization.get("overconsumption_penalty", True):
                print("Overconsumption Penalty: True")
                return
                
        print("Edge CPU Utilization:", utilization["edge_cpu_utilization"])
        print("Edge Memory Utilization:", utilization["edge_memory_utilization"])
        print("Fog CPU Utilization:", utilization["fog_cpu_utilization"])
        print("Fog Memory Utilization:", utilization["fog_memory_utilization"])
        print("Cloud CPU Utilization:", utilization["cloud_cpu_utilization"])
        print("Cloud Memory Utilization:", utilization["cloud_memory_utilization"])
        print("Dependency on Fog Nodes:", fog_count/self.num_pods)
        print("Dependency on Cloud Nodes:", cloud_count/self.num_pods)
        print("Executed Pods Ratio:", utilization["executed_pods_ratio"])
        return [utilization["executed_pods_ratio"], utilization["edge_cpu_utilization"], utilization["edge_memory_utilization"], fog_count/num_pods, cloud_count/num_pods]
    
    def plot_results(self):
        plot_network(self.model)
        for layer in ['edge', 'fog', 'cloud']:
            plot_CPU_utilization_over_time(self.model, layer)
            plot_Mem_utilization_over_time(self.model, layer)
        
        plot_Comm_Cost_over_time(self.model, self)
    
    def reset_EFC_model(self):
        self.arrive_time_point = 0
        self.pod_counter = 0
        self.pods = []
        self.behaviour_profile.reinit_ipc(self.ipc)
        self.model.reinit() # resets model time and data collector
        for agent in self.schedule.agents:
            if isinstance(agent, PodAgent):
                self.schedule.remove(agent)
                agent.remove()
                del agent
            elif isinstance(agent, DeviceAgent):
                agent.available_cpu = agent.cpu
                agent.available_memory = agent.memory
       
    def _receive_candidate_profile(self):
        message = self.ipc.receive_over_socket()
        nn_config = candidate_pb2.Candidate()
        nn_config.ParseFromString(message)
        return nn_config
    
    def _start_frevo_training(self):
        epoch = 0
        session_completed = False
        while True:
            print('Size of Agents in scheduler: ' + str(len(self.schedule._agents)))
            try:
                if epoch > 0:
                    session_completed = self._create_frevo_session()
                    if session_completed:
                        break 
                    self.reset_EFC_model() # removes created pods
                    gc.collect()
                self._init_frevo_session()
                nn_config = self._receive_candidate_profile()
                # print(nn_config)
                self.behaviour_profile.load_nn_from_config(nn_config)

                # TODO: set the number of steps to the number of samples
                while self.behaviour_profile.get_candidate_evaluated() < self.candidate_evals:
                    # print(f'Candidate evaluations: {self.behaviour_profile.get_candidate_evaluated()}')
                    for i in range(self.sim_steps):
                        self._step()
                

                # self.end_frevo_session() # only when remote training
                self.get_statitics()
                epoch += 1
                sleep(1)
            except Exception as e:
                print(f"Frevo training finished: {e}")
                print(traceback.format_exc())
                break
        
        self.frevo_training = False
        if session_completed:
            self.ipc.close_channel()
            print("Frevo training session completed. Loading results.")
            self._load_best_candidate()
    
    def _load_best_candidate(self):
        print(self.training_results_path)
        if not os.path.exists(self.training_results_path):
            print("Error: directory with the training results doesn't exit: " + self.training_results_path)
            sys.exit()
        best_candidate_conf = self.behaviour_profile.load_save_best_candidate_conf(self.training_results_path)
        print(f"Best candidate: {best_candidate_conf}")
        if self.evaluate_best_candidate:
            self.behaviour_profile.load_nn_from_config(best_candidate_conf)
            if self.plot_all:
                self.model.enable_datacollector()
            self.reset_EFC_model() # needs to be done after enabling datacollector
            for i in range(self.sim_steps):
                self._step()
            self.get_statitics()
            if self.plot_all:
                self.plot_results()
    
    def start(self):
        if self.frevo_training:
            self._start_frevo_training()
            return
        elif self.generate_samples:
            self.model.generate_samples(self.gen_samples_num)
            return
        
        for i in range(self.sim_steps):
            self._step()
        
        self.get_statitics()
        if self.plot_all:
            self.plot_results()
  
    def _create_pod(self):
        pod_id = f'pod_{self.pod_counter+1}'
        pod_demand, pod_demand_step, _, _, _ = get_pod_profile()
        cpu_req = pod_demand[0]
        memory_req = pod_demand[1]
        exec_time = pod_demand_step
        node = random.choice(self.model.edge_nodes)
        
        behaviour = self.behaviour_profile.create(cpu_req, memory_req, self.model.comm_costs, self.model.network.nodes)
        pod = PodAgent(pod_id, self.model, cpu_req, memory_req, node, exec_time, behaviour)
        self.schedule.add(pod)
        self.status = True

    def _get_final_pod_positions_and_fog_count(self):
        final_positions = {}
        fog_count = 0
        cloud_count = 0
        for agent in self.schedule.agents:
            if isinstance(agent, PodAgent):
                node = agent.node
                layer = self.model.network.nodes[node]['layer']
                final_positions[agent.unique_id] = (node, layer)
                if layer == 'fog':
                    fog_count += 1
                if layer == 'cloud':
                    cloud_count += 1
        return final_positions, fog_count, cloud_count

    def _calculate_utilization(self):
        edge_cpu_total = 0
        edge_memory_total = 0
        edge_cpu_used = 0
        edge_memory_used = 0

        fog_cpu_total = 0
        fog_memory_total = 0
        fog_cpu_used = 0
        fog_memory_used = 0
        
        cloud_cpu_total = 0
        cloud_memory_total = 0
        cloud_cpu_used = 0
        cloud_memory_used = 0
        
        for agent in self.schedule.agents:
            if isinstance(agent, DeviceAgent):
                if self.frevo_training:
                    if agent.available_cpu < 0 or agent.available_memory < 0:
                        return {
                            "overconsumption_penalty": True,
                        }
                if agent.layer == 'edge':
                    edge_cpu_total += agent.cpu
                    edge_memory_total += agent.memory
                    edge_cpu_used += (agent.cpu - agent.available_cpu)
                    edge_memory_used += (agent.memory - agent.available_memory)
                elif agent.layer == 'fog':
                    fog_cpu_total += agent.cpu
                    fog_memory_total += agent.memory
                    fog_cpu_used += (agent.cpu - agent.available_cpu)
                    fog_memory_used += (agent.memory - agent.available_memory)
                elif agent.layer == 'cloud':
                    cloud_cpu_total += agent.cpu
                    cloud_memory_total += agent.memory
                    cloud_cpu_used += (agent.cpu - agent.available_cpu)
                    cloud_memory_used += (agent.memory - agent.available_memory)

        print(f"Edge CPU Total: {edge_cpu_total}, Edge CPU Used: {edge_cpu_used}")
        print(f"Edge Memory Total: {edge_memory_total}, Edge Memory Used: {edge_memory_used}")
        print(f"Fog CPU Total: {fog_cpu_total}, Fog CPU Used: {fog_cpu_used}")
        print(f"Fog Memory Total: {fog_memory_total}, Fog Memory Used: {fog_memory_used}")
        print(f"Cloud CPU Total: {cloud_cpu_total}, Cloud CPU Used: {cloud_cpu_used}")
        print(f"Cloud Memory Total: {cloud_memory_total}, Cloud Memory Used: {cloud_memory_used}")
        
        edge_cpu_utilization = edge_cpu_used / edge_cpu_total if edge_cpu_total > 0 else 0
        edge_memory_utilization = edge_memory_used / edge_memory_total if edge_memory_total > 0 else 0
        fog_cpu_utilization = fog_cpu_used / fog_cpu_total if fog_cpu_total > 0 else 0
        fog_memory_utilization = fog_memory_used / fog_memory_total if fog_memory_total > 0 else 0
        cloud_cpu_utilization = cloud_cpu_used / cloud_cpu_total if cloud_cpu_total > 0 else 0
        cloud_memory_utilization = cloud_memory_used / cloud_memory_total if cloud_memory_total > 0 else 0

        executed_pods_ratio = self._get_executed_pods() / self.pod_counter

        return {
            "edge_cpu_utilization": edge_cpu_utilization,
            "edge_memory_utilization": edge_memory_utilization,
            "fog_cpu_utilization": fog_cpu_utilization,
            "fog_memory_utilization": fog_memory_utilization,
            "cloud_cpu_utilization": cloud_cpu_utilization,
            "cloud_memory_utilization": cloud_memory_utilization,
            "overconsumption_penalty": False,
            "executed_pods_ratio": executed_pods_ratio
        }

    def _get_pod_paths(self):
        layer_paths = {
            'edge': {'hops': 0, 'count': 0},
            'fog': {'hops': 0, 'count': 0},
            'cloud': {'hops': 0, 'count': 0},
            'edge_to_fog': 0,
            'fog_to_cloud': 0,
            'fog_to_edge': 0,
            'cloud_to_fog': 0,
            'cloud_to_edge': 0,
            'edge_to_cloud': 0,
        }

        for agent in self.schedule.agents:
            if isinstance(agent, PodAgent):
                path = agent.path
                #length = len(path)
                #transport_cost = length - 1  # Each hop costs 1 unit, path length is total nodes visited

                # Count hops within each layer
                for i in range(len(path)-1):
                    layer1 = self.model.network.nodes[path[i]]['layer']
                    layer2 = self.model.network.nodes[path[i+1]]['layer']
                    if layer1 == layer2:
                        layer_paths[layer1]['hops'] += 1
                    elif layer1 == 'edge' and layer2 == 'fog':
                        layer_paths['edge_to_fog'] += 1
                    elif layer1 == 'fog' and layer2 == 'cloud':
                        layer_paths['fog_to_cloud'] += 1
                    elif layer1 == 'fog' and layer2 == 'edge':
                        layer_paths['fog_to_edge'] += 1
                    elif layer1 == 'cloud' and layer2 == 'fog':
                        layer_paths['cloud_to_fog'] += 1
                    elif layer1 == 'cloud' and layer2 == 'edge':
                        layer_paths['cloud_to_edge'] += 1
                    elif layer1 == 'edge' and layer2 == 'cloud':
                        layer_paths['edge_to_cloud'] += 1

                # Count paths per layer
                layer_paths[self.model.network.nodes[path[0]]['layer']]['count'] += 1

        return layer_paths
    
    def _step(self):
        model_step_interval = 1
        
        if self.randomized_interval:
            # inter_arrival_mu=0.5
            interval=max(1,round(np.random.exponential(scale=1./self.inter_arrival_mu)))
        else:
            interval = model_step_interval
        
        if self.model.time >= self.arrive_time_point :
            if self.pod_counter < self.num_pods:
                for n in range(int(self.pods_per_step)):
                    self._create_pod()  # Create a new pod each step
                    self.pod_counter += 1
            self.arrive_time_point += interval
        
        self.schedule.step()
        self.model.step(model_step_interval)
        
