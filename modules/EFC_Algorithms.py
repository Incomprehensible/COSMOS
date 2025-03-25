import random
from abc import ABC, abstractmethod
from time import sleep
from interfaces import request_pb2, responce_pb2
from google.protobuf.internal import encoder, decoder
from EFC_MLP import MLP
import yaml
import os
from os import path
import glob
import datetime

class PodBehaviourProfile:
    def __init__(self, behaviour_type, frevo_training, ipc):
        self.behaviour_type = behaviour_type  # The type of behavior (e.g., RandomWalk, MLPActivation)
        self.frevo_training = frevo_training
        self.ipc = ipc
        self.candidate = None

    def create(self, cpu_req, memory_req, comm_costs, nodes):
        if self.behaviour_type == 'random_walk':
            return RandomWalkProfile(cpu_req, memory_req, comm_costs, nodes)
        elif self.behaviour_type == 'aco':
            return AntColonyOptimizationProfile(cpu_req, memory_req, comm_costs, nodes)
        elif self.behaviour_type == 'hba':
            return HormoneAlgorithmProfile(cpu_req, memory_req, comm_costs, nodes)
        elif self.behaviour_type == 'mlp':
            return MLPActivationProfile(cpu_req, memory_req, comm_costs, nodes, self.frevo_training, self.ipc, self.candidate)
        else:
            raise ValueError(f"Unknown behaviour type: {self.behaviour_type}")
    
    def get_candidate_evaluated(self):
        return self.candidate.evaluations
    
    # def reset_candidate(self):
    #     if self.candidate != None:
    #         self.candidate.evaluations = 0
    
    def reinit_ipc(self, ipc):
        self.ipc = ipc
        if ipc == None:
            raise ValueError("PodBehaviourProfile:reinit: IPC interface is not initialized")
    
    def load_nn_from_config(self, nn_config):
        if self.candidate == None:
            self.candidate = MLP(nn_config)
        else:
            self.candidate.reinit(nn_config)
        # TODO: keeping the same MLP instance and reloading the weights and biases only
    
    def load_nn_from_sample(self, nn_sample_file):
        # reverse the save_candidate method
        mlp_folder = 'trained_mlp'
        if not path.exists(mlp_folder):
            raise ValueError('No trained MLP models found')
        config = yaml.full_load(open(path.join(mlp_folder, nn_sample_file)))
        print('loading candidate from sample file: ' + nn_sample_file)
        nn_config = MLP.parse_nn_conf_from_yaml(config)
        self.candidate = MLP(nn_config)
    
    def save_candidate(nn_config):
        folder_path = 'trained_mlp'
        now = datetime.datetime.now()
        file_name = 'mlp_config_' + now.strftime("%Y%m%d_%H%M") + '.yaml'
        # file_name = 'mlp_config' + str(now.time()) + '.yaml'
        config_dict = {
            'input_nodes': nn_config.input_nodes,
            'output_nodes': nn_config.output_nodes,
            'hidden_layers': nn_config.hidden_layers,
            'nodes_per_layer': nn_config.nodes_per_layer,
            'biases': list(nn_config.biases),
            'weights': list(nn_config.weights)
        }
        
        if not path.exists(folder_path):
            os.makedirs(folder_path)
        with open(path.join(folder_path, file_name), 'w') as file:
            yaml.dump(config_dict, file)
        
    def load_save_best_candidate_conf(self, folder_path):
        nn_config = None
        list_of_files = glob.glob(folder_path + '/*.zre')
        file_path = max(list_of_files, key=path.getctime)
        print('Loading best candidate from: ', file_path)
        with open(file_path, 'r') as file:
            nn_config = MLP.parse_nn_conf_from_file(file)
        PodBehaviourProfile.save_candidate(nn_config)
        return nn_config
        
class PodBehaviour(ABC):
    def __init__(self, cpu_req, memory_req, comm_costs, nodes):
        # static data that doesn't change
        self.cpu_req = cpu_req
        self.memory_req = memory_req
        self.comm_costs = comm_costs
        self.nodes = nodes
    
    @abstractmethod
    def next_node(self, current_agent, current_node, neighbors, visited, randomizer=None):
        return current_node
    
class RandomWalkProfile(PodBehaviour):
    def __init__(self, cpu_req, memory_req, comm_costs, nodes):
        super().__init__(cpu_req, memory_req, comm_costs, nodes)

    def next_node(self, current_agent, current_node, neighbors, visited, randomizer=None):
        if current_agent.available_cpu >= self.cpu_req and current_agent.available_memory >= self.memory_req:
            return current_node
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        if len(unvisited_neighbors)>0:
            return random.choice(unvisited_neighbors)
        return None

class HormoneAlgorithmProfile(PodBehaviour):
    def __init__(self, cpu_req, memory_req, comm_costs, nodes):
        super().__init__(cpu_req, memory_req, comm_costs, nodes)

    def next_node(self, current_agent, current_node, neighbors, visited, randomizer=None):
        if current_agent.available_cpu >= self.cpu_req and current_agent.available_memory >= self.memory_req:
            return current_node
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        if len(unvisited_neighbors) > 0:
            HOR = 0
            HOR_store = []
            HOR_prob = []
            sum_HOR = 0
            Hormone_node = None
            for neighbor in unvisited_neighbors:  
                neighbor_agent = self.nodes[neighbor]['agent']
                eta = 1 / self.comm_costs.get((current_agent.layer, neighbor_agent.layer)) #CC
                HOR = eta**3 * neighbor_agent.hormone**4
                HOR_store.append(HOR)
                sum_HOR += HOR
            for HOR in HOR_store:
                HOR_p = float(HOR) / sum_HOR
                HOR_prob.append(HOR_p)
            Hormone_node = randomizer.choices(unvisited_neighbors, weights = HOR_prob, k=1)[0]
            neighbor_select = Hormone_node
        else:
            neighbor_select = random.choice(neighbors)
        return neighbor_select

class AntColonyOptimizationProfile(PodBehaviour):
    def __init__(self, cpu_req, memory_req, comm_costs, nodes):
        super().__init__(cpu_req, memory_req, comm_costs, nodes)
        self.evaporation_rate = self.get_evaporation_rate(cpu_req)
        self.released = self.get_released(cpu_req)
    
    def get_evaporation_rate(self, cpu_req):
        if cpu_req <= 2:
            return 0.15
        elif cpu_req <= 8:
            return 0.1
        return 0.05
    
    def get_released(self, cpu_req):
        if cpu_req <= 2:
            return 1
        elif cpu_req <= 8:
            return 2
        return 3
    
    def next_node(self, current_agent, current_node, neighbors, visited, randomizer):
        if current_agent.available_cpu >= self.cpu_req and current_agent.available_memory >= self.memory_req:
            if current_agent.layer == 'edge':
                   current_agent.pheromone += 1*(self.released)
            if current_agent.layer == 'fog':
               current_agent.pheromone += 0.6*(self.released)
            return current_node
        current_agent.pheromone = current_agent.pheromone*(1-self.evaporation_rate)
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        if len(unvisited_neighbors) > 0:
            ACO = 0
            ACO_node = None
            ACO_store = []
            ACO_prob = []
            sum_ACO = 0
            for neighbor in unvisited_neighbors:
                n_a = self.nodes[neighbor]['agent']
                eta = 1/self.comm_costs.get((current_agent.layer, n_a.layer), 1)
                phe = n_a.pheromone # pheromone value
                ACO = (eta**1)*(phe**(1.4)) # designed weight      
                ACO_store.append(ACO)
                sum_ACO += ACO
            for ACO in ACO_store:
              ACO_p = float(ACO)/sum_ACO # prob to the neighbor  
              ACO_prob.append(ACO_p)
            ACO_node = randomizer.choices(unvisited_neighbors, weights=ACO_prob, k=1)[0]
            neighbor_select = ACO_node
        else:
            neighbor_select = random.choice(neighbors)
        return neighbor_select

class MLPActivationProfile(PodBehaviour):
    def __init__(self, cpu_req, memory_req, comm_costs, nodes, frevo_training, ipc, MLP=None):
        super().__init__(cpu_req, memory_req, comm_costs, nodes)
        self.frevo_training = frevo_training
        self.ipc = ipc
        self.candidate = MLP

    def next_node(self, current_agent, current_node, neighbors, visited, randomizer=None):
        return self.next_node_inference(current_agent, current_node, neighbors, visited)
    
    def next_node_inference(self, current_agent, current_node, neighbors, visited):
        if current_agent.available_cpu >= self.cpu_req and current_agent.available_memory >= self.memory_req:
            return current_node
        pod_cpu_req = self.cpu_req
        pod_mem_req = self.memory_req
        id = 0
        agents = []
        agent = request_pb2.AgentProfile()
        agent.comm_costs = 0
        agent.agent_available_cpu = current_agent.available_cpu
        agent.agent_available_memory = current_agent.available_memory
        agent.agent_utilization_cpu = int((current_agent.cpu-current_agent.available_cpu)/current_agent.cpu)
        agent.id = id
        agents.append(agent)

        unvisited_neighbors = [n for n in neighbors if n not in visited]
        if len(unvisited_neighbors) == 0:
            return None
        for n in unvisited_neighbors:
            id += 1
            agent = request_pb2.AgentProfile()
            agent.comm_costs = self.comm_costs.get((current_agent.layer, self.nodes[n]['agent'].layer), 1)
            agent.agent_available_cpu = self.nodes[n]['agent'].available_cpu
            agent.agent_available_memory = self.nodes[n]['agent'].available_memory
            agent.agent_utilization_cpu = int((self.nodes[n]['agent'].cpu-agent.agent_available_cpu)/self.nodes[n]['agent'].cpu)
            agent.id = id
            agents.append(agent)
        
        activation = {}

        for i in range(len(agents)):
            agent = agents[i]
            comm_costs = agent.comm_costs
            agent_available_cpu = agent.agent_available_cpu
            agent_available_memory = agent.agent_available_memory
            agent_utilization_cpu = agent.agent_utilization_cpu
            agent_id = agent.id
            input_data = [
                float(agent_available_cpu-pod_cpu_req),
                float(agent_available_memory-pod_mem_req),
                comm_costs,
                float(agent_utilization_cpu)
            ]
            output = self.candidate.forward(input_data)
            activation[agent_id] = output[0]
        # Sort the activation dictionary by values in descending order
        activation = dict(sorted(activation.items(), key=lambda item: item[1], reverse=True))
        
        # Get the key of the first entry
        agent_id = next(iter(activation))
        
        if self.frevo_training == True:
            self.candidate.evaluations += 1
        # NAD: changed to None so not to confuse staying at the same node with using the current node's resources
        if agent_id == 0:
            return None
        return unvisited_neighbors[agent_id-1]
    
    # def next_node_local_train(self, current_agent, current_node, neighbors, visited):
    #     pod_cpu_req = self.cpu_req
    #     pod_mem_req = self.memory_req
    #     id = 0
    #     agents = []
    #     agent = request_pb2.AgentProfile()
    #     agent.comm_costs = 0
    #     agent.agent_available_cpu = current_agent.available_cpu
    #     agent.agent_available_memory = current_agent.available_memory
    #     agent.agent_utilization_cpu = int(current_agent.utilization_cpu)
    #     agent.id = id
    #     agents.append(agent)

    #     unvisited_neighbors = [n for n in neighbors if n not in visited]
    #     if not unvisited_neighbors:
    #         return current_node
    #     for n in unvisited_neighbors:
    #         id += 1
    #         agent = request_pb2.AgentProfile()
    #         agent.comm_costs = self.comm_costs.get((current_agent.layer, self.nodes[n]['agent'].layer), 1)
    #         agent.agent_available_cpu = self.nodes[n]['agent'].available_cpu
    #         agent.agent_available_memory = self.nodes[n]['agent'].available_memory
    #         # TODO: CHANGE MESSAGE TYPE INT64 TO FLOAT
    #         agent.agent_utilization_cpu = int(self.nodes[n]['agent'].utilization_cpu)
    #         agent.id = id
    #         agents.append(agent)
        
    #     activation = {}

    #     # print('agents length: ' + str(len(agents)))
    #     for i in range(len(agents)):
    #         agent = agents[i]
    #         comm_costs = agent.comm_costs
    #         agent_available_cpu = agent.agent_available_cpu
    #         agent_available_memory = agent.agent_available_memory
    #         agent_utilization_cpu = agent.agent_utilization_cpu
    #         agent_id = agent.id
    #         input_data = [
    #             float(pod_cpu_req),
    #             float(pod_mem_req),
    #             comm_costs,
    #             float(agent_available_cpu),
    #             float(agent_available_memory),
    #             float(agent_utilization_cpu)
    #         ]
    #         # input_data = torch.tensor(input_data, dtype=torch.float32)
    #         output = self.candidate.forward(input_data)
    #         activation[agent_id] = output[0]
    #         # print(f'Agent ID: {agent_id}, Activation Value: {output[0]}')
        
    #     # Sort the activation dictionary by values in descending order
    #     activation = dict(sorted(activation.items(), key=lambda item: item[1], reverse=True))
    #     # print(activation)
        
    #     # Get the key of the first entry
    #     agent_id = next(iter(activation))
    #     # print('agent_id: ' + str(agent_id))
        
    #     if agent_id == 0:
    #         return current_node
    #     return unvisited_neighbors[agent_id-1]
    
    def next_node_remote_inference(self, current_agent, current_node, neighbors, visited):
        request = request_pb2.Request()
        request.last = False
        request.pod_cpu_req = self.cpu_req
        request.pod_mem_req = self.memory_req
        
        id = 0       
        agent = request_pb2.AgentProfile()
        agent.comm_costs = 0
        agent.agent_available_cpu = current_agent.available_cpu
        agent.agent_available_memory = current_agent.available_memory
        # TODO: CHANGE MESSAGE TYPE INT64 TO FLOAT
        agent.agent_utilization_cpu = int(current_agent.utilization_cpu)
        if current_agent.layer == 'edge':
            agent.type = request_pb2.AgentProfile.EDGE
        elif current_agent.layer == 'fog':
            agent.type = request_pb2.AgentProfile.FOG
        else:
            agent.type = request_pb2.AgentProfile.CLOUD
        agent.id = id
        request.agents.append(agent)
        
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        if not unvisited_neighbors:
            return current_node
        for n in unvisited_neighbors:
            id += 1
            agent.comm_costs = self.comm_costs.get((current_agent.layer, self.nodes[n]['agent'].layer), 1)
            agent.agent_available_cpu = self.nodes[n]['agent'].available_cpu
            agent.agent_available_memory = self.nodes[n]['agent'].available_memory
            # TODO: CHANGE MESSAGE TYPE INT64 TO FLOAT
            agent.agent_utilization_cpu = int(self.nodes[n]['agent'].utilization_cpu)
            if self.nodes[n]['agent'].layer == 'edge':
                agent.type = request_pb2.AgentProfile.EDGE
            elif self.nodes[n]['agent'].layer == 'fog':
                agent.type = request_pb2.AgentProfile.FOG
            else:
                agent.type = request_pb2.AgentProfile.CLOUD
            agent.id = id
            request.agents.append(agent)
        
        self.ipc.send_over_socket(request.SerializeToString())
        message = self.ipc.receive_over_socket()
        response = responce_pb2.Response()
        response.ParseFromString(message)
        # print('response id: ' + str(response.id))
        if response.id == 0:
            return current_node
        return unvisited_neighbors[response.id-1]
