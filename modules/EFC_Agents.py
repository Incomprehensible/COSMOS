from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

class DeviceAgent(Agent):
    def __init__(self, unique_id, model, layer, cpu, memory, pheromone=None):
        super().__init__(model)
        self.unique_id = unique_id
        self.layer = layer  # Layer: 'edge', 'fog', or 'cloud'
        self.cpu = cpu      # CPU capacity
        self.memory = memory    # Memory capacity
        self.available_cpu = cpu
        self.available_memory = memory
        self.hormone = self.get_hormone(cpu, memory)
        self.pheromone = self.get_pheromone(cpu, memory) # pheromone value
        self.pos = None
    
    # for ACO behaviour profile
    def get_pheromone(self, cpu, memory):
        if cpu==16 and memory==16:
            return 4
        elif cpu==32 and memory==32:
            return 6
        return 5
    
    # for Hormone Algorithm behaviour profile
    def get_hormone(self, cpu, memory):
        if cpu==16 and memory==16:
            return 4
        elif cpu==32 and memory==32:
            return 6
        return 5

class PodAgent(Agent):
    def __init__(self, unique_id, model, cpu_req, memory_req, node, exec_time, behaviour):
        super().__init__(model)
        self.unique_id = unique_id
        self.cpu_req = cpu_req  # CPU requirement
        self.memory_req = memory_req  # Memory requirement
        self.node = node  # Node where the pod is located
        self.exec_Time = exec_time  # Execution Time
        self.TT = 0  # communication cost
        self.executed = False
        self.visited_nodes = {node}
        self.path = [node]  # Path of nodes visited
        self.arrval_Time = model.time
        self.departure_Time = self.arrval_Time + self.exec_Time
        self.behaviour = behaviour

    def step(self):
        if self.executed == False:
            current_node = self.node
            current_agent = self.model.network.nodes[current_node]['agent']
            
            neighbors = list(self.model.network.neighbors(current_node))
            neighbor_select = self.behaviour.next_node(current_agent, current_node, neighbors, self.visited_nodes, self.random)
            if neighbor_select != None:
                if neighbor_select == self.node:
                # Stay in the current node if it has enough resources
                    current_agent.available_cpu -= self.cpu_req
                    current_agent.available_memory -= self.memory_req
                    self.executed = True
                    self.model.executed_pods += 1
                    self.arrval_Time = self.model.time + self.TT
                    self.departure_Time = self.arrval_Time + self.exec_Time
                else:
                # Attempt to move to a neighboring node with available resources
                    neighbor_agent = self.model.network.nodes[neighbor_select]['agent']
                    self.node = neighbor_select
                    self.visited_nodes.add(self.node)
                    self.path.append(self.node) 
                    comm_cost = self.model.comm_costs.get((current_agent.layer, neighbor_agent.layer), 1)
                    self.TT += comm_cost
            
        if self.model.time >= self.departure_Time and self.executed == True:
            current_agent = self.model.network.nodes[self.node]['agent']
            current_agent.available_cpu += self.cpu_req
            current_agent.available_memory += self.memory_req
            # self.model.schedule.remove(self) # added this to remove the agent from the schedule after execution
            # but we can also remove the pod agent from the model to achieve the same effect
    
'''
get_small_pod, get_medium_pod, get_large_pod
profiles: define [(cpu_demand, mem_demand), (cpu_slack, mem_slack)]
demand_steps: list of required steps to execute the pod, uniform distribution
'''
def get_small_pod(demand_steps=(20,30)): 
    
    # considered small pod profiles
    small_profiles = [[(1,1),(0,0)], [(1,2),(0,0)], [(2,1),(0,0)], [(2,2),(0,0)]]
    
    prob1 = 1./len(small_profiles)*np.ones(len(small_profiles))
    demand, slack = random.choices(small_profiles, weights=prob1, k=1)[0]
    
    demand_step = random.randint(demand_steps[0], demand_steps[1])
    return demand, demand_step, slack

def get_medium_pod(demand_steps=(50,130)):  

    # considered medium pod profiles
    medium_profiles = [[(4,4),(1,1)], [(4,4),(1,2)], [(4,4),(2,1)],\
                      [(4,6),(1,2)], [(4,6),(2,1)], [(4,6),(2,2)],\
                      [(6,4),(1,2)], [(6,4),(2,1)], [(6,4),(2,2)],\
                      [(6,6),(1,2)], [(6,6),(2,1)], [(6,6),(2,2)]]
    
    prob1 = 1./len(medium_profiles)*np.ones(len(medium_profiles))
    
    demand, slack = random.choices(medium_profiles, weights=prob1, k=1)[0]
    
    demand_step = random.randint(demand_steps[0], demand_steps[1])
    return demand, demand_step, slack

def get_large_pod(demand_steps=(200,400)): 
    # considered large pod profiles
    large_profiles = [[(8,8),(4,4)],\
                      [(8,16),(4,4)], [(8,16),(4,6)],\
                      [(16,8),(4,4)], [(16,8),(6,4)],\
                      [(16,16),(4,6)], [(16,16),(6,4)], [(16,16),(6,6)]]
        
    prob1 = 1./len(large_profiles)*np.ones(len(large_profiles))
    demand, slack = random.choices(large_profiles, weights=prob1, k=1)[0]
    
    demand_step = random.randint(demand_steps[0], demand_steps[1])

    return demand, demand_step, slack


def get_pod_profile(categories_prob = (0.4, 0.4, 0.2), prob_elastisity=0.5):
    
    '''
    get the pod profile:
        pod_categories = ('s', 'm', 'l'), small, medium, large
        categories_prob = (0.4, 0.4, 0.2): corresponds to small , medium, large pods
        prob_elasticity: the prob. that pod is elastic
    '''

    rnd = random.random()
    if rnd <categories_prob[0]:
        # small demand
        pod_demand, pod_demand_step, pod_slack = get_small_pod(demand_steps=(60,120)) #(60,120)
        pod_is_elastic = True if random.random() < prob_elastisity else False
        
    elif categories_prob[0] <= rnd < categories_prob[0]+categories_prob[1]:
        # medium pod
        pod_demand, pod_demand_step, pod_slack = get_medium_pod(demand_steps=(100,200)) #(100,200)
        pod_is_elastic = True if random.random() < prob_elastisity else False
        
    else:
        #large pod
        pod_demand, pod_demand_step, pod_slack = get_large_pod(demand_steps=(150,300)) #(150,300)
        pod_is_elastic = False #True if random.random() < prob_elastisity else False
    
    if not pod_is_elastic:
        pod_demand_tolerance = int(0.1*pod_demand_step)
    else:
        pod_demand_tolerance = int(0.3*pod_demand_step) 
        pod_slack = [0,0]
        
    return pod_demand, pod_demand_step, pod_slack, pod_is_elastic, pod_demand_tolerance