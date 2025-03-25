import os
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

from EFC_Agents import *

class EdgeFogCloud_DB:
    def __init__(self, graph_name, import_sample, name_sample, dataset_path=''):
        self.name = graph_name
        # self.model = model # needed for Agents reconstruction
        self.import_sample = import_sample
        self.name_sample = name_sample
        self.dataset_path = dataset_path
        self.cnt = 0
        self.graph_files = []  # To store list of graph files
        self._load_file_names()  # Load file names into memory
    
    def _load_file_names(self):
        """Private method to load the list of GraphML files in the dataset directory."""
        if os.path.exists(self.dataset_path):
            self.graph_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.graphml')]
        else:
            raise FileNotFoundError(f"Directory {self.dataset_path} does not exist")
        self.cnt = 0  # Reset the counter when loading file names
    
    # copies graph and modifies a copy
    def _preprocess_graph(self, graph):
        """Preprocess the graph to replace non-serializable attributes."""
        graph_copy = graph.copy()

        # Iterate through all nodes and flatten the 'agent' attributes
        for node, node_data in graph_copy.nodes(data=True):
            if 'agent' in node_data:
                agent = node_data['agent']
                # Flatten the agent's attributes into individual node attributes
                node_data['layer'] = agent.layer
                node_data['cpu'] = agent.cpu
                node_data['memory'] = agent.memory
                # Remove the agent object from the graph since it's not serializable
                del node_data['agent']

        return graph_copy
    
    def export_graph(self, name, graph):
        """Export the graph with serializable attributes."""
        graph_copy = self._preprocess_graph(graph)
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
        nx.write_graphml(graph_copy, os.path.join(self.dataset_path, f'{name}.graphml'))
    
    # modifies graph
    def _postprocess_graph(self, model, graph):
        # Re-instantiate DeviceAgent objects from the serialized data
        for node, node_data in graph.nodes(data=True):
            # Recreate the DeviceAgent object
            agent = DeviceAgent(
                unique_id=node,  # Use node ID as the unique ID for the agent
                model=model,     # Pass the simulation model reference here
                # model=self.model,     # Pass the simulation model reference here
                layer=node_data['layer'],
                cpu=int(node_data['cpu']),
                memory=int(node_data['memory'])
            )

            # Restore the agent to the node data
            node_data['agent'] = agent

    def import_graph(self, model, name, show=False):
        print('importing: ' + name)
        """Import a graph and re-instantiate DeviceAgent objects."""
        if f'{name}.graphml' not in self.graph_files:
            raise FileNotFoundError(f"Graph file {name}.graphml does not exist")
        
        file_path = os.path.join(self.dataset_path, f'{name}.graphml')
        G = nx.read_graphml(file_path)

        self._postprocess_graph(model=model, graph=G)

        if show:
            nx.draw(G, with_labels=True)
            plt.show()

        return G
    
    def import_next_graph(self, model):
        """Generator-like method to import the next graph from the dataset directory."""
        if not self.graph_files:
            self._load_file_names()  # Reload files if directory content is updated

        if self.cnt >= len(self.graph_files):
            self.cnt = 0  # Reset counter if we've gone through all files

        next_file = self.graph_files[self.cnt]
        self.cnt += 1
        file_path = os.path.join(self.dataset_path, next_file)
        G = nx.read_graphml(file_path)
        self._postprocess_graph(model=model, graph=G)
        return G
    
    def generate_network(self, model):
        initialized = True
        if self.name_sample != None:
            return initialized, self.import_graph(model=model, name=self.name_sample)
        elif self.import_sample:
            return initialized, self.import_next_graph(model)
        else:
            initialized = False
            return initialized, nx.Graph(name=self.name)

class EdgeFogCloud_Continuum(Model):
    def __init__(self, behaviour_type, num_edge_devices, num_fog_nodes, num_cloud_servers, import_sample, name_sample, dataset_path, datacollector_enabled=True):
        super().__init__()
        
        self.behaviour_type = behaviour_type
        
        self.datacollector_enabled = datacollector_enabled
        
        self.dataset_db = EdgeFogCloud_DB(graph_name='EFC', import_sample=import_sample, name_sample=name_sample, dataset_path=dataset_path)
        (self.initialized, self.network) = self.dataset_db.generate_network(model=self)
                
        if not self.initialized:
            self.num_edge_devices = num_edge_devices
            self.num_fog_nodes = num_fog_nodes
            self.num_cloud_servers = num_cloud_servers
        else:
            self.num_edge_devices = 0
            self.num_fog_nodes = 0
            self.num_cloud_servers = 0
        
        self.time = 0
        self.executed_pods = 0
                
        # Define communication costs
        self.comm_costs = {
            ('edge', 'edge'): 1,
            ('edge', 'fog'): 5,
            ('fog', 'edge'): 5,
            ('fog', 'fog'): 2,
            ('fog', 'cloud'): 10,
            ('cloud', 'fog'): 10,
            ('edge', 'cloud'): 15
        }
        
        # Data Collector
        if self.datacollector_enabled:
            self.datacollector = DataCollector(
            agent_reporters={"Position": "node",
                             "Available_CPU": lambda agent: (agent.cpu-agent.available_cpu)/agent.cpu if isinstance(agent, DeviceAgent) else None,
                             "Available_Memory": lambda agent: (agent.memory-agent.available_memory)/agent.memory if isinstance(agent, DeviceAgent) else None,
                             "Comm_Cost": lambda agent: agent.TT if isinstance(agent, PodAgent) else None
                             }
            )
    
    def _finalize_graph(self, schedule):
        for node in self.network.nodes:
            if self.network.nodes[node]['layer'] == 'edge':
                self.num_edge_devices += 1
            elif self.network.nodes[node]['layer'] == 'fog':
                self.num_fog_nodes += 1
            elif self.network.nodes[node]['layer'] == 'cloud':
                self.num_cloud_servers += 1
            agent = self.network.nodes[node]['agent']
            schedule.add(agent)
        
        self.edge_nodes = [f'edge_{i}' for i in range(self.num_edge_devices)]
        self.fog_nodes = [f'fog_{i}' for i in range(self.num_fog_nodes)]
        self.cloud_nodes = [f'cloud_{i}' for i in range(self.num_cloud_servers)]
    
    def enable_datacollector(self):
        self.datacollector_enabled = True
    
    def reinit(self):
        self.time = 0
        self.executed_pods = 0
        # recreate Data Collector
        if self.datacollector_enabled:
            self.datacollector = DataCollector(
            agent_reporters={"Position": "node",
                             "Available_CPU": lambda agent: (agent.cpu-agent.available_cpu)/agent.cpu if isinstance(agent, DeviceAgent) else None,
                             "Available_Memory": lambda agent: (agent.memory-agent.available_memory)/agent.memory if isinstance(agent, DeviceAgent) else None,
                             "Comm_Cost": lambda agent: agent.TT if isinstance(agent, PodAgent) else None
                             }
            )
    
    def init_model(self, schedule):
        if not self.initialized:
            self._create_edge_layer(schedule)
            self._create_fog_layer(schedule)
            self._create_cloud_layer(schedule)
            self._construct_network()
        else:
            self._finalize_graph(schedule)
        
    def _create_edge_layer(self, schedule):
        self.edge_nodes = [f'edge_{i}' for i in range(self.num_edge_devices)]
        
    # Assign CPU and Memory capacities based on layer
        for node in self.edge_nodes:
           cpu = random.sample([16, 32], k=1)[0]
           memory = random.sample([16, 32], k=1)[0]
           agent = DeviceAgent(node, self, 'edge', cpu, memory)
           self.network.add_node(node, layer='edge', cpu=cpu, memory=memory, agent=agent)
           schedule.add(agent)

    def _create_fog_layer(self, schedule):
        self.fog_nodes = [f'fog_{i}' for i in range(self.num_fog_nodes)]
        
        for node in self.fog_nodes:
           cpu = random.sample([64, 128], k=1)[0]
           memory = random.sample([64, 128], k=1)[0]
           agent = DeviceAgent(node, self, 'fog', cpu, memory)
           self.network.add_node(node, layer='fog', cpu=cpu, memory=memory, agent=agent)
           schedule.add(agent)
    
    def _create_cloud_layer(self, schedule):
        self.cloud_nodes = [f'cloud_{i}' for i in range(self.num_cloud_servers)]

        for node in self.cloud_nodes:
           cpu = random.randint(2560000, 5120000)
           memory = random.randint(2560000, 5120000)
           agent = DeviceAgent(node, self, 'cloud', cpu, memory)
           self.network.add_node(node, layer='cloud', cpu=cpu, memory=memory, agent=agent)
           schedule.add(agent)
    
    def _construct_network(self):
        if self.initialized: # not needed, TODO: remove
            return
    
        prob_ee=random.random()
        prob_ff=random.random()
        prob_ef=random.random()
        prob_fc=random.random()
        
        self._add_layer_connections(self.edge_nodes, prob_ee)
        self._add_layer_connections(self.fog_nodes, prob_ff)
        # self.add_layer_connections(self.cloud_nodes, 0)
        
        self._add_interlayer_connections(self.edge_nodes, self.fog_nodes, prob_ef)
        self._add_interlayer_connections(self.fog_nodes, self.cloud_nodes, prob_fc)
        self._ensure_fog_connected()
        self._ensure_connected()
        #if self.behaviour_type == 'hba':
        #    self.hormone_update

    def _add_layer_connections(self, nodes, prob):
        for i in nodes:
            for j in nodes:
                if i != j and random.random() < prob:
                    self.network.add_edge(i, j)
                    
    def _add_interlayer_connections(self, layer1, layer2, prob):
        for i in layer1:
            for j in layer2:
                if random.random() < prob:
                    self.network.add_edge(i, j)
                    
    def _ensure_fog_connected(self):
        while not nx.is_connected(self.network.subgraph(self.fog_nodes)):
            fog_edges=self.network.subgraph(self.fog_nodes).edges
            self.network.remove_edges_from(fog_edges)
            self._add_layer_connections(self.fog_nodes, 0.3)
                    
    def _ensure_connected(self):
        # Ensure that the network is fully connected
        while not nx.is_connected(self.network):
            edges=self.network.edges
            fog_edges=self.network.subgraph(self.fog_nodes).edges
            remove_edges=edges-fog_edges
            self.network.remove_edges_from(remove_edges)
            self._add_layer_connections(self.edge_nodes, 0.2)
            #self.add_layer_connections(self.fog_nodes, 0.3)
            self._add_layer_connections(self.cloud_nodes, 0)
            
            self._add_interlayer_connections(self.edge_nodes, self.fog_nodes, 0.1)
            self._add_interlayer_connections(self.fog_nodes, self.cloud_nodes, 0.3)

    # for Hormone Algorithm behaviour profile
    def _hormone_update(self):    
        for resource_agent in (self.edge_nodes + self.fog_nodes + self.cloud_nodes):
            sum_eta=0
            RA=self.network.nodes[resource_agent]['agent']
            neighbors = list(self.network.neighbors(resource_agent)) 
            delta_1_hormone=0.6*((RA.available_cpu/RA.cpu) + (RA.available_memory/RA.memory))
            delta_2m_hormone=0.1*delta_1_hormone
            RA.hormone = 0.9*RA.hormone + delta_1_hormone-delta_2m_hormone
            if neighbors:
              for neighbor in neighbors:           
                Nei=self.network.nodes[neighbor]['agent'] 
                eta=1/self.comm_costs.get((RA.layer, Nei.layer))#CC
                sum_eta+=eta
              for neighbor in neighbors:                
                Nei=self.network.nodes[neighbor]['agent']
                eta=1/self.comm_costs.get((RA.layer, Nei.layer))    
                Nei.hormone += eta*delta_2m_hormone/sum_eta
    
    # implement model-wide agents update based on behaviour profile
    def _model_update(self):
        if self.behaviour_type == 'hba':
            self._hormone_update()
        
    def step(self, interval):
        self.time += interval
        self._model_update() # update EFC model based on activated behaviour
        if self.datacollector_enabled:
            self.datacollector.collect(self)
    
    def generate_samples(self, gen_samples_num):
        for i in range(gen_samples_num):
            edges=self.network.edges
            self.network.remove_edges_from(edges)
            self._construct_network()
            self.dataset_db.export_graph(self.network.name+str(i), self.network)

    