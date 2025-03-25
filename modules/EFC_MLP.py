import torch
import torch.nn as nn
from interfaces import candidate_pb2
import xml.etree.ElementTree as ET
import yaml

class MLP(nn.Module):
    def __init__(self, nn_config):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        input_size = nn_config.input_nodes
        weight_index = 0
        bias_index = 0
        self.evaluations = 0
        
        for i in range(nn_config.hidden_layers):
            layer = nn.Linear(input_size, nn_config.nodes_per_layer)
            # Assign biases
            layer.bias.data = torch.tensor(nn_config.biases[bias_index:bias_index + nn_config.nodes_per_layer])
            bias_index += nn_config.nodes_per_layer
            # Assign weights
            for j in range(nn_config.nodes_per_layer):
                layer.weight.data[j] = torch.tensor(nn_config.weights[weight_index:weight_index + input_size])
                weight_index += input_size
            self.layers.append(layer)
            input_size = nn_config.nodes_per_layer
        
        self.output_layer = nn.Linear(input_size, nn_config.output_nodes)
        # Assign biases for output layer
        self.output_layer.bias.data = torch.tensor(nn_config.biases[bias_index:bias_index + nn_config.output_nodes])
        bias_index += nn_config.output_nodes
        # Assign weights for output layer
        for j in range(nn_config.output_nodes):
            self.output_layer.weight.data[j] = torch.tensor(nn_config.weights[weight_index:weight_index + input_size])
            weight_index += input_size
    
    def parse_nn_conf_from_file(file):
        tree = ET.parse(file)
        root = tree.getroot()
        
        best_candidate = root.find('.//ThreeLayerNetwork')
        print('Best candidate fitness: ', best_candidate.attrib['fitness'])
        nn_config = candidate_pb2.Candidate()
        nn_config.input_nodes = int(best_candidate.attrib['input_nodes'])
        nn_config.output_nodes = int(best_candidate.attrib['output_nodes'])
        nn_config.hidden_layers = int(best_candidate.attrib['hidden_layers'])
        nn_config.nodes_per_layer = int(best_candidate.attrib['nodes_per_layer'])
        
        for layer in best_candidate.findall('.//layer'):
            for node in layer.findall('.//node'):
                nn_config.biases.append(float(node.find('bias').text))
                for weight in node.find('weights').findall('weight'):
                    nn_config.weights.append(float(weight.text))
        
        for node in best_candidate.find('.//output_nodes').findall('node'):
            nn_config.biases.append(float(node.find('bias').text))
            for weight in node.find('weights').findall('weight'):
                nn_config.weights.append(float(weight.text))
        
        return nn_config

    def parse_nn_conf_from_yaml(config):
        nn_config = candidate_pb2.Candidate()
        nn_config.input_nodes = int(config['input_nodes'])
        nn_config.output_nodes = int(config['output_nodes'])
        nn_config.hidden_layers = int(config['hidden_layers'])
        nn_config.nodes_per_layer = int(config['nodes_per_layer'])
        
        for bias in config['biases']:
            nn_config.biases.append(float(bias))
        for weight in config['weights']:
            nn_config.weights.append(float(weight))
        
        return nn_config
    
    # def reinit(self, nn_config):
    #     self.evaluations = 0
    #     input_size = nn_config.input_nodes
    #     weight_index = 0
    #     bias_index = 0
    #     for i in range(nn_config.hidden_layers):
    #         # Assign biases 
    #         self.layers[i].bias.data = torch.tensor(nn_config.biases[bias_index:bias_index + nn_config.nodes_per_layer])
    #         bias_index += nn_config.nodes_per_layer
    #         # Assign weights
    #         for j in range(nn_config.nodes_per_layer):
    #             self.layers[i].weight.data[j] = torch.tensor(nn_config.weights[weight_index:weight_index + input_size])
    #             weight_index += input_size
    #         input_size = nn_config.nodes_per_layer
            
    #     # Assign biases for output layer
    #     self.output_layer.bias.data = torch.tensor(nn_config.biases[bias_index:bias_index + nn_config.output_nodes])
    #     bias_index += nn_config.output_nodes
    #     # Assign weights for output layer
    #     for j in range(nn_config.output_nodes):
    #         self.output_layer.weight.data[j] = torch.tensor(nn_config.weights[weight_index:weight_index + input_size])
    #         weight_index += input_size
    
    def reinit(self, nn_config):
        self.evaluations = 0
        input_size = nn_config.input_nodes
        weight_index = 0
        bias_index = 0
        for i in range(nn_config.hidden_layers):
            # Assign biases
            for bi in range (nn_config.nodes_per_layer):
                self.layers[i].bias.data[bi] = nn_config.biases[bias_index + bi]
            # ^ loop replaced:
            # self.layers[i].bias.data = torch.tensor(nn_config.biases[bias_index:bias_index + nn_config.nodes_per_layer])
            bias_index += nn_config.nodes_per_layer
            # Assign weights
            for j in range(nn_config.nodes_per_layer):
                for wi in range(input_size):
                    self.layers[i].weight.data[j][wi] = nn_config.weights[weight_index + wi]
                # ^ loop replaced:
                # self.layers[i].weight.data[j] = torch.tensor(nn_config.weights[weight_index:weight_index + input_size])
                weight_index += input_size
            input_size = nn_config.nodes_per_layer
        
        # Assign biases for output layer
        for bi in range (nn_config.output_nodes):
            self.output_layer.bias.data[bi] = nn_config.biases[bias_index + bi]
        # same self.output_layer.bias.data = torch.tensor(nn_config.biases[bias_index:bias_index + nn_config.output_nodes])
        bias_index += nn_config.output_nodes
        # Assign weights for output layer
        for j in range(nn_config.output_nodes):
            for wi in range(input_size):
                self.output_layer.weight.data[j][wi] = nn_config.weights[weight_index + wi]
            # same self.output_layer.weight.data[j] = torch.tensor(nn_config.weights[weight_index:weight_index + input_size])
            weight_index += input_size
    
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        for layer in self.layers:
            x = torch.sigmoid(layer(x))
        x = self.output_layer(x)
        return x