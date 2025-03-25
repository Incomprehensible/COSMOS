from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

from EFC_Agents import PodAgent

def plot_network(model):
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(model.network)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    
    layers = nx.get_node_attributes(model.network, 'layer')
    #cpus = nx.get_node_attributes(model.network, 'cpu')
    #memories = nx.get_node_attributes(model.network, 'memory')
    colors = {'edge': 'blue', 'fog': 'green', 'cloud': 'red'}
    
    # Assign numerical labels to nodes starting from 1
    node_labels = {node: idx for idx, node in enumerate(model.network.nodes)}
    labels = {node: f'{node_labels[node]}' for node in model.network.nodes}
    
    for layer, color in colors.items():
        nodes = [node for node in model.network if layers[node] == layer]
        nx.draw_networkx_nodes(model.network, pos, nodelist=nodes, node_color=color, label=layer.capitalize(), node_size=500, ax=ax)
    
    nx.draw_networkx_edges(model.network, pos, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(model.network, pos, labels=labels, font_size=20, ax=ax)
    
    # Visualize pods
    plt.legend(scatterpoints=1)
    plt.show()

def plot_CPU_utilization_over_time(model, layer):
    # Get data from the data collector
    data = model.datacollector.get_agent_vars_dataframe()
    
    # Filter data for DeviceAgent only
    data = data.dropna(subset=['Available_CPU', 'Available_Memory'])
    
    # Extract unique nodes and their layers for filtering
    nodes = list(model.network.nodes(data=True))
    nodes = [node[0] for node in nodes if node[1]['layer'] == layer]

    # Plot utilization over time for each node in one figure
    fig, ax = plt.subplots(figsize=(12, 8))

    for node in nodes :
        node_data = data.xs(node, level='AgentID')
        ax.plot(node_data['Available_CPU'].values, label=f'{node} CPU')
    
    ax.set_title('Utilization over Time')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel(f'Utilization of CPU on {layer.capitalize()} layer')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    
def plot_Mem_utilization_over_time(model, layer):
    # Get data from the data collector
    data = model.datacollector.get_agent_vars_dataframe()
    
    # Filter data for DeviceAgent only
    data = data.dropna(subset=['Available_CPU', 'Available_Memory'])
    
    # Extract unique nodes and their layers for filtering
    nodes = list(model.network.nodes(data=True))
    nodes = [node[0] for node in nodes if node[1]['layer'] == layer]

    # Plot utilization over time for each node in one figure
    fig, ax = plt.subplots(figsize=(12, 8))

    for node in nodes :
        node_data = data.xs(node, level='AgentID')
        tmp = node_data['Available_Memory']
        time_span = range(len(tmp))
        run_avg = np.cumsum(tmp)/time_span
        ax.plot(run_avg, label=f'{node} Memory')
    
    ax.set_title('Utilization over Time')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel(f'Utilization of Memory on {layer.capitalize()} layer')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
def plot_Comm_Cost_over_time(model, sim):
    # Get data from the data collector
    data = model.datacollector.get_agent_vars_dataframe()
    fig, ax = plt.subplots(figsize=(12, 8))
    for agent in sim.schedule.agents:
        if isinstance(agent, PodAgent):
            agent_data = data.xs(agent.unique_id, level="AgentID")
            tmp = agent_data['Comm_Cost']
            time_span = range(len(tmp))
            run_avg = np.cumsum(tmp)/time_span
            ax.plot(run_avg, label=f"Pod_{agent.unique_id}")
            # ax.plot(agent_data.index, run_avg, label=f"Pod_{agent.unique_id}")
    
    ax.set_title('Communication costs over Time')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel(f'Communication costs')
    # ax.legend()
    plt.tight_layout()
    plt.show()


# def plot_network_vertically(model, edge_spacing=4, fog_spacing=6, cloud_spacing=6, y_spacing=1):
#     max_spacing = max(edge_spacing, fog_spacing, cloud_spacing)
#     x_spacing = max_spacing
#     fig, ax = plt.subplots(figsize=(12, 8))
    
#     layers = nx.get_node_attributes(model.network, 'layer')
#     colors = {'edge': '#2f76b5', 'fog': '#75a150', 'cloud': '#fb0505'}
    
#     # Define fixed y-coordinates for each layer
#     layer_heights = {'edge': 0, 'fog': y_spacing, 'cloud': 2 * y_spacing}
    
#     pos = {}
#     max_nodes = max(len([n for n in model.network if layers[n] == l]) for l in colors.keys())
#     print("max_nodes: ", max_nodes)
#     # Assign evenly spaced x-coordinates for each layer
#     for layer in colors.keys():
#         if layer == 'edge':
#             x_spacing = edge_spacing
#         elif layer == 'fog':
#             x_spacing = fog_spacing
#         else:
#             x_spacing = cloud_spacing
#         nodes = [node for node in model.network if layers[node] == layer]
#         x_positions = [i * x_spacing for i in range(len(nodes))]
        
#         # Center nodes by shifting them
#         if layer == 'edge':
#             shift_spacing = 1
#         elif layer == 'fog':
#             shift_spacing = 2.6
#         else:
#             shift_spacing = 3.5
#         shift = (max_nodes - len(nodes))*shift_spacing / 2
#         print(x_positions)
#         for i, node in enumerate(nodes):
#             pos[node] = (x_positions[i] + shift, layer_heights[layer])
    
#     # Assign numerical labels to nodes
#     node_labels = {node: idx for idx, node in enumerate(model.network.nodes)}
#     labels = {node: f'{node_labels[node]}' for node in model.network.nodes}

#     for layer, color in colors.items():
#         nodes = [node for node in model.network if layers[node] == layer]
#         node_size = 560
#         if layer != 'edge':
#             node_size = 660
#         nx.draw_networkx_nodes(model.network, pos, nodelist=nodes, node_color=color, label=layer.capitalize(), node_size=node_size, ax=ax)
    
#     nx.draw_networkx_edges(model.network, pos, alpha=0.3, ax=ax)
#     nx.draw_networkx_labels(model.network, pos, labels=labels, font_size=14, font_weight="heavy", ax=ax)
    
#     # Manually set axis limits for proper spacing
#     ax.set_ylim(-y_spacing, 3 * y_spacing)
#     ax.set_xlim(-edge_spacing, max_nodes * edge_spacing + edge_spacing)

#     plt.legend(scatterpoints=1)
#     plt.show()