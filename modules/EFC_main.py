import yaml
from EFC_Simulation import *

# 3 modes of operation:
# normal mode (either generate a model or use model from dataset)
# training (using frevo ipc interface)
# samples generation (no runtime required)

if __name__ == '__main__':
    sim_config = yaml.safe_load(open("config/config.yml"))

    # model = EdgeFogCloud_Continuum(num_edge_devices, num_fog_nodes, num_cloud_servers, import_sample, model_name, generate_samples, path_to_dataset)
    # sim = EdgeFogCloud_Simulation(model=model, sim_steps=sim_steps, num_pods=num_pods, behaviour_type=behaviour_type, frevo_training=frevo_training, ipc=ipc)
    sim = EdgeFogCloud_Simulation(sim_config)

    sim.start()