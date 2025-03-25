import pandas as pd
import matplotlib.pyplot as plt
import sys

filename = 'eval.csv'
if len(sys.argv) > 1:
    filename = sys.argv[1]
df = pd.read_csv(filename, index_col='Step', header = 0)

print(df)

pd.set_option("display.max.columns", None)
df.head()

random_df = df[(df['Algorithm'] == 'random_walk')]
aco_df = df[(df['Algorithm'] == 'aco')]
mlp_df = df[(df['Algorithm'] == 'mlp')]

# TODO: get these values from csv
mus = [0.5, 0.8]
num_pods = [100, 250, 500, 1000, 2500]
metrics = ['Fog Dependency', 'Cloud Dependency', 'Edge CPU Utilization', 'Edge Memory Utilization', 'Executed Pods Ratio']
random_df_plots = []
aco_df_plots = []
mlp_df_plots = []

for m in metrics:
    for mu in mus:
        for p in num_pods:
            random_df_plots.append(random_df.groupby('mu').get_group(mu))  
            aco_df_plots.append(aco_df.groupby('mu').get_group(mu))
            mlp_df_plots.append(mlp_df.groupby('mu').get_group(mu))

            random_df_plots[-1] = random_df_plots[-1].groupby('Num Pods').get_group(p)
            aco_df_plots[-1] = aco_df_plots[-1].groupby('Num Pods').get_group(p)
            mlp_df_plots[-1] = mlp_df_plots[-1].groupby('Num Pods').get_group(p)

            plt.figure(figsize=(16, 8), dpi=150) 

            random_df_plots[-1][m].plot(label='random', color='orange')
            aco_df_plots[-1][m].plot(label='aco', color='blue')
            mlp_df_plots[-1][m].plot(label='mlp', color='green')

            plt.title(m + ' for mu = ' + str(mu) + ' and ' + str(p) + ' pods')
            plt.xlabel('Step')
            plt.legend()
            plt.show()
