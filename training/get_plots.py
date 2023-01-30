import pandas as pd
from matplotlib import pyplot as plt

tasks = ['matbench_phonons', 'matbench_perovskites', 'matbench_mp_gap', 'matbench_mp_e_form', 'matbench_expt_gap']

for task in tasks:
    df_graph = pd.read_csv('results/result {}.csv'.format(task))
    #df_graph = pd.read_csv('results/result matbench_perovskites.csv')
    df_x = df_graph['percent_of_dataset_as_training']
    
    df_y_mae = df_graph['mae']
    df_y_mae_base = df_graph['mae_base']
    df_y_mae_j = df_graph['mae_onehot']
    
    fig, ax1 = plt.subplots(nrows=1, ncols=1 ,figsize=(7,5))

    ax1.plot(df_x, df_y_mae, marker='o', markersize=10, mfc= 'white', linestyle='solid', label='our model')
    ax1.plot(df_x, df_y_mae_base, marker='o', markersize=10, mfc= 'white', linestyle='dashed', label='mat2vec original')
    ax1.plot(df_x, df_y_mae_j, marker='o', markersize=10, mfc= 'white', linestyle='dashed', label='onehot')

    ax1.legend()
    ax1.set_title('{}'.format(task))
    ax1.set_xlabel('Percentage of dataset')
    ax1.set_ylabel('mae value')

    plt.tight_layout()
    plt.show()
    fig.savefig('results/{}.png'.format(task))