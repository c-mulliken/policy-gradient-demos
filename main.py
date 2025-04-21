import matplotlib.pyplot as plt
import numpy as np
from train import main_train
from utils import ema
import time

EXPERIMENT_NAME = 'computational_cost'  # Change this to the desired experiment name

if EXPERIMENT_NAME == 'trpo_vs_reinforce_duration':
    trpo_model, trpo_results = main_train(method='trpo', max_kl=0.005, plot_result=False)
    reinforce_model, reinforce_results = main_train(method='reinforce', plot_result=False)

    plt.plot(trpo_results['iteration_durations'], label='TRPO')
    plt.plot(reinforce_results['iteration_durations'], label='REINFORCE')
    plt.xlabel('Iteration')
    plt.ylabel('Average Duration')
    plt.title('Average Duration vs Iteration')
    plt.legend()

    plt.savefig(f'visualisations/plots/{EXPERIMENT_NAME}.png')
elif EXPERIMENT_NAME == 'max_kl_test':
    max_kl_values = [0.1, 0.01, 0.001, 0.0001]
    n_trials = 2

    plt.figure(figsize=(12, 7))
    for max_kl in max_kl_values:
        model, results = main_train(method='trpo', max_kl=max_kl, plot_result=False)
        
        window = 10  # You can adjust the window size as needed
        durations = np.array(results['iteration_durations'])
        ema_durations = ema(durations)
        rolling_std = np.array([
            durations[max(0, i - window + 1):i + 1].std()
            for i in range(len(durations))
        ])
        plt.plot(ema_durations, label=f'max_kl={max_kl}')
        plt.fill_between(
            np.arange(len(ema_durations)),
            ema_durations - rolling_std,
            ema_durations + rolling_std,
            alpha=0.2
        )

    plt.xlabel('Iteration')
    plt.ylabel('Average Duration')
    plt.title('Max KL Test (Mean EMA ± Std)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'visualisations/plots/{EXPERIMENT_NAME}_ema_var.png')
    plt.show()
elif EXPERIMENT_NAME == 'model_size_reinforce':
    model_sizes = [8, 64, 128, 256]
    n_trials = 5

    plt.figure(figsize=(12, 7))
    avg_durations = []
    std_durations = []

    for size in model_sizes:
        size_durations = []
        for i in range(n_trials):
            model, results = main_train(method='reinforce', hidden_neurons=size, plot_result=False)
            final_twenty = results['iteration_durations'][-20:]
            size_durations.append(np.mean(final_twenty))
        avg_durations.append(np.mean(size_durations))
        std_durations.append(np.std(size_durations))
    
    for idx, size in enumerate(model_sizes):
        plt.bar(idx, avg_durations[idx], yerr=std_durations[idx], label=f'{size}')
    plt.xticks(range(len(model_sizes)), model_sizes)
    plt.xlabel('Model Size (Neurons)')
    plt.ylabel('Average Duration')
    plt.title('Model Size vs Average Duration (REINFORCE)')
    plt.tight_layout()
    plt.savefig(f'visualisations/plots/{EXPERIMENT_NAME}.png')
elif EXPERIMENT_NAME == "taxi_trpo_reinforce":
    trpo_model, trpo_results = main_train(method='trpo', max_kl=0.005, plot_result=False)
    reinforce_model, reinforce_results = main_train(method='reinforce', plot_result=False)

    plt.plot(trpo_results['iteration_durations'], label='TRPO')
    plt.plot(reinforce_results['iteration_durations'], label='REINFORCE')
    plt.xlabel('Iteration')
    plt.ylabel('Average Duration')
    plt.title('Average Duration vs Iteration')
    plt.legend()

    plt.savefig(f'visualisations/plots/{EXPERIMENT_NAME}.png')
elif EXPERIMENT_NAME == "computational_cost":
    n_trials = 3
    times_trpo_lowkl = []
    times_trpo_highkl = []
    times_reinforce = []
    for i in range(n_trials):
        t1 = time.time()
        trpo_model, trpo_results = main_train(method='trpo', max_kl=0.01, plot_result=False)
        t2 = time.time()
        times_trpo_highkl.append(t2 - t1)
        t1 = time.time()
        trpo_model, trpo_results = main_train(method='trpo', max_kl=0.001, plot_result=False)
        t2 = time.time()
        times_trpo_lowkl.append(t2 - t1)
        t1 = time.time()
        reinforce_model, reinforce_results = main_train(method='reinforce', plot_result=False)
        t2 = time.time()
        times_reinforce.append(t2 - t1)
    print("Average time for TRPO KL = 0.01: ", np.mean(times_trpo_highkl))
    print("Average time for TRPO KL = 0.001: ", np.mean(times_trpo_lowkl))
    print("Average time for REINFORCE: ", np.mean(times_reinforce))
    print("Standard deviation for TRPO KL = 0.01: ", np.std(times_trpo_highkl))
    print("Standard deviation for TRPO KL = 0.001: ", np.std(times_trpo_lowkl))
    print("Standard deviation for REINFORCE: ", np.std(times_reinforce))

