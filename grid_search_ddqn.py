import os
import random

import gym
import numpy as np
import pandas as pd
import torch

from double_dqn import DoubleDQNAgent
from dqn import DQN
from playing import play_games_using_agent
from replay_buffer import ExperienceReplayBuffer
from utils import plot_rewards, plot_losses, plot_evaluation_rewards

if __name__ == '__main__':
    # Inicialización:
    environment = gym.make('LunarLander-v2', render_mode='rgb_array')
    # Utilizamos la cpu porque en este caso es más rápida:
    DEVICE = torch.device('cpu')
    agent_name = "gs_ddqn"

    gs_results_file = f"{agent_name}/experiments.csv"
    try:
        os.mkdir(agent_name)
    except FileExistsError:
        pass

    # Hyperparams:
    MEMORY_SIZE = 10000  # Máxima capacidad del buffer
    BURN_IN = 100  # Número de pasos iniciales usados para rellenar el buffer antes de entrenar
    MAX_EPISODES = 1000  # Número máximo de episodios (el agente debe aprender antes de llegar a este valor)
    INIT_EPSILON = 1  # Valor inicial de epsilon
    EPSILON_DECAY = .98  # Decaimiento de epsilon
    MIN_EPSILON = 0.01  # Valor mínimo de epsilon en entrenamiento
    GAMMA = 0.99  # Valor gamma de la ecuación de Bellman
    BATCH_SIZES = [16, 32, 64]  # Conjunto a coger del buffer para la red neuronal
    LRS = [0.001, 0.0005, 0.0001]  # Velocidad de aprendizaje
    DNN_UPDS = [1, 3]  # Frecuencia de actualización de la red neuronal
    DNN_SYNCS = [1000, 2000]  # Frecuencia de sincronización de pesos entre la red neuronal y la red objetivo

    # Grid search:
    gs_results = []
    for BATCH_SIZE in BATCH_SIZES:
        for LR in LRS:
            for DNN_UPD in DNN_UPDS:
                for DNN_SYNC in DNN_SYNCS:
                    # Fijamos las semillas utilizadas, por reproducibilidad:
                    RANDOM_SEED = 666
                    random.seed(RANDOM_SEED)
                    torch.manual_seed(RANDOM_SEED)
                    np.random.seed(RANDOM_SEED)
                    environment.np_random, _ = gym.utils.seeding.np_random(RANDOM_SEED)
                    environment.action_space.seed(RANDOM_SEED)
                    # Parameters:
                    parameters = (f"BATCH_SIZE={BATCH_SIZE}, LR={LR}, "
                                  f"DNN_UPD={DNN_UPD}, DNN_SYNC={DNN_SYNC}.")
                    print(f"Running experiment with {parameters}.")
                    # Agent initialization:
                    er_buffer = ExperienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)
                    dqn = DQN(env=environment, learning_rate=LR, device=DEVICE)
                    double_dqn_agent = DoubleDQNAgent(
                        env=environment,
                        dnnetwork=dqn,
                        buffer=er_buffer,
                        epsilon=INIT_EPSILON,
                        eps_decay=EPSILON_DECAY,
                        batch_size=BATCH_SIZE,
                        min_epsilon=MIN_EPSILON
                    )

                    # Agent training:
                    training_time = double_dqn_agent.train(
                        gamma=GAMMA,
                        max_episodes=MAX_EPISODES,
                        dnn_update_frequency=DNN_UPD,
                        dnn_sync_frequency=DNN_SYNC
                    )
                    print(f"Training time: {training_time} minutes.")
                    # Training evaluation:
                    plot_rewards(
                        training_rewards=double_dqn_agent.training_rewards,
                        mean_training_rewards=double_dqn_agent.mean_training_rewards,
                        reward_threshold=environment.spec.reward_threshold,
                        title=parameters,
                        save_file_name=f'{agent_name}/{parameters}_rewards.png'
                    )
                    plot_losses(
                        training_losses=double_dqn_agent.training_losses,
                        title=parameters,
                        save_file_name=f'{agent_name}/{parameters}_losses.png'
                    )

                    # Evaluation:
                    tr, _ = play_games_using_agent(environment, double_dqn_agent, 100)
                    plot_evaluation_rewards(
                        rewards=tr,
                        reward_threshold=environment.spec.reward_threshold,
                        title=parameters,
                        save_file_name=f'{agent_name}/{parameters}_evaluation.png'
                    )
                    # Store metrics:
                    run_results = {
                        'solved': double_dqn_agent.mean_training_rewards[-1] >= environment.spec.reward_threshold,
                        'train episodes': len(double_dqn_agent.mean_training_rewards),
                        "training time": training_time,
                        'mean evaluation rewards': round(tr.mean(), 2),
                        'median evaluation rewards': round(np.median(tr), 2),
                        'well landed eval. episodes': sum(tr >= 200),
                        'landed eval. episodes': sum((tr < 200) & (tr >= 100)),
                        'crashed eval. episodes': sum(tr < 100),
                        "LEARNING RATE": LR,
                        "DNN UPD FREQ": DNN_UPD,
                        "DNN SYNC FREQ": DNN_SYNC,
                        "BATCH SIZE": BATCH_SIZE,
                        "MAX EPISODES": MAX_EPISODES
                    }
                    # Update the list:
                    gs_results.append(run_results)
                    # Update the csv:
                    pd.DataFrame(gs_results).to_csv(gs_results_file)
