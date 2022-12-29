import os

import gym
import numpy as np
import torch

from double_dqn import DoubleDQNAgent
from dueling_dqn import DuelingDQN
from playing import play_games_using_agent
from replay_buffer import ExperienceReplayBuffer
from utils import plot_rewards, plot_losses, plot_evaluation_rewards, save_agent_gif

if __name__ == '__main__':
    # Inicialización:
    environment = gym.make('LunarLander-v2', render_mode='rgb_array')
    DEVICE = torch.device('mps')
    agent_name = "double_dueling_dqn"
    try:
        os.mkdir(agent_name)
    except FileExistsError:
        pass

    # Fijamos las semillas utilizadas, por reproducibilidad:
    # Referencias:
    # + https://pytorch.org/docs/stable/notes/randomness.html,
    # + https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
    RANDOM_SEED = 66
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    environment.action_space.seed(RANDOM_SEED)

    # Hyperparams:
    lr = 0.001  # Velocidad de aprendizaje
    MEMORY_SIZE = 10000  # Máxima capacidad del buffer
    MAX_EPISODES = 1000  # Número máximo de episodios (el agente debe aprender antes de llegar a este valor)
    EPSILON = 1  # Valor inicial de epsilon
    EPSILON_DECAY = .99  # Decaimiento de epsilon
    GAMMA = 0.99  # Valor gamma de la ecuación de Bellman
    BATCH_SIZE = 32  # Conjunto a coger del buffer para la red neuronal
    BURN_IN = 100  # Número de episodios iniciales usados para rellenar el buffer antes de entrenar
    DNN_UPD = 1  # Frecuencia de actualización de la red neuronal
    DNN_SYNC = 1000  # Frecuencia de sincronización de pesos entre la red neuronal y la red objetivo

    # Agent initialization:
    er_buffer = ExperienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)
    double_dqn = DuelingDQN(env=environment, learning_rate=lr, device=DEVICE)
    double_dqn_agent = DoubleDQNAgent(
        env=environment,
        dnnetwork=double_dqn,
        buffer=er_buffer,
        epsilon=EPSILON,
        eps_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE
    )

    # Agent training:
    training_time = double_dqn_agent.train(
        gamma=GAMMA,
        max_episodes=MAX_EPISODES,
        dnn_update_frequency=DNN_UPD,
        dnn_sync_frequency=DNN_SYNC,
        env_seed=RANDOM_SEED
    )
    print(f"Training time: {training_time} minutes.")
    # dqn_agent.dnnetwork.load_state_dict(torch.load(f'dqn_Trained_Model.pth'))
    # Training evaluation:
    plot_rewards(
        training_rewards=double_dqn_agent.training_rewards,
        mean_training_rewards=double_dqn_agent.mean_training_rewards,
        reward_threshold=environment.spec.reward_threshold,
        save_file_name=f'{agent_name}/{agent_name}_rewards.png'
    )
    plot_losses(training_losses=double_dqn_agent.training_losses,
                save_file_name=f'{agent_name}/{agent_name}_losses.png')

    # Saving:
    torch.save(obj=double_dqn_agent.dnnetwork.state_dict(),
               f=f'{agent_name}/{agent_name}_Trained_Model.pth')

    # Evaluation:
    tr, _ = play_games_using_agent(environment, double_dqn_agent, 100)
    plot_evaluation_rewards(
        rewards=tr,
        reward_threshold=environment.spec.reward_threshold,
        save_file_name=f'{agent_name}/{agent_name}_evaluation.png'
    )
    save_agent_gif(env=environment, ag=double_dqn_agent, save_file_name=f'{agent_name}/agente_{agent_name}.gif')
