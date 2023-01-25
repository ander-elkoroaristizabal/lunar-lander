import os
import random

import gym
import numpy as np
import torch

from double_dqn import DoubleDQNAgent
from playing import play_games_using_agent
from replay_buffer import ExperienceReplayBuffer
from utils import plot_rewards, plot_losses, plot_evaluation_rewards, save_agent_gif


class DuelingDQN(torch.nn.Module):

    def __init__(self, env: gym.Env, learning_rate: float, device=torch.device('cpu')):
        """
        Params
        ======
        n_inputs: tamaño del espacio de estados
        n_outputs: tamaño del espacio de acciones
        actions: array de acciones posibles
        """
        super(DuelingDQN, self).__init__()
        self.device = device
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate

        # Construcción de la red neuronal
        # Red común
        self.common_net = torch.nn.Sequential(
            torch.nn.Linear(self.n_inputs, 256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128, bias=True),
            torch.nn.ReLU()
        )
        self.common_net.to(device=self.device)

        # Sub-red de la función de Valor
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(128, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1, bias=True),
            torch.nn.ReLU()
        )
        self.value_net.to(device=self.device)

        # Sub-red de la Ventaja A(s,a)
        self.adv_net = torch.nn.Sequential(
            torch.nn.Linear(128, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.n_outputs, bias=True),
            torch.nn.ReLU()
        )
        self.adv_net.to(device=self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        """Calcula los Q-values resultado de aplicar la red a cierto estado."""
        common_x = self.common_net(state)

        val = self.value_net(common_x)

        adv = self.adv_net(common_x)

        # Agregamos las dos subredes:
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        action = val + adv - adv.mean()

        return action

    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.forward(state_t)


if __name__ == '__main__':
    # Inicialización:
    env_dict = {'id': 'LunarLander-v2', 'render_mode': 'rgb_array'}
    environment = gym.make(**env_dict)
    # Utilizamos la cpu porque en este caso es más rápida:
    DEVICE = torch.device('cpu')
    agent_name = "dueling_dqn"
    agent_title = 'Agente Dueling DQN'
    try:
        os.mkdir(agent_name)
    except FileExistsError:
        pass

    # Fijamos las semillas utilizadas, por reproducibilidad:
    # Referencias:
    # + https://pytorch.org/docs/stable/notes/randomness.html,
    # + https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
    # + https://gymnasium.farama.org/content/migration-guide/
    RANDOM_SEED = 666
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    environment.reset(seed=RANDOM_SEED)
    environment.action_space.seed(RANDOM_SEED)

    # Hyperparams:
    MEMORY_SIZE = 10000  # Máxima capacidad del buffer
    BURN_IN = 1000  # Número de pasos iniciales usados para rellenar el buffer antes de entrenar
    MAX_EPISODES = 1000  # Número máximo de episodios (el agente debe aprender antes de llegar a este valor)
    INIT_EPSILON = 1  # Valor inicial de epsilon
    EPSILON_DECAY = .985  # Decaimiento de epsilon
    MIN_EPSILON = 0.01  # Valor mínimo de epsilon en entrenamiento
    GAMMA = 0.99  # Valor gamma de la ecuación de Bellman
    BATCH_SIZE = 32  # Conjunto a coger del buffer para la red neuronal
    LR = 0.001  # Velocidad de aprendizaje
    DNN_UPD = 1  # Frecuencia de actualización de la red neuronal
    DNN_SYNC = 1000  # Frecuencia de sincronización de pesos entre la red neuronal y la red objetivo

    # Agent initialization:
    er_buffer = ExperienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)
    dueling_dqn = DuelingDQN(env=environment, learning_rate=LR, device=DEVICE)
    dueling_dqn_agent = DoubleDQNAgent(
        env=environment,
        dnnetwork=dueling_dqn,
        buffer=er_buffer,
        epsilon=INIT_EPSILON,
        eps_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        min_epsilon=MIN_EPSILON
    )

    # Agent training:
    training_time = dueling_dqn_agent.train(
        gamma=GAMMA,
        max_episodes=MAX_EPISODES,
        dnn_update_frequency=DNN_UPD,
        dnn_sync_frequency=DNN_SYNC
    )
    print(f"Training time: {training_time} minutes.")
    # dueling_dqn_agent.dnnetwork.load_state_dict(torch.load(f'{agent_name}/{agent_name}_Trained_Model.pth'))
    # Training evaluation:
    plot_rewards(
        training_rewards=dueling_dqn_agent.training_rewards,
        mean_training_rewards=dueling_dqn_agent.mean_training_rewards,
        reward_threshold=environment.spec.reward_threshold,
        title=agent_title,
        save_file_name=f'{agent_name}/{agent_name}_rewards.png'
    )
    plot_losses(
        training_losses=dueling_dqn_agent.training_losses,
        title=agent_title,
        save_file_name=f'{agent_name}/{agent_name}_losses.png'
    )

    # Saving:
    torch.save(obj=dueling_dqn_agent.dnnetwork.state_dict(),
               f=f'{agent_name}/{agent_name}_Trained_Model.pth')

    # Evaluation:
    eval_eps = 0
    eval_games_seed = 0
    tr, _ = play_games_using_agent(
        enviroment_dict=env_dict,
        agent=dueling_dqn_agent,
        n_games=100,
        games_seed=eval_games_seed,
        eps=eval_eps
    )
    plot_evaluation_rewards(
        rewards=tr,
        reward_threshold=environment.spec.reward_threshold,
        title=agent_title,
        save_file_name=f'{agent_name}/{agent_name}_evaluation.png'
    )
    print(f'well_landed_eval_episodes: {sum(tr >= 200)}')
    print(f'landed_eval_episodes: {sum((tr < 200) & (tr >= 100))}')
    print(f'crashed_eval_episodes: {sum(tr < 100)}')
    # Bad implementation, no rendering.
    # Saving random game:
    save_agent_gif(
        env_dict=env_dict,
        ag=dueling_dqn_agent,
        save_file_name=f'{agent_name}/agente_{agent_name}.gif',
        eps=eval_eps
    )
