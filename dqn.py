"""Script que implementa las clases de red y agente DQN y permite entrenar y evaluar un agente.

Define la clase DQN, que contiene la red neuronal a utilizar por el agente DQN.
Define la clase DQNAgent, una implementación de un agente DQN.
Al ejecutarse define, entrena y evalúa un agente DQN en el entorno LunarLander-v2.

Fuente del código base utilizado: https://github.com/jcasasr/Aprendizaje-por-refuerzo/blob/main/M09
"""
import os
import random
import time
from copy import deepcopy

import gym
import numpy as np
import torch

from playing import play_games_using_agent
from replay_buffer import ExperienceReplayBuffer
from utils import plot_evaluation_rewards, save_agent_gif, render_agent_episode, plot_rewards, plot_losses


class DQN(torch.nn.Module):
    """
    Red neuronal a utilizar por un agente DQN.
    """

    def __init__(self, env: gym.Env, learning_rate: float = 1e-3, device: torch.device = torch.device('cpu')):
        """
        Inicializa la clase DQN utilizando la información del entorno.
        Inicializa el optimizador Adam usando el 'learning_rate' proporcionado.
        Mueve la red al dispositivo 'device'.

        Args:
            env: entorno gym donde se utilizará la red
            learning_rate: tasa de aprendizaje de la optimización
            device: dispositivo que se utilizará para entrenar la red
        """
        super(DQN, self).__init__()
        self.device = device
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate

        # Construcción de la red neuronal
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_inputs, 256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.n_outputs, bias=True)
        )
        self.model.to(self.device)

        # Inicialización del optimizador:
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_qvals(self, state):
        """
        Devuelve los Q-valores aproximados por la red para la observación 'state'.

        Args:
            state: observación del entorno
        """
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.model(state_t)


class DQNAgent:
    """
    Agente DQN.
    """

    def __init__(self,
                 env: gym.Env,
                 dnnetwork, buffer,
                 epsilon: float = 0.1,
                 eps_decay: float = 0.99,
                 min_epsilon: float = 0.01,
                 batch_size: int = 32):
        """
        Inicializa el agente DQN.

        Args:
            env: entorno gym
            dnnetwork: red neuronal principal a entrenar
            buffer: buffer de repetición a utilizar
            epsilon: epsilon inicial
            eps_decay: decaimiento de epsilon
            min_epsilon: epsilon mínimo de entrenamiento
            batch_size: tamaño del batch de entrenamiento
        """
        self.env = env  # Entorno
        self.dnnetwork = dnnetwork  # Red principal
        self.target_network = deepcopy(dnnetwork)  # red objetivo (copia de la principal)
        self.buffer = buffer
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.nblock = 100  # bloque de los X últimos episodios de los que se calculará la media de recompensa
        self.reward_threshold = self.env.spec.reward_threshold  # recompensa media a partir de la cual se considera
        # que el agente ha aprendido a jugar
        # Otras variables inicializadas:
        self.update_loss = []
        self.training_rewards = []
        self.mean_training_rewards = []
        self.training_losses = []
        self.sync_eps = []
        self.total_reward = 0
        self.step_count = 0
        self.state0 = None
        self.gamma = None  # Defined on training

    # Tomamos una nueva acción
    def take_step(self, eps: float, mode: str = 'train'):
        """
        Avanza un paso en el episodio.

        Args:
            eps: epsilon a utilizar por el agente
            mode: modo de elección de acciones
        """
        if mode == 'explore':
            # acción aleatoria en el burn-in y en la fase de exploración (epsilon)
            action = self.env.action_space.sample()
        else:
            # acción a partir del valor de Q (elección de la acción con mejor Q)
            action = self.get_action(self.state0, eps)
            self.step_count += 1

        # Realizamos la acción y obtenemos el nuevo estado y la recompensa
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        self.total_reward += reward
        self.buffer.append(self.state0, action, reward, terminated, new_state)  # guardamos experiencia en el buffer
        self.state0 = new_state.copy()

        if done:
            self.state0, _ = self.env.reset()
        return done

    # Entrenamiento
    def train(self,
              gamma: float = 0.99,
              max_episodes: int = 50000,
              dnn_update_frequency: int = 4,
              dnn_sync_frequency: int = 2000):
        """
        Entrena al agente.

        Args:
            gamma: valor de la constante gamma de la ecuación de Bellman
            max_episodes: número máximo de episodios de entrenamiento
            dnn_update_frequency: frecuencia de actualización de la red principal
            dnn_sync_frequency: frecuencia de sincronización de las redes

        Returns:
            tiempo de entrenamiento (en minutos)
        """
        start_time = time.time()

        self.gamma = gamma

        # Rellenamos el buffer con N experiencias aleatorias:
        self.state0, _ = self.env.reset()
        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(self.epsilon, mode='explore')

        # Iniciamos el entrenamiento:
        episode = 0
        training = True
        print("Training...")
        while training:
            self.state0, _ = self.env.reset()
            self.total_reward = 0
            done_game = False
            while not done_game:
                # El agente toma una acción
                done_game = self.take_step(self.epsilon, mode='train')

                # Actualizamos la red principal según la frecuencia establecida
                if self.step_count % dnn_update_frequency == 0:
                    self.update()
                # Sincronizamos la red principal y la red objetivo según la frecuencia establecida
                if self.step_count % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.dnnetwork.state_dict())
                    self.sync_eps.append(episode)

                # Si el episodio ha concluido:
                if done_game:
                    episode += 1
                    self.training_rewards.append(self.total_reward)  # guardamos las recompensas obtenidas
                    self.training_losses.append(sum(self.update_loss) / len(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(  # calculamos la media de recompensa de los últimos X episodios
                        self.training_rewards[-self.nblock:])
                    self.mean_training_rewards.append(mean_rewards)

                    print("\rEpisode {:d} Mean Rewards {:.2f} Epsilon {}\t\t".format(
                        episode, mean_rewards, self.epsilon), end="")

                    # Comprobamos que todavía quedan episodios
                    if episode >= max_episodes:
                        print('\nEpisode limit reached.')
                        end_time = time.time()
                        return round((end_time - start_time) / 60, 2)

                    # Termina el juego si la media de recompensas ha llegado al umbral fijado para este juego
                    if mean_rewards >= self.reward_threshold:
                        print('\nEnvironment solved in {} episodes!'.format(
                            episode))
                        end_time = time.time()
                        return round((end_time - start_time) / 60, 2)

                    # Actualizamos epsilon según la velocidad de decaimiento fijada
                    self.epsilon = max(self.epsilon * self.eps_decay, self.min_epsilon)

    # Cálculo de la pérdida
    def calculate_loss(self, batch):
        """
        Calcula la pérdida correspondiente a un batch de experiencias.
        """
        # Separamos las variables de la experiencia y las convertimos a tensores
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_vals = torch.FloatTensor(rewards).to(device=self.dnnetwork.device)
        actions_vals = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(
            device=self.dnnetwork.device)
        dones_t = torch.BoolTensor(dones).to(device=self.dnnetwork.device)

        # Obtenemos los valores de Q de la red principal
        qvals = torch.gather(self.dnnetwork.get_qvals(states), 1, actions_vals)
        # Obtenemos los valores de Q objetivo. El parámetro detach() evita que estos valores actualicen la red objetivo
        qvals_next = torch.max(self.target_network.get_qvals(next_states),
                               dim=-1)[0].detach()
        qvals_next.masked_fill_(dones_t, 0)  # 0 en estados terminales

        # Calculamos la ecuación de Bellman
        expected_qvals = self.gamma * qvals_next + rewards_vals

        # Calculamos la pérdida
        loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1, 1))
        return loss

    def update(self):
        """
        Actualiza la red principal.
        """
        self.dnnetwork.optimizer.zero_grad()  # eliminamos cualquier gradiente pasado
        batch = self.buffer.sample_batch(batch_size=self.batch_size)  # seleccionamos un conjunto del buffer
        loss = self.calculate_loss(batch)  # calculamos la pérdida
        loss.backward()  # hacemos la diferencia para obtener los gradientes
        self.dnnetwork.optimizer.step()  # aplicamos los gradientes a la red neuronal
        # Guardamos los valores de pérdida
        if self.dnnetwork.device != 'cpu':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

    def get_action(self, state, epsilon=0.01):
        """
        Devuelve la acción a seguir según el agente siguiendo una política eps-greedy en la observación 'state'.

        Args:
            state: observación del entorno
            epsilon: epsilon a utilizar por la política epsilon-greedy
        """
        if np.random.random() < epsilon:
            action = np.random.choice(self.dnnetwork.actions)  # acción aleatoria
        else:
            qvals = self.dnnetwork.get_qvals(state)  # acción a partir del cálculo del valor de Q para esa acción
            action = torch.max(qvals, dim=-1)[1].item()
        return action


if __name__ == '__main__':
    # Inicialización:
    env_dict = {'id': 'LunarLander-v2', 'render_mode': 'rgb_array'}
    environment = gym.make(**env_dict)
    # Utilizamos la cpu porque en este caso es más rápida:
    DEVICE = torch.device('cpu')
    agent_name = "dqn"
    agent_title = 'Agente DQN'
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
    dqn = DQN(env=environment, learning_rate=LR, device=DEVICE)
    dqn_agent = DQNAgent(
        env=environment,
        dnnetwork=dqn,
        buffer=er_buffer,
        epsilon=INIT_EPSILON,
        eps_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        min_epsilon=MIN_EPSILON
    )

    # Agent training:
    training_time = dqn_agent.train(
        gamma=GAMMA,
        max_episodes=MAX_EPISODES,
        dnn_update_frequency=DNN_UPD,
        dnn_sync_frequency=DNN_SYNC
    )
    print(f"Training time: {training_time} minutes.")
    # dqn_agent.dnnetwork.load_state_dict(torch.load(f'{agent_name}/{agent_name}_Trained_Model.pth'))
    # Training evaluation:
    plot_rewards(
        training_rewards=dqn_agent.training_rewards,
        mean_training_rewards=dqn_agent.mean_training_rewards,
        reward_threshold=environment.spec.reward_threshold,
        title=agent_title,
        save_file_name=f'{agent_name}/{agent_name}_rewards.png'
    )
    plot_losses(
        training_losses=dqn_agent.training_losses,
        title=agent_title,
        save_file_name=f'{agent_name}/{agent_name}_losses.png'
    )

    # Saving:
    torch.save(obj=dqn_agent.dnnetwork.state_dict(),
               f=f'{agent_name}/{agent_name}_Trained_Model.pth')

    # Evaluation:
    eval_eps = 0
    eval_games_seed = 0
    tr, _ = play_games_using_agent(
        environment_dict=env_dict,
        agent=dqn_agent,
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
    print(f"Rewards std: {tr.std()}")
    print(f'well_landed_eval_episodes: {sum(tr >= 200)}')
    print(f'landed_eval_episodes: {sum((tr < 200) & (tr >= 100))}')
    print(f'crashed_eval_episodes: {sum(tr < 100)}')
    # Rendering interesting games:
    gif_games = sorted(np.where(tr < 200)[0])
    render_env_dict = {'id': 'LunarLander-v2', 'render_mode': 'human'}
    for episode_n in gif_games + [27, 61, 72, 94]:  # Episodios donde la DDQN no es óptima
        print(f"Rendering episode number {episode_n}.")
        render_agent_episode(
            env_dict=render_env_dict,
            ag=dqn_agent,
            game_seed=eval_games_seed + int(episode_n),
            eps=eval_eps
        )
    # Saving random game:
    save_agent_gif(
        env_dict=env_dict,
        ag=dqn_agent,
        save_file_name=f'{agent_name}/agente_{agent_name}.gif',
        eps=eval_eps
    )
