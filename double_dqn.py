import os
import time
from copy import deepcopy

import gym
import numpy as np
import torch

from dqn import DQN
from playing import play_games_using_agent
from replay_buffer import ExperienceReplayBuffer
from utils import plot_rewards, plot_losses, plot_evaluation_rewards, save_agent_gif


class DoubleDQNAgent:

    def __init__(self, env, dnnetwork, buffer, epsilon=0.1, eps_decay=0.99, batch_size=32, min_epsilon=0.01):

        self.env = env
        self.dnnetwork = dnnetwork
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
    def take_step(self, eps, mode='train'):
        if mode == 'explore':
            # acción aleatoria en el burn-in y en la fase de exploración (epsilon)
            action = self.env.action_space.sample()
        else:
            # acción a partir del valor de Q (elección de la acción con mejor Q)
            action = self.get_action(self.state0, eps)
            self.step_count += 1

        # Realizamos la acción y obtenemos el nuevo estado y la recompensa
        new_state, reward, done, _, _ = self.env.step(action)

        self.total_reward += reward
        self.buffer.append(self.state0, action, reward, done, new_state)  # guardamos experiencia en el buffer
        self.state0 = new_state.copy()

        if done:
            self.state0, _ = self.env.reset()
        return done

    # Entrenamiento
    def train(self, gamma=0.99,
              max_episodes=50000,
              dnn_update_frequency=4,
              dnn_sync_frequency=2000):
        start_time = time.time()

        self.gamma = gamma

        # Rellenamos el buffer con N experiencias aleatorias ()
        self.state0, _ = self.env.reset()
        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(self.epsilon, mode='explore')

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

                if self.env._elapsed_steps >= self.env.spec.max_episode_steps:
                    done_game = True

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
        # Separamos las variables de la experiencia y las convertimos a tensores
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_vals = torch.FloatTensor(rewards).to(device=self.dnnetwork.device).reshape(-1, 1)
        actions_vals = torch.LongTensor(np.array(actions)).to(
            device=self.dnnetwork.device).reshape(-1, 1)
        dones_t = torch.BoolTensor(dones).to(device=self.dnnetwork.device).reshape(-1, 1)

        # Obtenemos los valores de Q de la red principal
        qvals = torch.gather(self.dnnetwork.get_qvals(states), 1, actions_vals)
        next_actions = torch.max(self.dnnetwork.get_qvals(next_states), dim=-1)[1]
        next_actions_vals = next_actions.reshape(-1, 1)
        # Obtenemos los valores de Q de la red objetivo
        target_qvals = self.target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_vals).detach()
        #####

        qvals_next.masked_fill_(dones_t, 0)  # 0 en estados terminales

        # Calculamos la ecuación de Bellman
        expected_qvals = self.gamma * qvals_next + rewards_vals

        # Calculamos la pérdida
        loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1, 1))
        return loss

    def update(self):
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

    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.dnnetwork.actions)  # acción aleatoria
        else:
            qvals = self.dnnetwork.get_qvals(state)  # acción a partir del cálculo del valor de Q para esa acción
            action = torch.max(qvals, dim=-1)[1].item()
        return action


if __name__ == '__main__':
    # Inicialización:
    environment = gym.make('LunarLander-v2', render_mode='rgb_array')
    DEVICE = torch.device('cpu')
    agent_name = "double_dqn"
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
    environment.np_random, _ = gym.utils.seeding.np_random(RANDOM_SEED)
    environment.action_space.seed(RANDOM_SEED)

    # Hyperparams:
    MEMORY_SIZE = 10000  # Máxima capacidad del buffer
    BURN_IN = 100  # Número de pasos iniciales usados para rellenar el buffer antes de entrenar
    MAX_EPISODES = 1000  # Número máximo de episodios (el agente debe aprender antes de llegar a este valor)
    INIT_EPSILON = 1  # Valor inicial de epsilon
    EPSILON_DECAY = .98  # Decaimiento de epsilon
    MIN_EPSILON = 0.01  # Valor mínimo de epsilon en entrenamiento
    GAMMA = 0.99  # Valor gamma de la ecuación de Bellman
    BATCH_SIZE = 32  # Conjunto a coger del buffer para la red neuronal
    LR = 0.001  # Velocidad de aprendizaje
    DNN_UPD = 1  # Frecuencia de actualización de la red neuronal
    DNN_SYNC = 1000  # Frecuencia de sincronización de pesos entre la red neuronal y la red objetivo

    # Agent initialization:
    er_buffer = ExperienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)
    double_dqn = DQN(env=environment, learning_rate=LR, device=DEVICE)
    double_dqn_agent = DoubleDQNAgent(
        env=environment,
        dnnetwork=double_dqn,
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
    # dqn_agent.dnnetwork.load_state_dict(torch.load(f'dqn_Trained_Model.pth'))
    # Training evaluation:
    plot_rewards(
        training_rewards=double_dqn_agent.training_rewards,
        mean_training_rewards=double_dqn_agent.mean_training_rewards,
        reward_threshold=environment.spec.reward_threshold,
        title=agent_name,
        save_file_name=f'{agent_name}/{agent_name}_rewards.png'
    )
    plot_losses(
        training_losses=double_dqn_agent.training_losses,
        title=agent_name,
        save_file_name=f'{agent_name}/{agent_name}_losses.png'
    )

    # Saving:
    torch.save(obj=double_dqn_agent.dnnetwork.state_dict(),
               f=f'{agent_name}/{agent_name}_Trained_Model.pth')

    # Evaluation:
    tr, _ = play_games_using_agent(environment, double_dqn_agent, 100)
    plot_evaluation_rewards(
        rewards=tr,
        reward_threshold=environment.spec.reward_threshold,
        title=agent_name,
        save_file_name=f'{agent_name}/{agent_name}_evaluation.png'
    )
    save_agent_gif(env=environment, ag=double_dqn_agent, save_file_name=f'{agent_name}/agente_{agent_name}.gif')
