"""Script que implementa las clases de red y agente Reinforce with Baseline y permite entrenar y evaluar un agente.

Define la clase PGReinforce, que contiene la red neuronal a utilizar por el agente Reinforce.
Define la clase ReinforceAgent, una implementación de un agente Reinforce with Baseline.
Al ejecutarse define, entrena y evalúa un agente Reinforce with Baseline en el entorno LunarLander-v2.

Fuente del código base utilizado: https://github.com/jcasasr/Aprendizaje-por-refuerzo/blob/main/M10
"""
import os
import random
import time

import gym
import numpy as np
import torch

from playing import play_games_using_agent
from utils import plot_evaluation_rewards, save_agent_gif, plot_rewards, plot_losses


class PGReinforce(torch.nn.Module):
    """
    Red neuronal a utilizar por un agente Reinforce.
    """

    def __init__(self, env: gym.Env, learning_rate: float = 1e-3, device: torch.device = torch.device('cpu')):
        """
        Inicializa la clase PGReinforce utilizando la información del entorno.
        Inicializa el optimizador Adam usando el 'learning_rate' proporcionado.
        Mueve la red al dispositivo 'device'.

        Args:
            env: entorno gym donde se utilizará la red
            learning_rate: tasa de aprendizaje de la optimización
            device: dispositivo que se utilizará para entrenar la red
        """
        super(PGReinforce, self).__init__()
        self.device = device
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.learning_rate = learning_rate

        # Construcción de la red neuronal
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_inputs, 256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.n_outputs, bias=True),
            torch.nn.Softmax(dim=-1)
        )
        self.model.to(self.device)

        # Inicialización del optimizador:
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # Obtención de las probabilidades de las posibles acciones
    def get_action_prob(self, state, *kwargs):
        """
        Obtiene la probabilidad aproximada de cada acción en la observación 'state'.
        Args:
            state: observación del entorno
        """
        if type(state) is tuple:
            state = np.array(state)
        state = torch.FloatTensor(state).to(device=self.device)
        action_probs = self.model(state)
        return action_probs


class ReinforceAgent:
    """
    Agente Reinforce with Baseline.
    """

    def __init__(self, env: gym.Env, dnnetwork):
        """
        Inicializa el agente Reinforce.

        Args:
            env: entorno gym
            dnnetwork: red neuronal a entrenar
        """
        self.env = env
        self.dnnetwork = dnnetwork
        self.nblock = 100  # bloque de los X últimos episodios de los que se calculará la media de recompensa
        self.reward_threshold = self.env.spec.reward_threshold  # recompensa media a partir de la cual se considera
        # que el agente ha aprendido a jugar
        # Other initializations:
        self.action_space = np.arange(self.env.action_space.n)
        self.batch_rewards = []
        self.batch_actions = []
        self.batch_states = []
        self.batch_counter = 1
        self.training_rewards = []
        self.training_losses = []
        self.mean_training_rewards = []
        self.update_loss = []
        self.gamma = None

    def get_action(self, state, **kwargs):
        """
        Devuelve la acción a seguir según la política del agente en la observación 'state'.

        Args:
            state: observación del entorno
        """
        action_probs = self.dnnetwork.get_action_prob(state).detach().cpu().numpy()
        return np.random.choice(self.action_space, p=action_probs.flatten())

    # Entrenamiento
    def train(self,
              gamma: float = 0.99,
              max_episodes: int = 2000,
              batch_size: int = 10):
        """
        Entrena al agente.

        Args:
            gamma: valor de la constante gamma de la ecuación de Bellman
            max_episodes: número máximo de episodios de entrenamiento
            batch_size: tamaño del batch de entrenamiento

        Returns:
            tiempo de entrenamiento (en minutos)
        """
        start_time = time.time()

        self.gamma = gamma

        # Iniciamos el entrenamiento:
        episode = 0
        training = True
        print("Training...")
        while training:
            state_t, _ = self.env.reset()
            episode_states = []
            episode_rewards = []
            episode_actions = []
            done = False

            while not done:
                # Obtenemos las acciones
                action = self.get_action(state_t)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Almacenamos las experiencias que se van obteniendo en este episodio
                episode_states.append(state_t)
                episode_rewards.append(reward)
                episode_actions.append(action)
                state_t = next_state

                if done:
                    episode += 1
                    # Calculamos el término del retorno menos la línea de base
                    self.batch_rewards.extend(self.discount_rewards(episode_rewards))
                    self.batch_states.extend(episode_states)
                    self.batch_actions.extend(episode_actions)
                    self.training_rewards.append(sum(episode_rewards))  # guardamos las recompensas obtenidas

                    # Actualizamos la red cuando se completa el tamaño del batch
                    if self.batch_counter == batch_size:
                        self.update(self.batch_states, self.batch_rewards, self.batch_actions)
                        self.training_losses.append(sum(self.update_loss) / len(self.update_loss))
                        self.update_loss = []

                        # Reseteamos las variables del episodio
                        self.batch_rewards = []
                        self.batch_actions = []
                        self.batch_states = []
                        self.batch_counter = 1

                    # Actualizamos el contador del batch
                    self.batch_counter += 1

                    # Calculamos la media de recompensa de los últimos X episodios
                    mean_rewards = np.mean(self.training_rewards[-self.nblock:])
                    self.mean_training_rewards.append(mean_rewards)

                    print("\rEpisode {:d} Mean Rewards {:.2f}\t\t".format(
                        episode, mean_rewards), end="")

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

    def discount_rewards(self, rewards):
        """
        Calcula las recompensas descontadas aplicando el descuento y el baseline fijados.

        Args:
            rewards: recompensas del episodio
        """
        discount_r = np.zeros_like(rewards)
        steps = range(len(rewards))
        reward_sum = 0
        for i in reversed(steps):  # revertimos la dirección del vector para hacer la suma cumulativa
            reward_sum = rewards[i] + self.gamma * reward_sum
            discount_r[i] = reward_sum
        # Utilizamos el retorno descontado para calcular el retorno modificado:
        new_discount_r = (discount_r - np.mean(discount_r)) / np.std(discount_r)
        # Y devolvemos el retorno modificado:
        return new_discount_r

    # Actualización
    def update(self, batch_s, batch_r, batch_a):
        """
        Actualiza la red principal.

        Args:
            batch_s: estados del batch
            batch_r: recompensas del batch
            batch_a: acciones del batch
        """
        self.dnnetwork.optimizer.zero_grad()  # eliminamos cualquier gradiente pasado
        state_t = torch.FloatTensor(np.array(batch_s))
        reward_t = torch.FloatTensor(np.array(batch_r))
        action_t = torch.LongTensor(np.array(batch_a))
        loss = self.calculate_loss(state_t, action_t, reward_t)  # calculamos la pérdida
        loss.backward()  # hacemos la diferencia para obtener los gradientes
        self.dnnetwork.optimizer.step()  # aplicamos los gradientes a la red neuronal
        # Guardamos los valores de pérdida
        if self.dnnetwork.device != 'cpu':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

    # Cálculo de la pérdida
    # Recordatorio: cada actualización es proporcional al producto del retorno y el gradiente de la probabilidad
    # de tomar la acción tomada, dividido por la probabilidad de tomar esa acción (logaritmo natural)
    def calculate_loss(self, state_t, action_t, reward_t):
        """
        Calcula la pérdida correspondiente a un batch de experiencias.

        Args:
            state_t: tensor de estados del batch
            reward_t: tensor de recompensas del batch
            action_t: tensor de acciones del batch
        """
        reward_t = reward_t.to(self.dnnetwork.device)
        logprobs = torch.log(self.dnnetwork.get_action_prob(state_t))
        selected_logprobs = reward_t * logprobs[np.arange(len(action_t)), action_t]
        loss = -selected_logprobs.mean()
        return loss


if __name__ == '__main__':
    # Inicialización:
    env_dict = {'id': 'LunarLander-v2', 'render_mode': 'rgb_array'}
    environment = gym.make(**env_dict)
    # Utilizamos la cpu porque en este caso es más rápida:
    DEVICE = torch.device('cpu')
    agent_name = "reinforce"
    agent_title = 'Agente Reinforce'
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
    LR = 0.001  # Velocidad aprendizaje
    MAX_EPISODES = 10000  # Número máximo de episodios (el agente debe aprender antes de llegar a este valor)
    GAMMA = 0.99
    BATCH_SIZE = 32

    # Agent initialization:
    reinforce_network = PGReinforce(env=environment, learning_rate=LR, device=DEVICE)
    reinforce_agent = ReinforceAgent(
        env=environment,
        dnnetwork=reinforce_network
    )

    # Agent training:
    training_time = reinforce_agent.train(
        gamma=GAMMA,
        max_episodes=MAX_EPISODES
    )
    print(f"Training time: {training_time} minutes.")

    # Training evaluation:
    plot_rewards(
        training_rewards=reinforce_agent.training_rewards,
        mean_training_rewards=reinforce_agent.mean_training_rewards,
        reward_threshold=environment.spec.reward_threshold,
        title=agent_title,
        save_file_name=f'{agent_name}/{agent_name}_rewards.png'
    )
    plot_losses(
        training_losses=reinforce_agent.training_losses,
        title=agent_title,
        save_file_name=f'{agent_name}/{agent_name}_losses.png'
    )

    # Saving:
    torch.save(obj=reinforce_agent.dnnetwork.state_dict(),
               f=f'{agent_name}/{agent_name}_Trained_Model.pth')

    # Evaluation:
    eval_eps = 0
    eval_games_seed = 0
    tr, _ = play_games_using_agent(
        environment_dict=env_dict,
        agent=reinforce_agent,
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
    # Bad results, no rendering.
    # Saving random game:
    save_agent_gif(
        env_dict=env_dict,
        ag=reinforce_agent,
        save_file_name=f'{agent_name}/agente_{agent_name}.gif',
        eps=eval_eps
    )
