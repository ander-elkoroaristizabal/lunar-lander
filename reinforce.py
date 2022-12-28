import os
import time

import gym
import numpy as np
import torch

from playing import play_games_using_agent
from utils import plot_evaluation_rewards, save_agent_gif, plot_rewards, plot_losses


class PGReinforce(torch.nn.Module):

    def __init__(self, env, learning_rate=1e-3, device=torch.device('cpu')):
        super(PGReinforce, self).__init__()
        self.device = device
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.learning_rate = learning_rate

        # Construcción de la red neuronal
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_inputs, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.n_outputs),
            torch.nn.Softmax(dim=-1))
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # Obtención de las probabilidades de las posibles acciones
    def get_action_prob(self, state, *kwargs):
        if type(state) is tuple:
            state = np.array(state)
        state = torch.FloatTensor(state).to(device=self.device)
        action_probs = self.model(state)
        return action_probs


class ReinforceAgent:

    def __init__(self, env, dnnetwork):

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
        action_probs = self.dnnetwork.get_action_prob(state).detach().cpu().numpy()
        return np.random.choice(self.action_space, p=action_probs.flatten())

    # Entrenamiento
    def train(self, gamma=0.99, max_episodes=2000, batch_size=10, env_seed=666):
        start_time = time.time()
        self.gamma = gamma

        episode = 0
        training = True
        print("Training...")
        while training:
            state_t, _ = self.env.reset(seed=env_seed + episode)
            episode_states = []
            episode_rewards = []
            episode_actions = []
            done_game = False

            while not done_game:
                # Obtenemos las acciones
                action = self.get_action(state_t)
                next_state, reward, done_game, _, _ = self.env.step(action)

                # Almacenamos las experiencias que se van obteniendo en este episodio
                episode_states.append(state_t)
                episode_rewards.append(reward)
                episode_actions.append(action)
                state_t = next_state

                if self.env._elapsed_steps >= self.env.spec.max_episode_steps:
                    done_game = True

                if done_game:
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
        reward_t = reward_t.to(self.dnnetwork.device)
        logprobs = torch.log(self.dnnetwork.get_action_prob(state_t))
        selected_logprobs = reward_t * logprobs[np.arange(len(action_t)), action_t]
        loss = -selected_logprobs.mean()
        return loss


if __name__ == '__main__':
    # Inicialización:
    environment = gym.make('LunarLander-v2', render_mode='rgb_array')
    DEVICE = torch.device('mps')
    saves_path = "reinforce"
    try:
        os.mkdir(saves_path)
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
    lr = 0.001  # Velocidad aprendizaje
    MAX_EPISODES = 2000  # Número máximo de episodios (el agente debe aprender antes de llegar a este valor)
    GAMMA = 0.97
    BATCH_SIZE = 32

    # Agent initialization:
    reinforce_network = PGReinforce(env=environment, learning_rate=lr, device=DEVICE)
    reinforce_agent = ReinforceAgent(
        env=environment,
        dnnetwork=reinforce_network
    )

    # Agent training:
    training_time = reinforce_agent.train(
        gamma=GAMMA,
        max_episodes=MAX_EPISODES,
        env_seed=RANDOM_SEED
    )
    print(f"Training time: {training_time} minutes.")

    # Training evaluation:
    plot_rewards(
        training_rewards=reinforce_agent.training_rewards,
        mean_training_rewards=reinforce_agent.mean_training_rewards,
        reward_threshold=environment.spec.reward_threshold,
        save_file_name=f'{saves_path}/reinforce_rewards.png'
    )
    plot_losses(training_losses=reinforce_agent.training_losses, save_file_name=f'{saves_path}/reinforce_losses.png')
    # reinforce_agent.dnnetwork.load_state_dict(torch.load(f'{saves_path}/reinforce_Trained_Model.pth'))

    # Saving:
    torch.save(reinforce_agent.dnnetwork.state_dict(), f'{saves_path}/reinforce_Trained_Model.pth')

    # Evaluation:
    tr, _ = play_games_using_agent(environment, reinforce_agent, 100)
    plot_evaluation_rewards(
        rewards=tr,
        save_file_name=f'{saves_path}/reinforce_evaluation.png',
        reward_threshold=environment.spec.reward_threshold
    )
    save_agent_gif(environment, reinforce_agent, f'{saves_path}/agente_reinforce.gif')
