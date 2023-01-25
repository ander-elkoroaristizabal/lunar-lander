"""
Módulo que implementa las utilidades utilizadas por los scripts principales.

Incluye las funciones para observar y guardar en un gif episodios de un agente aleatorio o entrenado,
además de las funciones para realizar las gráficas de evaluación del entrenamiento y de la prueba de agentes.

Código parcialmente basado en la PEC 2 de la asignatura Aprendizaje por Refuerzo 2022-2023 S1.
"""
from typing import Dict, List

import PIL.ImageDraw as ImageDraw
import gym
import imageio
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Diccionario que mapea el entero de cada acción con su nombre:
ACTIONS_DICT = {
    0: 'None',
    1: 'Left engine',
    2: 'Main engine',
    3: 'Right engine'
}


def render_random_agent_episode(env: gym.Env):
    """
    Reproduce un episodio de un agente aleatorio.
    """
    if env.render_mode != 'human':
        raise ValueError('Env render_mode needs to be "human".')
    env.reset()
    total_reward = 0
    for timestep in range(env.spec.max_episode_steps):
        env.render()
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if done:
            print(f"Episode finished after {timestep + 1} time steps "
                  f"with reward {round(total_reward)}.")
            break


def _label_with_text(frame, state, action: str, reward: float, total_reward: float = None):
    """
    Añade información al 'frame' que recibe como entrada.

    Args:
        frame: imagen de un entorno GYM
        state: estado del entorno en el frame
        action: acción tomada
        reward: recompensa recibida
        total_reward: recompensa total del episodio

    Returns:
        Imagen con información añadida.
    """
    im = Image.fromarray(frame)
    im = im.resize((im.size[0] * 2, im.size[1] * 2))
    drawer = ImageDraw.Draw(im)
    total_reward_text = f"Total Reward = {total_reward}" if total_reward else ""
    drawer.text(xy=(1, im.size[1] - im.size[1] / 10),
                text="Uoc Aprendizaje Por Refuerzo.\n"
                     f"State = {state}\n"
                     f"Action = {action}\n"
                     f"Reward = {reward}\n" +
                     total_reward_text,
                fill=(0, 0, 0, 128))
    return im


def save_random_agent_gif(env: gym.Env, path: str = 'random_agent.gif'):
    """
    Guarda un gif de un agente aleatorio en la ruta indicada.
    """
    if env.render_mode != 'rgb_array':
        raise ValueError('Env render_mode needs to be "rgb_array".')
    frames = []
    env.reset()
    tr = 0
    ###########################################
    # Jugar una partida aleatoria:
    max_steps = env.spec.max_episode_steps
    for timestep in range(max_steps):
        action = env.action_space.sample()
        frame = env.render()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        tr += reward
        total_reward = round(tr, 2) if done else None
        frames.append(
            _label_with_text(
                frame=frame,
                state=list(obs),
                action=ACTIONS_DICT[action],
                reward=round(reward, 2),
                total_reward=total_reward
            )
        )
        if done:
            break
    ##############################################
    env.close()
    imageio.mimwrite(path, frames, fps=60)


def plot_rewards(training_rewards: List[float],
                 mean_training_rewards: List[float],
                 reward_threshold: float,
                 title: str,
                 save_file_name: str = None):
    """
    Dibuja las recompensas de cada episodio y medias durante el entrenamiento de un agente
    junto con el umbral de recompensa.

    Args:
        training_rewards: recompensa obtenida en cada episodio
        mean_training_rewards: recompensa media obtenida en cada episodio
        reward_threshold: umbral de recompensa a graficar
        title: título de la gráfica
        save_file_name: ruta donde guardar la gráfica
    """
    plt.figure(figsize=(12, 8))
    plt.plot(training_rewards, label='Rewards')
    plt.plot(mean_training_rewards, label='Mean Rewards')
    plt.axhline(reward_threshold, color='r', label="Reward threshold")
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.title(title)
    if save_file_name:
        plt.savefig(save_file_name)
    plt.show()


def plot_losses(training_losses: List[float],
                title: str,
                save_file_name: str = None):
    """
    Dibuja la pérdida en cada episodio del entrenamiento de un agente.

    Args:
        training_losses: pérdidas durante el entrenamiento
        title: título de la gráfica
        save_file_name: ruta donde guardar la gráfica
    """
    plt.figure(figsize=(12, 8))
    plt.plot(training_losses, label='Real Training loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    if save_file_name:
        plt.savefig(save_file_name)
    plt.show()


def plot_evaluation_rewards(rewards: np.ndarray,
                            reward_threshold: float,
                            title: str,
                            save_file_name: str = None):
    """
    Dibuja la recompensa de cada episodio de evaluación junto con la media, la mediana y el umbral de recompensa.

    Args:
        rewards: recompensa obtenida en cada episodio
        reward_threshold: umbral de recompensa
        title: título de la gráfica
        save_file_name: ruta donde guardar la gráfica

    Returns:

    """
    plt.figure(figsize=(12, 8))
    plt.plot(rewards, label='Total episode reward')
    r_mean = rewards.mean()
    plt.axhline(y=r_mean, label=f'Mean of episode Rewards ({round(r_mean)})', color='orange')
    r_median = float(np.median(rewards))
    plt.axhline(y=r_median, label=f'Median of episode Rewards ({round(r_median)})', color='purple')
    plt.axhline(y=reward_threshold, label='Reward Threshold', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.title(title)
    if save_file_name:
        plt.savefig(save_file_name)
    plt.show()


def save_agent_gif(env_dict: Dict,
                   ag,
                   save_file_name: str,
                   eps: float = 0.01,
                   game_seed: int = None):
    """
    Guarda un gif del agente 'ag' en la ruta indicada.

    Args:
        env_dict: diccionario que define el entorno gym a utilizar
        ag: agente entrenado
        eps: epsilon de la política epsilon-greedy del agente
        save_file_name: nombre del fichero
        game_seed: semilla de la partida
    """
    # Comprobamos que el 'render_mode' sea adecuado e inicializamos el entorno:
    if env_dict['render_mode'] != 'rgb_array':
        raise ValueError('Env render_mode needs to be "rgb_array".')
    env = gym.make(**env_dict)

    # Utilizamos una semilla para generar el episodio si se ha recibido como argumento:
    if game_seed:
        obs, _ = env.reset(seed=game_seed)
        env.action_space.seed(game_seed)
        np.random.seed(game_seed)
    # Si no es así el episodio es aleatorio:
    else:
        obs, _ = env.reset()

    # Jugamos el episodio almacenando los frames informados:
    frames = []
    tr = 0
    max_steps = env.spec.max_episode_steps
    for timestep in range(max_steps):
        action = ag.get_action(state=obs, epsilon=eps)
        frame = env.render()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        tr += reward
        total_reward = round(tr, 2) if done else None
        # Añadimos la información al frame:
        frames.append(
            _label_with_text(
                frame=frame,
                state=list(obs),
                action=ACTIONS_DICT[action],
                reward=round(reward, 2),
                total_reward=total_reward
            )
        )
        if done:
            break

    env.close()
    # Guardamos el gif:
    imageio.mimwrite(save_file_name, frames, fps=60)
    return tr


def render_agent_episode(env_dict: Dict, ag, game_seed: int = None, eps: float = 0.01):
    """
    Reproduce una partida

    Args:
        env_dict: diccionario que define el entorno gym a utilizar
        ag: agente a utilizar
        game_seed: semilla aleatoria a utilizar
        eps: epsilon a utilizar en la política epsilon-greedy del agente
    """
    # Comprobamos que el 'render_mode' sea adecuado e inicializamos el entorno:
    if env_dict['render_mode'] != 'human':
        raise ValueError('Env render_mode needs to be "human".')
    env = gym.make(**env_dict)

    # Utilizamos una semilla para generar el episodio si se ha recibido como argumento:
    if game_seed:
        obs, _ = env.reset(seed=game_seed)
        env.action_space.seed(game_seed)
        np.random.seed(game_seed)
    else:
        # Si no es así el episodio es aleatorio:
        obs, _ = env.reset()

    # Jugamos el episodio reproduciéndolo en vivo:
    tr = 0
    max_steps = env.spec.max_episode_steps
    for timestep in range(max_steps):
        action = ag.get_action(state=obs, epsilon=eps)
        env.render()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        tr += reward
        total_reward = round(tr, 2) if done else None
        if done:
            break
    print(f"Episode finished after {timestep + 1} time steps "
          f"with reward {round(total_reward)}.")
    env.close()
