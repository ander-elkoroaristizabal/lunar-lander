"""
Módulo que implementa las funciones utilizadas para jugar episodios por los scripts principales.

Incluye las funciones para jugar una o varias partidas usando tanto una política determinada
(que puede ser la aleatoria) como un agente.

Las funciones que emplean agentes entrenados reciben un diccionario que caracteriza el entorno por reproducibilidad,
mientras que las funciones de agente aleatorio reciben directamente el entorno.
"""
from typing import Callable, Tuple

import gym
import numpy as np
import tqdm


def play_game(environment: gym.Env, policy: Callable) -> Tuple[float, int]:
    """
    Ejecución de un episodio del entorno
    en el que el agente sigue la política 'policy'.

    La política debe depender del argumento 'observation'.

    Devuelve la recompensa del episodio y el número de pasos.
    """

    # Inicializamos el entorno:
    obs_t, _ = environment.reset()
    t_step = 0
    total_game_reward = 0
    done = False

    while not done:
        # Elegir una acción:
        action_t = policy(observation=obs_t)

        # Ejecutar la acción y esperar la respuesta del entorno
        obs_t_plus_one, reward_t_plus_one, terminated, truncated, _ = environment.step(action_t)
        done = terminated or truncated
        # Actualizar variables
        total_game_reward += reward_t_plus_one
        t_step += 1
        obs_t = obs_t_plus_one

    return total_game_reward, t_step


def play_games(environment: gym.Env, policy: Callable, n_games: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Juega 'n_games' partidas siguiendo la política 'policy'.

    La política debe depender del argumento 'observation'.

    Devuelve el vector de recompensas y el vector de duración de partidas.
    """
    # Jugamos los episodios indicados y registramos el resultado de cada:
    game_rewards = np.zeros(n_games)
    game_steps = np.zeros(n_games)
    for game in tqdm.tqdm(range(n_games), desc="Progress playing games"):
        game_reward, steps = play_game(environment=environment, policy=policy)
        game_rewards[game] = game_reward
        game_steps[game] = steps

    return game_rewards, game_steps


def play_game_using_agent(env_dict: dict, agent, eps: float, game_seed: int = None) -> Tuple[float, int]:
    """
    Ejecución de un episodio del entorno
    utilizando el agente 'agent'.

    Devuelve la recompensa total del episodio y el número de pasos necesitados.
    """
    env = gym.make(**env_dict)

    if game_seed:
        obs_t, _ = env.reset(seed=game_seed)
        env.action_space.seed(game_seed)
        np.random.seed(game_seed)
    else:
        obs_t, _ = env.reset()

    t_step = 0
    total_game_reward = 0
    done = False

    while not done:
        # Elegir una acción utilizando el agente y el epsilon recibido:
        action_t = agent.get_action(state=obs_t, epsilon=eps)

        # Ejecutar la acción y esperar la respuesta del entorno
        obs_t_plus_one, reward_t_plus_one, terminated, truncated, _ = env.step(action_t)
        done = terminated or truncated

        # Actualizar variables
        total_game_reward += reward_t_plus_one
        t_step += 1
        obs_t = obs_t_plus_one

    return total_game_reward, t_step


def play_games_using_agent(environment_dict: dict, agent, n_games: int, eps: float = 0.01, games_seed: int = 0) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Juega 'n_games' partidas utilizando el agente 'agent'.

    Devuelve el vector de recompensas y el vector de duración de partidas.
    """
    # Jugamos los episodios indicados y registramos el resultado de cada:
    game_rewards = np.zeros(n_games)
    game_steps = np.zeros(n_games)
    for game_n in tqdm.tqdm(range(n_games), desc="Progress playing games"):
        game_reward, steps = play_game_using_agent(
            env_dict=environment_dict,
            agent=agent, eps=eps,
            game_seed=games_seed + game_n
        )
        game_rewards[game_n] = game_reward
        game_steps[game_n] = steps

    return game_rewards, game_steps
