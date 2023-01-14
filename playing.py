import numpy as np
import tqdm


def play_game(environment, policy):
    """
    Ejecución de un episodio del entorno
    en el que el agente sigue la política 'policy'.

    Devuelve la recompensa del episodio y el número de pasos.
    """

    # Inicializamos el entorno:
    obs_t, _ = environment.reset()
    t_step = 0
    total_game_reward = 0
    episode_is_done = False

    while (not episode_is_done) and (t_step < environment.spec.max_episode_steps):
        # Elegir una acción:
        action_t = policy(observation=obs_t)

        # Ejecutar la acción y esperar la respuesta del entorno
        obs_t_plus_one, reward_t_plus_one, episode_is_done, _, _ = environment.step(action_t)

        # Actualizar variables
        total_game_reward += reward_t_plus_one
        t_step += 1
        obs_t = obs_t_plus_one

    return total_game_reward, t_step


def play_games(environment, policy, n_games: int):
    """
    Juega 'n_games' partidas siguiendo la política 'policy'.

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


def play_game_using_agent(environment, agent):
    """
    Ejecución de un episodio del entorno
    utilizando el agente 'agent'.

    Devuelve la recompensa total del episodio y el número de pasos necesitados.
    """

    # Inicializamos el entorno:
    obs_t, _ = environment.reset()
    t_step = 0
    total_game_reward = 0
    episode_is_done = False

    while (not episode_is_done) and (t_step < environment.spec.max_episode_steps):
        # Elegir una acción utilizando el agente y un epsilon de 0.01:
        action_t = agent.get_action(state=obs_t, epsilon=0.01)

        # Ejecutar la acción y esperar la respuesta del entorno
        obs_t_plus_one, reward_t_plus_one, episode_is_done, _, _ = environment.step(action_t)

        # Actualizar variables
        total_game_reward += reward_t_plus_one
        t_step += 1
        obs_t = obs_t_plus_one

    return total_game_reward, t_step


def play_games_using_agent(environment, agent, n_games: int):
    """
    Juega 'n_games' partidas utilizando el agente 'agent'.

    Devuelve el vector de recompensas y el vector de duración de partidas.
    """
    # Jugamos los episodios indicados y registramos el resultado de cada:
    game_rewards = np.zeros(n_games)
    game_steps = np.zeros(n_games)
    for game in tqdm.tqdm(range(n_games), desc="Progress playing games"):
        game_reward, steps = play_game_using_agent(environment=environment, agent=agent)
        game_rewards[game] = game_reward
        game_steps[game] = steps

    return game_rewards, game_steps
