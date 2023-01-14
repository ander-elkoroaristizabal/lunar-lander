import PIL.ImageDraw as ImageDraw
import gym
import imageio
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

ACTIONS_DICT = {
    0: 'None',
    1: 'Left engine',
    2: 'Main engine',
    3: 'Right engine'
}


def render_random_agent_episode(env: gym.Env):
    if env.render_mode != 'human':
        raise ValueError('Env render_mode needs to be "human".')
    env.reset()
    total_reward = 0
    for timestep in range(env.spec.max_episode_steps):
        env.render()
        action = env.action_space.sample()
        _, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            print(f"Episode finished after {timestep + 1} time steps "
                  f"with reward {round(total_reward)}.")
            break


def _label_with_text(frame, state, action: str, reward: float, total_reward: float = None):
    """

    Args:
        frame: estado de un entorno GYM.
        action: acción tomada.
        reward: recompensa recibida.

    Returns:

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


# Método que permite crear un gif con la evolución de una partida dado un entorno GYM.
def save_random_agent_gif(env):
    frames = []
    env.reset()
    tr = 0
    ###########################################
    # Jugar una partida aleatoria:
    max_steps = env.spec.max_episode_steps
    for timestep in range(max_steps):
        action = env.action_space.sample()
        frame = env.render()
        obs, reward, done, _, _ = env.step(action)
        tr += reward
        total_reward = round(tr, 2) if (done or (timestep == max_steps - 1)) else None
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
    imageio.mimwrite('random_agent.gif', frames, fps=60)


def plot_rewards(training_rewards, mean_training_rewards, reward_threshold: float,
                 title: str, save_file_name: str = None):
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


def plot_losses(training_losses, title: str, save_file_name: str = None):
    plt.figure(figsize=(12, 8))
    plt.plot(training_losses, label='Real Training loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    if save_file_name:
        plt.savefig(save_file_name)
    plt.show()


def plot_evaluation_rewards(rewards: np.ndarray, reward_threshold: float,
                            title: str, save_file_name: str = None):
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


def save_agent_gif(env: gym.Env, ag, save_file_name: str):
    """

    Args:
        env: entorno GYM
        ag: agente entrenado
        save_file_name: nombre del fichero
    """
    frames = []
    env.reset()
    obs, _ = env.reset()
    tr = 0
    max_steps = env.spec.max_episode_steps
    for timestep in range(max_steps):
        action = ag.get_action(state=obs)
        frame = env.render()
        obs, reward, done, _, _ = env.step(action)
        tr += reward
        total_reward = round(tr, 2) if (done or (timestep == max_steps - 1)) else None
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
    imageio.mimwrite(save_file_name, frames, fps=60)
