import PIL.ImageDraw as ImageDraw
import gym
import imageio
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def _label_with_text(frame, action, reward):
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
    drawer.text((1, 1),
                f"Uoc Aprendizaje Por Refuerzo. Action={action} and Reward={reward}",
                fill=(255, 255, 255, 128))
    return im


# Método que permite crear un gif con la evolución de una partida dado un entorno GYM.
def save_random_agent_gif(env):
    frames = []
    done = False
    env.reset()
    ###########################################
    # Jugar una partida aleatoria:
    while not done:
        action = env.action_space.sample()
        frame = env.render(mode='rgb_array')
        state, reward, done, _ = env.step(action)
        frames.append(_label_with_text(frame=frame, action=action, reward=int(reward)))
    ##############################################

    env.close()
    imageio.mimwrite('random_agent.gif', frames, fps=60)


def plot_rewards(training_rewards, mean_training_rewards, reward_threshold: float, save_file_name: str = None):
    plt.figure(figsize=(12, 8))
    plt.plot(training_rewards, label='Rewards')
    plt.plot(mean_training_rewards, label='Mean Rewards')
    plt.axhline(reward_threshold, color='r', label="Reward threshold")
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    if save_file_name:
        plt.savefig(save_file_name)
    plt.show()


def plot_losses(training_losses, save_file_name: str = None):
    plt.figure(figsize=(12, 8))
    plt.plot(training_losses, label='Real Training loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.legend()
    if save_file_name:
        plt.savefig(save_file_name)
    plt.show()


def plot_evaluation_rewards(rewards: np.ndarray, reward_threshold: float, save_file_name: str = None):
    plt.figure(figsize=(12, 8))
    plt.plot(rewards, label='Total episode reward')
    r_mean = rewards.mean()
    plt.axhline(y=r_mean, label=f'Mean of episode Rewards ({round(r_mean)})', color='orange')
    r_median = float(np.median(rewards))
    plt.axhline(y=r_median, label=f'Median of episode Rewards ({round(r_median)})', color='purple')
    plt.axhline(y=reward_threshold, label='Reward Threshold', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.ylim(ymin=0)
    plt.legend()
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
    observation, _ = env.reset()
    total_reward = 0
    t = 0
    while True:
        frame = env.render()
        action = ag.get_action(state=observation)
        observation, reward, done, _, _ = env.step(action)
        frames.append(_label_with_text(frame=frame, action=action, reward=reward))
        total_reward += reward
        t = t + 1
        if done:
            break

    env.close()
    imageio.mimwrite(save_file_name, frames, fps=60)
