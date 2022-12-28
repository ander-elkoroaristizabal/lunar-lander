import os

import PIL.ImageDraw as ImageDraw
import imageio
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
    imageio.mimwrite(os.path.join('./videos/', 'random_agent_space_invader_usuario.gif'), frames, fps=60)


def plot_rewards(training_rewards, mean_training_rewards, reward_threshold: float, save_file_name: str = None):
    plt.figure(figsize=(12, 8))
    plt.plot(training_rewards, label='Rewards')
    plt.plot(mean_training_rewards, label='Mean Rewards')
    plt.axhline(reward_threshold, color='r', label="Reward threshold")
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend(loc="upper left")
    plt.show()
    if save_file_name:
        plt.savefig(save_file_name)
