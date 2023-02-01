# helper functions for visualizing the RL agent

# import dependencies
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, clear_output

# wrapper for DQN agent
class DQNWrapper:

    def __init__(self, agent, n_actions=4, eps=0.01):
        self.agent = agent
        self.n_actions = n_actions
        self.eps = eps

    def act(self, observation):
        return self.agent.get_action(observation, self.n_actions, self.eps).cpu().numpy()[0]

# generating simulation for an agent
def simulate(agent, env, max_frames=500):
    
    # Reset the environment.
    state = env.reset()[0]
    
    # Collect the first observation.
    img = env.render()
    
    # Create a frame buffer capable of holding max_frames frames of size of img.
    frame_buffer = torch.zeros(size=img.shape, dtype=torch.uint8).unsqueeze(dim=0).repeat([max_frames, 1, 1, 1])
    frame_buffer[0] = torch.tensor(img.copy())
    frame = 1
    
    # Simulate the agent.
    done = False
    truncated = False
    rewards = 0
    while not (done or truncated):
        
        # Step the environment.
        action = agent.act(state)
        state, reward, done, truncated, _ = env.step(action)
        rewards += reward
        
        # Render image if it can be stored in the frame buffer.
        if frame < max_frames:
            frame_buffer[frame] = torch.tensor(env.render().copy())
            frame += 1

    return frame_buffer[:frame], rewards

# animate the collected frames
def animate(frame_buffer):

    # Create figure with a single plot, initialized to the first frame.
    fig, ax = plt.subplots(1, 1, figsize=(9,3.25), dpi=90)
    img = ax.imshow(frame_buffer[0])
    plt.tight_layout()

    # Local-scope animate function for reading images from the frame buffer into img.
    def _animate_frame(i):
        img.set_data(frame_buffer[i])
        return (img, )

    # Construct an animation to visualize the agent, interval of 1000 mS / 60 FPS.
    anim = animation.FuncAnimation(fig, _animate_frame, frames=frame_buffer.shape[0], interval=1000/60)
    html = HTML(anim.to_html5_video())

    # Done with the figure, close it to release memory (and avoid plotting it in cell).
    plt.close()
    
    return html

# plot training progress by frames
def plot(frame_idx, rewards, losses):
    """
    Plot evolution of rewards and losses during training
    Args:
         rewards (list): Cummulative rewards for episodes seen so far
         losses (list): Prediction error at each training step
    
    """
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('Steps %s.\nCummulative reward last 10 episodes: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.ylabel("Avg. cummulative reward")
    plt.xlabel("No. of steps")
    plt.subplot(132)
    plt.title('MSE Loss')
    plt.ylabel("Avg. cummulative TD-loss")
    plt.xlabel("No. of steps")
    plt.plot(losses)
    plt.show()

# plot training progress by episodes    
def plot_ep(ep_idx, rewards, losses):
    """
    Plot evolution of rewards and losses during training
    Args:
         rewards (list): Cummulative rewards for episodes seen so far
         losses (list): Prediction error at each training step
    
    """
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('Episodes %s.\nCummulative reward last 10 episodes: %s' % (ep_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.ylabel("Avg. cummulative reward")
    plt.xlabel("No. of episodes")
    plt.subplot(132)
    plt.title('MSE Loss')
    plt.ylabel("Avg. cummulative TD-loss")
    plt.xlabel("No. of episodes")
    plt.plot(losses)
    plt.show()
