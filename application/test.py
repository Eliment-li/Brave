import os

import imageio
import numpy as np

from application.q_learning_mountain_car import discretize_state

Q_table = np.load('q_table.npy')
print(f"Q-table size: {Q_table.shape}")
print("Q-table after training:")
print(Q_table)


def play_game_with_frames(env, Q_table, max_steps=1000):
    output_directory = "frames"
    os.makedirs(output_directory, exist_ok=True)
    state = env.reset()
    state = discretize_state(state, state_bins)
    print(f"Initial state: {state}")
    episode_frames = []
    total_reward = 0
    for step in range(max_steps):
        frame = env.render()
        if frame is not None:
            episode_frames.append(frame)
        action = np.argmax(Q_table[state])  # select action using q-table
        next_state, reward, done, _ = env.step(action)[:4]
        state = discretize_state(next_state, state_bins)
        total_reward += reward
        if done:
            break
    print(f"Total Reward: {total_reward}")
    for i, frame in enumerate(episode_frames): # save frames for this episode
        image_path = os.path.join(output_directory, f"frame_{i:03d}.png")
        imageio.imwrite(image_path, (frame * 255).astype(np.uint8))
    env.close()

state_bins = [np.linspace(-1.2, 0.6, 20), np.linspace(-0.07, 0.07, 20)] # assuming we have the state_bin from training
import gymnasium as gym
env = gym.make('MountainCar-v0', render_mode="rgb_array")
initial_state = env.reset()
print(f"Initial state before the loop: {initial_state}")
initial_state = discretize_state(initial_state, state_bins)

# playing the game with the trained q-table for 1 episode, save frames, and print total reward
play_game_with_frames(env, Q_table)
# useing ffmpeg to create a video from saved frames
video_filename = "cassietvid.mp4"
os.system(f"ffmpeg -framerate 30 -pattern_type glob -i 'frames/*.png' -c:v libx264 -pix_fmt yuv420p {video_filename}")