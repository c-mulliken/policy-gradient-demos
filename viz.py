import os
import gym
import imageio
import torch

def viz_episode(policy, env_name="CartPole-v1", filename="policy.gif", fps=30, mode="gif", device="cpu"):
    """
    Run one episode of `env_name` with `policy` and save a gif or video.
    
    policy: a torch.nn.Module with a .get_action(state) method.
    env_name: the Gym environment.
    filename: output file (should end in .gif for mode="gif", .mp4 for mode="mp4").
    fps: frames per second in the output.
    mode: "gif" or "mp4"
    device: torch device for the policy.
    """
    # 1) Create the env in rgb_array mode
    #    (newer Gym API: pass render_mode to constructor)
    env = gym.make(env_name, render_mode="rgb_array")
    
    # 2) Run a single episode, collecting frames
    frames = []
    state, _ = env.reset()
    done = False
    
    while not done:
        # render returns an RGB array
        frame = env.render()
        frames.append(frame)
        
        # get an action from the policy
        state_tensor = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            # assume get_action returns (action, log_prob)
            action, _ = policy.get_action(state_tensor, device)
        # step once
        step = env.step(action)
        # handle new API: (obs, reward, terminated, truncated, info)
        if len(step) == 5:
            next_state, reward, term, trunc, _ = step
            done = term or trunc
        else:
            next_state, reward, done, _ = step
        
        state = next_state
    
    env.close()
    
    # 3) Write out
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if mode == "gif":
        # imageio will infer GIF from extension
        imageio.mimsave(filename, frames, fps=fps)
    elif mode == "mp4":
        # use ffmpeg via imageio
        writer = imageio.get_writer(filename, fps=fps, codec="libx264")
        for f in frames:
            writer.append_data(f)
        writer.close()
    else:
        raise ValueError("mode must be 'gif' or 'mp4'")
    
    print(f"Saved recording to {filename}")
