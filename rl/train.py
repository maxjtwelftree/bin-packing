# train.py
from agent import model, env
from callbacks import RenderOnEpisodeEndCallback, BestScoreCallback
from stable_baselines3.common.callbacks import CallbackList

def train_agent():
    total_timesteps = 100000
    render_callback = RenderOnEpisodeEndCallback(verbose=1)
    best_score_callback = BestScoreCallback(verbose=1)
    callback = CallbackList([render_callback, best_score_callback])
    
    # Increase verbosity to track progress
    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10)
    model.save("ppo_bin_packing_model")
    
    print(f"Best score achieved: {best_score_callback.best_score}")
    # Save the best packing state
    import numpy as np
    np.save('best_packing_state.npy', best_score_callback.best_state, allow_pickle=True)

if __name__ == "__main__":
    train_agent()
