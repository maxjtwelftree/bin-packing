# rl/callbacks.py

from stable_baselines3.common.callbacks import BaseCallback

class RenderOnEpisodeEndCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RenderOnEpisodeEndCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Check if the episode has ended
        done_array = self.locals.get('dones')
        if done_array is not None and any(done_array):
            # Get indices of environments where episodes have ended
            done_indices = [i for i, done in enumerate(done_array) if done]
            for idx in done_indices:
                # Render the final state of each environment that finished
                self.training_env.env_method('render', indices=idx)
        return True

class BestScoreCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(BestScoreCallback, self).__init__(verbose)
        self.best_score = float('-inf')
        self.best_state = None

    def _on_step(self) -> bool:
        # Check if the episode has ended
        done_array = self.locals.get('dones')
        if done_array is not None and any(done_array):
            # Get indices of environments where episodes have ended
            done_indices = [i for i, done in enumerate(done_array) if done]
            # Retrieve 'placed_boxes' from all environments
            placed_boxes_list = self.training_env.get_attr('placed_boxes')
            for idx in done_indices:
                info = self.locals['infos'][idx]
                episode_info = info.get('episode')
                if episode_info:
                    total_volume_packed = episode_info.get('total_volume_packed', 0)
                    num_boxes_used = episode_info.get('num_boxes_used', 0)
                    # Avoid division by zero
                    if num_boxes_used > 0:
                        score = total_volume_packed / num_boxes_used
                    else:
                        score = 0
                    if score > self.best_score:
                        self.best_score = score
                        # Save the best state (placed_boxes)
                        env_placed_boxes = placed_boxes_list[idx]
                        self.best_state = env_placed_boxes.copy()
                        if self.verbose > 0:
                            print(f"New best score: {self.best_score}")
        return True
