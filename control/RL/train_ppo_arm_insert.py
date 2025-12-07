# control/RL/train_ppo_arm_insert.py
import os
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from arm_insert_env import ArmInsertEnv


def make_env():
    # SB3는 gym.Env 인스턴스를 직접 써도 되고, 
    # vec-env 쓸 거면 래핑해도 된다. 여기선 단순 버전.
    return ArmInsertEnv()


def main():
    logdir = os.path.join("runs", "arm_insert", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)

    env = make_env()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logdir,
        gamma=0.99,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
    )

    # 체크포인트 저장 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=os.path.join(logdir, "checkpoints"),
        name_prefix="ppo_arm_insert",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    total_timesteps = 200_000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # 최종 모델 저장
    save_path = os.path.join(logdir, "ppo_arm_insert_final")
    model.save(save_path)
    print(f"[train_ppo_arm_insert] saved model to {save_path}")


if __name__ == "__main__":
    main()