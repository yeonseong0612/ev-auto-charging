from stable_baselines3 import PPO
import numpy as np

from arm_insert_env import ArmInsertEnv


MODEL_PATH = "runs/arm_insert/20251207-013946/ppo_arm_insert_final"  # 네 로그에 찍힌 경로로 수정


def main(n_episodes: int = 20):
    env = ArmInsertEnv()
    model = PPO.load(MODEL_PATH)

    success_count = 0
    collision_count = 0
    total_steps = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_rew = 0.0
        step = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_rew += reward
            step += 1

        total_steps += step

        dist = info.get("dist", np.linalg.norm(obs))
        flag = "none"
        if info.get("success", False):
            success_count += 1
            flag = "success"
        if info.get("collision", False):
            collision_count += 1
            flag = "collision"
        if info.get("timeout", False):
            flag = "timeout"

        print(
            f"Episode {ep:02d}: steps={step}, final_dist={dist:.4f}, "
            f"reward={ep_rew:.3f}, result={flag}"
        )

    print("===================================")
    print(f"Eval episodes: {n_episodes}")
    print(f"Success:   {success_count} / {n_episodes}")
    print(f"Collision: {collision_count} / {n_episodes}")
    print(f"Avg steps: {total_steps / n_episodes:.2f}")


if __name__ == "__main__":
    main()