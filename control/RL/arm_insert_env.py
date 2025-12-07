# ============================
# ArmInsertEnv OBS 수정본
# obs = [dx, dy, dz, ori_err]
# ============================
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ArmInsertEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 관측 공간: dx, dy, dz, ori_err (라디안)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        # 행동 공간: Δx, Δy, Δz (PPO가 출력)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # 초기 상태 저장용
        self.reset()

    def _compute_orientation_error(self, q_rel):
        # q_rel: TCP 기준 Socket quaternion [x, y, z, w]
        w = np.clip(q_rel[3], -1.0, 1.0)
        ori_err = 2 * math.acos(w)   # rad (0 ~ π)
        return ori_err

    def _get_obs(self):
        dx, dy, dz = self.rel_pos   # TCP → Socket 상대 위치
        ori_err = self._compute_orientation_error(self.rel_quat)
        return np.array([dx, dy, dz, ori_err], dtype=np.float32)

    def reset(self, *, seed: int = None, options: dict = None):
        """
        Gym 스타일 reset(seed=..., options=...)을 지원하면서,
        구버전 Gym처럼 obs만 반환하도록 구현한다.
        (SB3의 Monitor/VecEnv가 obs만 기대하므로 info는 반환하지 않음)
        """
        # Gym.Env의 기본 reset 호출 (난수 시드 설정용)
        super().reset(seed=seed)

        # 랜덤 초기 위치 (예시)
        self.rel_pos = np.array([
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-0.12, -0.08)
        ], dtype=np.float32)

        # 초기 상대 쿼터니언 (단위 quaternion)
        self.rel_quat = np.array([0, 0, 0, 1], dtype=np.float32)

        return self._get_obs()

    def step(self, action):
        # action: [-1,1] → Δx,Δy,Δz
        dx = action[0] * 0.005
        dy = action[1] * 0.005
        dz = action[2] * 0.005

        # 위치 업데이트
        self.rel_pos = self.rel_pos + np.array([dx, dy, dz], dtype=np.float32)

        # 보상
        dist = np.linalg.norm(self.rel_pos)
        ori_err = self._compute_orientation_error(self.rel_quat)

        reward = -dist - 0.1 * ori_err

        done = dist < 0.01 and ori_err < (10 * np.pi / 180)

        return self._get_obs(), reward, done, {}