from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from stable_baselines3 import PPO
import uvicorn

# -----------------------
# 설정
# -----------------------
MODEL_PATH = "runs/arm_insert/20251207-000237/ppo_arm_insert_final"  # 필요시 경로 수정

# -----------------------
# FastAPI 앱 생성
# -----------------------
app = FastAPI(title="Arm Insert RL Inference Server")

# 🔥 CORS 설정: 브라우저(프론트)에서 오는 요청 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=[  # 개발용: 로컬 프론트엔드 주소들
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",  # 필요시 위 주소들만 남겨도 됨
    ],
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, OPTIONS 등 모두 허용
    allow_headers=["*"],
)

# -----------------------
# 요청/응답 스키마
# -----------------------
class ObsRequest(BaseModel):
    # TCP 기준 Socket 상대 위치 (dx, dy, dz) [m]
    pos: List[float]
    # TCP ↔ Socket 상대 회전 각도 (rad)
    ori_err: float


class ActionResponse(BaseModel):
    # Δx, Δy, Δz (env에서 [-1,1] 범위; JS에서 action_scale로 곱해서 씀)
    action: List[float]


# -----------------------
# 모델 로드
# -----------------------
print(f"[RL Inference] Loading PPO model from: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)
print("[RL Inference] Model loaded.")


# -----------------------
# 엔드포인트
# -----------------------
@app.post("/predict", response_model=ActionResponse)
async def predict(req: ObsRequest):
    """
    입력: TCP->Socket 상대 위치 pos = [dx, dy, dz] (m)
    출력: PPO가 예측한 action = [ax, ay, az] ([-1,1] 범위)
    """
    obs = np.array(
        [req.pos[0], req.pos[1], req.pos[2], req.ori_err],
        dtype=np.float32
    )

    # SB3의 predict 사용 (deterministic=True: 탐험 없이 추론만)
    action, _ = model.predict(obs, deterministic=True)

    # numpy → python list
    action_list = action.astype(float).tolist()

    return ActionResponse(action=action_list)


if __name__ == "__main__":
    # python arm_insert_infer_server.py 로 실행하면 여기서 uvicorn이 뜸
    uvicorn.run(app, host="0.0.0.0", port=8000)