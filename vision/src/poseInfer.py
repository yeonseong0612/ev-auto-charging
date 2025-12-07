"""
Pose inference helper/CLI

용도
- 실시간: 별도 프로세스를 띄워 `--stdin-loop`로 `{mode:"pose", image:<b64>}` JSON 라인을 받아 좌표를 반환할 수 있음. 키 입력(P 버튼)이나 RL 쪽에서 워커 프로세스에 프레임을 보내는 방식.
- 임베드: 파이썬 코드에서 `load_pose_model`, `infer_pose`를 import해 같은 프로세스에서 직접 호출할 수 있음.
- 단일 테스트: `--test`로 샘플 이미지를 바로 추론해 JSON을 출력.

CLI 사용법
- 기본 실행: `python poseInfer.py --weights <ckpt>` (stdin 단일 b64를 읽어 `{pred:[...]}` 출력)
- 샘플 테스트: `python poseInfer.py --test [--weights <ckpt>]`
- 스트림 모드: `python poseInfer.py --stdin-loop [--weights <ckpt>]`
  - 입력 JSON 예: `{"mode":"pose","image":"<base64>"}` (`mode` 생략 시 기본 pose)
  - 출력 JSON: `{"pred": [...]}` 또는 지원하지 않는 모드면 `{"error":"unsupported mode","mode":...}`

가중치 기본값은 `vision/SEGU/checkpoints/best.pth` 상대 경로를 사용. 좌표는 학습 시 스케일(POS_SCALE) 복원 후 반환.
"""

import argparse
import base64
import io
import json
import os
import sys

import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # vision/
DEFAULT_WEIGHTS = ROOT / "SEGU" / "checkpoints" / "best.pth"
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "SEGU" / "model"))  # for mobilenetv3 import
from SEGU.model.suPoseModel import PoseRegressor  # noqa: E402

# 학습 시 pos_scale(예: 100)로 좌표를 스케일했다면 추론 시 되돌림
POS_SCALE = float(os.getenv("POSE_POS_SCALE", "100.0"))


def decode_image(b64str: str) -> Image.Image:
    buf = base64.b64decode(b64str)
    return Image.open(io.BytesIO(buf)).convert("RGB")


def load_model(weights_path: str, device: str = "cpu"):
    checkpoint = torch.load(weights_path, map_location=device)
    model = PoseRegressor(backbone="mobilenet_v3", pretrained_path=None)
    # 다양한 저장 형식 지원: model_state / model / state_dict / raw
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state = checkpoint["model_state"]
        elif "model" in checkpoint:
            state = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        state = checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        sys.stderr.write(f"[poseInfer] loaded with missing={len(missing)}, unexpected={len(unexpected)}\n")
    model.to(device)
    model.eval()
    return model


def preprocess(img: Image.Image):
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tfm(img).unsqueeze(0)


def load_pose_model(weights_path: str | Path | None = None, device: str | None = None):
    """Load pose regressor with optional weights path/device."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = str(weights_path or DEFAULT_WEIGHTS)
    return load_model(weights_path, device), device


def infer_pose(model, img: Image.Image, device: str | None = None):
    """Run pose inference on a PIL image and return list of floats."""
    device = device or next(model.parameters()).device
    tensor = preprocess(img).to(device)
    with torch.no_grad():
        pred = model(tensor)
    arr = pred.squeeze(0).detach().cpu().numpy()
    arr[:3] = arr[:3] / POS_SCALE  # 좌표 스케일 복원 (m 단위)
    return arr.tolist()


def infer_pose_b64(model, b64_image: str, device: str | None = None):
    img = decode_image(b64_image)
    return infer_pose(model, img, device)


def run_once(model, device, b64_image: str):
    return infer_pose_b64(model, b64_image, device)


def main():
    parser = argparse.ArgumentParser(description="Pose regression inference worker")
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to best.pth checkpoint",
    )
    parser.add_argument("--test", action="store_true", help="Run inference on bundled sample image and exit")
    parser.add_argument("--stdin-loop", action="store_true", help="Keep process alive and read JSON lines")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)

    if args.test:
        sample_path = ROOT / "SEGU" / "datasets" / "sample" / "m_251207_183158454_m0d115_0d014_m0d334_0d726_0d032_m0d013_0d687_0d354_1.png"
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample image not found: {sample_path}")
        img = Image.open(sample_path).convert("RGB")
        pred = infer_pose(model, img, device)
        sys.stdout.write(json.dumps({"sample_path": str(sample_path), "pred": pred}) + "\n")
        return

    if args.stdin_loop:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                mode = payload.get("mode", "pose")
                if mode != "pose":
                    msg = {"error": "unsupported mode", "mode": mode}
                    sys.stdout.write(json.dumps(msg) + "\n")
                    sys.stdout.flush()
                    continue
                b64 = payload.get("image") or ""
                pred = run_once(model, device, b64)
                sys.stdout.write(json.dumps({"pred": pred}) + "\n")
                sys.stdout.flush()
            except Exception as e:  # pragma: no cover
                sys.stderr.write(f"[poseInfer] error: {e}\n")
                sys.stderr.flush()
        return

    # single run (stdin one image)
    b64 = sys.stdin.read().strip()
    if not b64:
        raise RuntimeError("No image provided on stdin")
    pred = run_once(model, device, b64)
    sys.stdout.write(json.dumps({"pred": pred}))


if __name__ == "__main__":
    main()
