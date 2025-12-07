// frontend/src/control/rlAgent.js
import * as THREE from 'three';

export class RLAgent {
  constructor({ controller, input, plugFrame }) {
    this.controller = controller;
    this.input = input;
    this.plugFrame = plugFrame;   // TCP 프레임 (빨간 점 있는 그 노드)

    this.mode = 'IK'; // 'IK' | 'RL'

    this.distTarget    = 0.10;   // [m] 1차 목표 거리 (10cm)
    this.distTolerance = 0.0001; // [m] 1차 목표 거리 허용 오차 (0.1mm)
    this.successRadius = 0.01;   // [m] RL에서 성공으로 보는 거리 (1cm 이내)
    this.actionScale   = 0.002;
    this.lastDist      = null;                    // 디버깅용으로 마지막 거리 기록

    // 각도 제약: TCP→Socket 상대 회전이 이 값(라디안) 이하일 때만 RL로 전환
    // 예: 10도 ≈ 0.1745 rad
    this.oriToleranceRad = Math.PI;
  }

  updatePhase({ relPose }) {
    if (!relPose || !relPose.position) return;

    const p = relPose.position;
    const dist = Math.hypot(p.x, p.y, p.z);
    this.lastDist = dist;

    // 상대 회전 오차 (TCP 기준 Socket 상대 회전) 계산
    // relPose.quaternion이 주어졌다고 가정하고, 회전 각(0~π)을 구함.
    let oriErr = 0;
    if (relPose.quaternion) {
      const q = relPose.quaternion;
      // 단위 quaternion이라고 가정하고, w를 이용해 회전 각도 추정
      const w = THREE.MathUtils.clamp(q.w, -1, 1);
      oriErr = 2 * Math.acos(w); // [rad], 0 ~ π
    }

    // 1) IK 모드일 때:
    //    - 거리: distTarget(≈0.10m) 근처(±distTolerance)
    //    - 각도: oriToleranceRad(예: 10도) 이하
    //    두 조건을 모두 만족할 때만 RL로 전환
    if (this.mode === 'IK') {
      const diff = Math.abs(dist - this.distTarget);
      const distOk = diff < this.distTolerance;
      const oriOk = oriErr < this.oriToleranceRad;

      if (distOk && oriOk) {
        console.log(
          `[RLAgent] IK phase complete: dist=${dist.toFixed(4)} (|dist - ${
            this.distTarget
          }|=${diff.toFixed(4)} < tol=${this.distTolerance}), ` +
          `oriErr=${(oriErr * 180 / Math.PI).toFixed(2)}deg < ${(this.oriToleranceRad * 180 / Math.PI).toFixed(
            2
          )}deg. → RL 모드 전환`
        );
        this._switchToRL();
      } else {
        // 아직 1차 목표 근처가 아니면 RL로 넘기지 않음 (IK 유지)
        console.log('[RLAgent] IK phase: dist/ori not in RL window',
          'dist=', dist.toFixed(4),
          'oriErr(deg)=', (oriErr * 180 / Math.PI).toFixed(2)
        );
      }
      return;
    }

    // 2) RL 모드일 때: 충분히 가까워지면 DONE 상태로 전환
    if (this.mode === 'RL') {
      if (dist < this.successRadius) {
        console.log(
          `[RLAgent] RL success: dist=${dist.toFixed(4)} < ${this.successRadius}. → DONE 모드 전환`
        );
        this.mode = 'DONE';
        // IK는 계속 OFF 상태로 두고, 이후 handleAction은 더 이상 적용하지 않음
      }
    }
  }

  _switchToRL() {
    this.mode = 'RL';
    if (this.input) this.input.IK_ON = false;
  }

  reset() {
    this.mode = 'IK';
    this.lastDist = null;
    if (this.input) this.input.IK_ON = true;
  }

  /**
   * PPO에서 온 action: [ax, ay, az] ([-1,1] 범위)
   * → TCP 로컬 좌표계 기준 Δx,Δy,Δz [m]로 해석
   * → world로 변환 후, IK 한 스텝
   */
  handleAction(action) {
    if (this.mode !== 'RL') return; // IK 또는 DONE 모드일 때는 액션 무시
    if (!Array.isArray(action) || action.length < 3) return;
    if (!this.plugFrame) {
      console.warn('[RLAgent] plugFrame 없음');
      return;
    }

    const [ax, ay, az] = action;
    const dx = -ax * this.actionScale;
    const dy = -ay * this.actionScale;
    const dz = -az * this.actionScale;

    console.log(
      '[RLAgent] action =',
      action,
      'deltaLocal(m) =',
      { dx, dy, dz },
      'mode =',
      this.mode
    );

    // 1) 현재 TCP 포즈 (world 기준)
    const tcpPos = new THREE.Vector3();
    const tcpQuat = new THREE.Quaternion();
    this.plugFrame.getWorldPosition(tcpPos);
    this.plugFrame.getWorldQuaternion(tcpQuat);

    // 2) TCP 로컬 Δ → world Δ
    const deltaLocal = new THREE.Vector3(dx, dy, dz);
    const deltaWorld = deltaLocal.clone().applyQuaternion(tcpQuat);

    // 3) 새 타깃 포즈 = 현재 + Δ, 방향은 그대로
    const targetPose = new THREE.Object3D();
    targetPose.position.copy(tcpPos.clone().add(deltaWorld));
    targetPose.quaternion.copy(tcpQuat);

    // 4) Jacobian IK 한 스텝
    this.controller.applyIK(targetPose);
  }
}