/*
RobotArm: FK/IK & Joint State
-----------------------------------------------------------------------
역할:
    - GLTF에서 조인트 노드 찾기(별칭 NAME_MAP 사용)
    - 각 조인트 기본 자세 캐시, FK(회전 적용)
    - CCD 기반 IK(목표 점 추종), 엔드 이펙터 조회
    - GLTF 원래 계층 보존

주요 Export:
    - class RobotArm

자주 수정하는 지점:
    - IK 반복/허용오차(solveIK_CCD 파라미터)
    - 조인트 클램프/ 속도 제한 등 정책 추가 지점
*/

import * as THREE from 'three';
import { JOINT_ORDER, JOINT_META, NAME_MAP } from '../config/jointMeta';

function findJointNode(root, logicalName) {
  const aliases = NAME_MAP[logicalName] || [logicalName];
  const matches = [];
  root.traverse((o) => {
    const nm = o.name || '';
    if (aliases.some((a) => nm === a || nm.toLowerCase() === a.toLowerCase())) matches.push(o);
  });
  if (matches.length === 0) return null;
  // prefer non-mesh
  return matches.find((o) => !o.isMesh) || matches[0];
}

function depthFromRoot(node) {
  let d = 0,
    p = node;
  while (p) {
    d++;
    p = p.parent;
  }
  return d;
}

export class RobotArm {
  constructor(meta = JOINT_META, chainOrder = JOINT_ORDER) {
    this.meta = meta;
    this.order = [...chainOrder];
    this.joints = {}; // name -> Object3D
    this.angles = {}; // name -> radians
    this._defaults = {}; // name -> Quaternion
    this.root = null; // GLTF root
    this.eeOverride = null; // optional custom end-effector node
  }

  attach(root) {
    this.root = root;
    // resolve joints by alias map
    for (const n of JOINT_ORDER) this.joints[n] = findJointNode(root, n) || null;

    // chain sorted by scene depth (respect GLTF hierarchy!)
    const available = JOINT_ORDER.filter((n) => !!this.joints[n]).sort(
      (a, b) => depthFromRoot(this.joints[a]) - depthFromRoot(this.joints[b])
    );
    this.order.length = 0;
    this.order.push(...available);

    // cache defaults & reset
    for (const [name, node] of Object.entries(this.joints))
      if (node) this._defaults[name] = node.quaternion.clone();
    for (const n of this.order) this.angles[n] = 0;
    this.applyFK();
  }

  setJointAngle(name, rad) {
    const node = this.joints[name];
    if (!node) return;
    const meta = this.meta[name] || {};
    const clamped = THREE.MathUtils.clamp(rad, meta.min ?? -Infinity, meta.max ?? Infinity);
    this.angles[name] = clamped;
    node.quaternion.copy(this._defaults[name]);
    const axis = (meta.axis ?? new THREE.Vector3(0, 0, 1)).clone().normalize();
    node.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(axis, clamped));
  }

  applyFK() {
    for (const n of this.order) this.setJointAngle(n, this.angles[n] ?? 0);
  }

  setEndEffector(node) {
    this.eeOverride = node || null;
  }

  getEndEffector() {
    if (this.eeOverride) return this.eeOverride;
    for (let i = this.order.length - 1; i >= 0; --i) {
      const n = this.order[i];
      if (this.joints[n]) return this.joints[n];
    }
    return null;
  }

  solveIK_CCD(target, iterations = 10, tol = 1e-3) {
    const ee = this.getEndEffector();
    if (!ee) return;
    const targetWorld =
      target instanceof THREE.Vector3 ? target.clone() : new THREE.Vector3().copy(target.position);
    const tmp = new THREE.Vector3();

    for (let it = 0; it < iterations; it++) {
      let solved = false;
      for (let i = this.order.length - 1; i >= 0; i--) {
        const name = this.order[i],
          j = this.joints[name];
        if (!j) continue;

        const jw = j.getWorldPosition(new THREE.Vector3());
        const ew = ee.getWorldPosition(new THREE.Vector3());
        const v1 = ew.clone().sub(jw);
        const v2 = targetWorld.clone().sub(jw);
        if (v1.lengthSq() < 1e-10 || v2.lengthSq() < 1e-10) continue;
        v1.normalize();
        v2.normalize();

        const localAxis = (this.meta[name]?.axis.clone() ?? new THREE.Vector3(0, 0, 1)).normalize();
        const worldQuat = j.getWorldQuaternion(new THREE.Quaternion());
        const worldAxis = localAxis.applyQuaternion(worldQuat).normalize();

        const cross = new THREE.Vector3().crossVectors(v1, v2);
        const sin = THREE.MathUtils.clamp(cross.dot(worldAxis), -1, 1);
        const cos = THREE.MathUtils.clamp(v1.dot(v2), -1, 1);
        const delta = Math.atan2(sin, cos);

        this.setJointAngle(name, (this.angles[name] ?? 0) + delta);
        this.applyFK();

        const dist = ee.getWorldPosition(tmp).distanceTo(targetWorld);
        if (dist < tol) {
          solved = true;
          break;
        }
      }
      if (solved) break;
    }
  }

  /**
   * 6D Jacobian-based IK (position + orientation).
   *
   * target:
   *  - THREE.Vector3: position only (orientation is kept as current EE orientation)
   *  - Object3D: world position + world quaternion are treated as the desired pose
   *
   * options:
   *  - stepSize: Jacobian transpose gain (default 0.4)
   *  - maxDelta: per-step joint angle limit in radians (default 0.05)
   *  - wPos: weight for position error (default 1.0)
   *  - wOri: weight for orientation error (default 1.0)
   */
  solveIK_Jacobian(target, options = {}) {
    console.log('[IK] Jacobian called, target =', target);
    const ee = this.getEndEffector();
    if (!ee) return;

    // current EE pose (world)
    const eePos = new THREE.Vector3();
    const eeQuat = new THREE.Quaternion();
    ee.getWorldPosition(eePos);
    ee.getWorldQuaternion(eeQuat);

    console.log(
      '[IK] eePos =',
      eePos.toArray().map((v) => v.toFixed(3)),
      'targetPos(local) =',
      target.position ? target.position.toArray().map((v) => v.toFixed(3)) : 'N/A'
    );

    // desired pose
    const targetPos = new THREE.Vector3();
    const targetQuat = eeQuat.clone(); // default: keep current orientation

    if (target instanceof THREE.Vector3) {
      // position only
      targetPos.copy(target);
    } else if (target && typeof target === 'object') {
      if (target.isObject3D) {
        target.getWorldPosition(targetPos);
        target.getWorldQuaternion(targetQuat);
      } else {
        if (target.position instanceof THREE.Vector3) {
          targetPos.copy(target.position);
        }
        if (target.quaternion instanceof THREE.Quaternion) {
          targetQuat.copy(target.quaternion);
        }
      }
    } else {
      // fallback: do nothing if target is invalid
      return;
    }

    // position error: ep = p_target - p_ee
    const posErr = new THREE.Vector3().subVectors(targetPos, eePos);

    // orientation error: q_err = q_target * q_ee^{-1}
    const eeQuatInv = eeQuat.clone().invert();
    const qErr = targetQuat.clone().multiply(eeQuatInv).normalize();

    const axis = new THREE.Vector3(0, 0, 0);
    let angle = 0.0;
    // guard against numerical issues
    const w = THREE.MathUtils.clamp(qErr.w, -1, 1);
    angle = 2 * Math.acos(w);
    const s = Math.sqrt(1 - w * w);
    if (s < 1e-6 || !isFinite(s)) {
      axis.set(0, 0, 0);
      angle = 0.0;
    } else {
      axis.set(qErr.x / s, qErr.y / s, qErr.z / s);
    }
    // orientation error as 3D vector (axis * angle)
    const oriErr = axis.multiplyScalar(angle);
    console.log(
      '[IK] posErr len =', posErr.length().toFixed(4),
      'oriErr len =', oriErr.length().toFixed(4),
      'joints =', this.order.length
    );
    const wPos = options.wPos ?? 1.0;
    const wOri = options.wOri ?? 1.0;

    // 6x1 error vector [position; orientation]
    const e6 = [
      wPos * posErr.x,
      wPos * posErr.y,
      wPos * posErr.z,
      wOri * oriErr.x,
      wOri * oriErr.y,
      wOri * oriErr.z,
    ];

    const stepSize = options.stepSize ?? 0.4;
    const maxDelta = options.maxDelta ?? 0.05;

    // 6xN Jacobian (position + orientation)
    const J = this._computeJacobian6D(eePos);
    const numJoints = this.order.length;

    // dq = stepSize * J^T * e6
    const dq = new Array(numJoints).fill(0);
    for (let j = 0; j < numJoints; j++) {
      let acc = 0;
      for (let r = 0; r < 6; r++) {
        acc += J[r][j] * e6[r];
      }
      dq[j] = stepSize * acc;
    }

    // apply joint updates with per-step clamp
    for (let idx = 0; idx < numJoints; idx++) {
      const name = this.order[idx];
      const joint = this.joints[name];
      if (!joint) continue;

      let delta = dq[idx];
      if (Math.abs(delta) > maxDelta) {
        delta = maxDelta * Math.sign(delta);
      }
      const cur = this.angles[name] ?? 0;

      // ▼ 실제로 움직이려는 조인트만 찍기
      if (Math.abs(delta) > 1e-5) {
        console.log('[IK] joint', name, 'cur =', cur.toFixed(4), 'delta =', delta.toFixed(4));
      }
      this.setJointAngle(name, cur + delta);
    }

    // apply FK once after updating all joints
    this.applyFK();
  }

  /**
   * Compute 6xN Jacobian (position + orientation) for the current pose.
   * For revolute joints:
   *  - Jp_i = z_i × (p_ee - p_i)
   *  - Jo_i = z_i
   * where z_i is the joint axis in world frame.
   */
  _computeJacobian6D(eePosWorld) {
    const rows = 6;
    const cols = this.order.length;
    const J = new Array(rows);
    for (let r = 0; r < rows; r++) {
      J[r] = new Array(cols).fill(0);
    }

    const tmpPos = new THREE.Vector3();
    const tmpQuat = new THREE.Quaternion();
    const zLocalDefault = new THREE.Vector3(0, 0, 1);
    const zWorld = new THREE.Vector3();
    const toEE = new THREE.Vector3();

    for (let c = 0; c < cols; c++) {
      const name = this.order[c];
      const joint = this.joints[name];
      if (!joint) continue;

      joint.getWorldPosition(tmpPos);
      joint.getWorldQuaternion(tmpQuat);

      // local joint axis from meta or default Z
      const localAxis =
        (this.meta[name]?.axis && this.meta[name].axis.clone()) || zLocalDefault.clone();
      localAxis.normalize();

      // world joint axis
      zWorld.copy(localAxis).applyQuaternion(tmpQuat).normalize();

      // vector from joint to EE
      toEE.subVectors(eePosWorld, tmpPos);

      // position part: z × (p_ee - p_i)
      const Jp = new THREE.Vector3().crossVectors(zWorld, toEE);

      J[0][c] = Jp.x;
      J[1][c] = Jp.y;
      J[2][c] = Jp.z;
      J[3][c] = zWorld.x;
      J[4][c] = zWorld.y;
      J[5][c] = zWorld.z;
    }

    return J;
  }
}
