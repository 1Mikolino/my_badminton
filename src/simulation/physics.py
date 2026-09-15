import numpy as np
def drag_coefficient(v,param):
    #**变速阻力系数**：低速阻力大、高速阻力小
    return param["cd_low"]+(param["cd_high"]-param["cd_low"])*np.exp(-1*v/param["transition_speed"])
def acceleration(state,param):
    #输入球当前状态 + 物理参数，输出加速度（重力 + 阻力）。阻力系数按上面的公式随 `v` 变化。
    state = np.asarray(state, dtype=float)
    v=state[3:6]
    speed=np.linalg.norm(v)#速度大小 |v|
    a=np.array([0.0,0.0,-param["gravity"]])
    A=np.pi*param["radius"]**2#面积
    #**二次阻力**：`F_drag = ½ ρ C_d A v²`，方向与速度相反（`A` 是截面积，`ρ` 是空气密度）。
    F_D=-0.5*param["air_density"]*drag_coefficient(speed,param)*A*speed*v
    a=a+F_D/param["mass"]
    return a
def _deriv(state, param):
    return np.concatenate([state[3:6], acceleration(state, param)])   # [速度, 加速度]

def step(state, dt, param):
    k1 = _deriv(state, param)
    k2 = _deriv(state + 0.5 * dt * k1, param)
    k3 = _deriv(state + 0.5 * dt * k2, param)
    k4 = _deriv(state + dt * k3, param)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rollout(pos0, vel0, param):
    """从起始位置/速度外推轨迹，返回 (N, 3) 位置点数组。"""
    sample_dt = param["rollout_dt"]          # 0.01
    substeps  = param["rollout_substeps"]    # 2
    duration  = param["rollout_horizon"]     # 2.0
    ground_z  = param["ground_z"]            # 0.0

    h = sample_dt / substeps                 # 0.005
    state = np.concatenate([np.asarray(pos0, float),
                            np.asarray(vel0, float)])
    traj = [state[:3].copy()]
    t = 0.0
    while t < duration:
        for _ in range(substeps):
            state = step(state, h, param)
        t += sample_dt
        traj.append(state[:3].copy())
        if state[2] < ground_z:              # 规则在代码，阈值在配置
            z1, z2 = traj[-2][2], traj[-1][2]
            alpha = (ground_z - z1) / (z2 - z1)
            traj[-1] = traj[-2] + alpha * (traj[-1] - traj[-2])
            break
    return np.array(traj)
