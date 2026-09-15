# -*- coding: utf-8 -*-
"""
M2 · 物理与球动力学 —— 验收自检脚本

用法（在任意目录下都能跑）：
    python tests/check_physics.py
    python tests/check_physics.py --plot     # 额外生成轨迹图 tests/trajectory_M2.png

覆盖 M2 的四条验收标准：
  [1] 给定 [px,py,pz,vx,vy,vz] 和初速，能跑出合理的下落/抛飞轨迹
  [2] 阻力系数随速度变化：低速趋近 cd_high，高速趋近 cd_low
  [3] 轨迹能与「普通斜抛」区分（有阻力落点更近）
  [4] 最小测试：自由下落 z 单调递减、能量变化符合常识

退出码：全部通过 = 0，有失败 = 1
"""

import sys
from pathlib import Path

import numpy as np


# ---------- 锚定项目根：不管从哪个目录运行，都能找到 config/ 和 src/ ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "simulation"))

try:
    sys.stdout.reconfigure(encoding="utf-8")      # 避免 Windows 控制台中文乱码
except Exception:
    pass

from config import load_yaml          # M1 的统一配置入口
import physics                        # M2 的物理模块


# ============================ 迷你测试框架 ============================
_results = []


def check(name, ok, detail=""):
    _results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")
    return ok


def section(title):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


# ============================ 工具函数 ============================
def integrate(state0, param, dt=0.005, max_steps=5000, ground_z=0.0):
    """用 physics.step 持续积分，保留每一步的完整状态（用于能量检查）。"""
    s = np.asarray(state0, dtype=float).copy()
    states = [s.copy()]
    for _ in range(max_steps):
        s = physics.step(s, dt, param)
        states.append(s.copy())
        if s[2] < ground_z:
            break
    return np.array(states)


def energy(states, param):
    """机械能 E = ½m|v|² + m·g·z"""
    v2 = np.sum(states[:, 3:6] ** 2, axis=1)
    return 0.5 * param["mass"] * v2 + param["mass"] * param["gravity"] * states[:, 2]


# ============================ 主流程 ============================
def main():
    param = load_yaml(PROJECT_ROOT / "config" / "physics.yaml")
    param_nodrag = dict(param, air_density=0.0)

    section("M2 · 物理与球动力学 · 验收自检")
    print("配置:", PROJECT_ROOT / "config" / "physics.yaml")
    for k, v in param.items():
        print(f"    {k} = {v}")

    # ------------------------------------------------------------------
    section("【验收 1】能跑出合理的下落 / 抛飞轨迹")
    # ------------------------------------------------------------------
    pos0 = np.array([0.0, 0.0, 2.0])
    vel0 = np.array([15.0, 0.0, 0.0])
    traj = physics.rollout(pos0, vel0, param)

    check("返回 (N, 3) 的 ndarray",
          isinstance(traj, np.ndarray) and traj.ndim == 2 and traj.shape[1] == 3,
          f"shape = {traj.shape}")
    check("轨迹全部有限（无 NaN / Inf）", bool(np.all(np.isfinite(traj))))
    check("起点等于给定初值", bool(np.allclose(traj[0], pos0)),
          f"traj[0] = {np.round(traj[0], 4).tolist()}")
    check("落地前高度非负", bool(np.all(traj[:-1, 2] >= 0.0)),
          f"末点 z = {traj[-1, 2]:.6f} m")

    # ------------------------------------------------------------------
    section("【验收 2】阻力系数：低速 → cd_high，高速 → cd_low")
    # ------------------------------------------------------------------
    cd_lo = float(physics.drag_coefficient(0.0, param))
    cd_hi = float(physics.drag_coefficient(1e6, param))
    check("v = 0 时趋近 cd_high", abs(cd_lo - param["cd_high"]) < 1e-9,
          f"Cd(0) = {cd_lo:.6f}   期望 {param['cd_high']}")
    check("v → ∞ 时趋近 cd_low", abs(cd_hi - param["cd_low"]) < 1e-9,
          f"Cd(1e6) = {cd_hi:.6f}   期望 {param['cd_low']}")

    speeds = np.linspace(0.0, 60.0, 601)
    cds = np.array([float(physics.drag_coefficient(v, param)) for v in speeds])
    check("随速度单调递减", bool(np.all(np.diff(cds) <= 0)),
          f"Cd(0)={cds[0]:.4f} → Cd(60)={cds[-1]:.4f}")
    check("始终落在 [cd_low, cd_high] 区间内",
          bool(np.all(cds >= param["cd_low"] - 1e-12) and np.all(cds <= param["cd_high"] + 1e-12)),
          f"min={cds.min():.6f}  max={cds.max():.6f}")

    # ------------------------------------------------------------------
    section("【验收 3】与「普通斜抛」区分：有阻力落点更近")
    # ------------------------------------------------------------------
    t_drag = physics.rollout(pos0, vel0, param)
    t_nodrag = physics.rollout(pos0, vel0, param_nodrag)

    x_drag = float(t_drag[-1, 0])
    x_nodrag = float(t_nodrag[-1, 0])
    t_drag_s = 0.01 * (len(t_drag) - 1)
    t_nodrag_s = 0.01 * (len(t_nodrag) - 1)

    check("有阻力落点比无阻力近", x_drag < x_nodrag,
          f"有阻力 x = {x_drag:.3f} m  <  无阻力 x = {x_nodrag:.3f} m  "
          f"(缩短 {100 * (1 - x_drag / x_nodrag):.1f}%)")
    check("有阻力滞空时间更长（阻力拖慢下落）", t_drag_s > t_nodrag_s,
          f"有阻力 {t_drag_s:.2f} s  >  无阻力 {t_nodrag_s:.2f} s")

    # 无阻力应退化为解析斜抛：R = vx · sqrt(2h/g)
    R_theory = vel0[0] * np.sqrt(2 * pos0[2] / param["gravity"])
    err = abs(x_nodrag - R_theory) / R_theory
    check("无阻力落点吻合解析解 R = vx·√(2h/g)", err < 0.03,
          f"实测 {x_nodrag:.3f} m vs 理论 {R_theory:.3f} m （偏差 {100 * err:.2f}%）")

    # ------------------------------------------------------------------
    section("【验收 4】最小测试：自由下落单调 + 能量符合常识")
    # ------------------------------------------------------------------
    free = physics.rollout(np.array([0.0, 0.0, 10.0]), np.zeros(3), param)
    z = free[:, 2]
    check("自由下落 z 严格单调递减", bool(np.all(np.diff(z) < 0)),
          f"z: {z[0]:.3f} → {z[-1]:.3f}，共 {len(z)} 点")

    st_drag = integrate([0.0, 0.0, 10.0, 0.0, 0.0, 0.0], param)
    E_drag = energy(st_drag, param)
    rises = int(np.sum(np.diff(E_drag) > 1e-12))
    check("有阻力：机械能单调不增（阻力做负功）", rises == 0,
          f"能量上升次数 = {rises}，E: {E_drag[0]:.5f} J → {E_drag[-1]:.5f} J")

    st_nd = integrate([0.0, 0.0, 10.0, 0.0, 0.0, 0.0], param_nodrag)
    E_nd = energy(st_nd, param_nodrag)
    drift = abs(E_nd[-1] - E_nd[0]) / E_nd[0]
    check("无阻力：机械能守恒（RK4 几乎无漂移）", drift < 1e-6,
          f"相对漂移 = {drift:.2e}")

    # ------------------------------------------------------------------
    section("【附加检查】终端速度 / 落地精度 / 前瞻上限")
    # ------------------------------------------------------------------
    s = np.array([0.0, 0.0, 1000.0, 0.0, 0.0, 0.0])
    for _ in range(50000):
        nxt = physics.step(s, 0.005, param)
        if abs(nxt[5] - s[5]) < 1e-13:
            s = nxt
            break
        s = nxt
    v_term = abs(float(s[5]))
    a_at_term = float(physics.acceleration(s, param)[2])
    check("自由下落收敛到终端速度（不无限加速）", 1.0 < v_term < 20.0 and abs(a_at_term) < 1e-6,
          f"v_term = {v_term:.4f} m/s,  此时 a_z = {a_at_term:.2e} m/s²")

    depth = -float(t_drag[-1, 2])
    check("落地穿刺深度可接受（< 5 mm）", depth < 0.005,
          f"末点 z = {t_drag[-1, 2]:.6f} m（穿地 {depth * 1000:.3f} mm）")

    high = physics.rollout(np.array([0.0, 0.0, 0.5]), np.array([30.0, 0.0, 60.0]), param)
    span = 0.01 * (len(high) - 1)
    check("前瞻时间不超过 2.0 s", span <= 2.0 + 1e-9,
          f"共 {len(high)} 点，覆盖 {span:.3f} s")

    # ------------------------------------------------------------------
    if "--plot" in sys.argv:
        section("生成轨迹图")
        make_plot(param, param_nodrag, pos0, vel0)

    # ------------------------------------------------------------------
    n_pass = sum(_results)
    n_all = len(_results)
    print()
    print("=" * 68)
    print(f"结果：{n_pass} / {n_all} 项通过")
    print("=" * 68)
    if n_pass == n_all:
        print("M2 验收全部通过 ✅")
    else:
        print("存在未通过项，请检查上面的 FAIL 行 ❌")
    return 0 if n_pass == n_all else 1


def make_plot(param, param_nodrag, pos0, vel0):
    """画水平抛出：有阻力 vs 无阻力（横轴水平位移，纵轴高度）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (未安装 matplotlib，跳过绘图)")
        return

    # 中文字体（Windows 自带微软雅黑 / 黑体）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    t_d = physics.rollout(pos0, vel0, param)
    t_n = physics.rollout(pos0, vel0, param_nodrag)

    x_d, x_n = float(t_d[-1, 0]), float(t_n[-1, 0])
    drop = 100.0 * (1.0 - x_d / x_n)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.plot(t_n[:, 0], t_n[:, 2], "--", lw=1.8, color="#888888",
            label="无阻力（标准抛物线）")
    ax.plot(t_d[:, 0], t_d[:, 2], "-", lw=2.4, color="#c0392b",
            label="有阻力（后半段急坠）")
    ax.axhline(0, color="black", lw=1.2)

    # ---- 落点标记 ----
    ax.plot([x_n], [0], "o", color="#888888", ms=8, zorder=5)
    ax.plot([x_d], [0], "o", color="#c0392b", ms=8, zorder=5)
    ax.annotate(f"无阻力落点\n{x_n:.2f} m",
                xy=(x_n, 0), xytext=(x_n - 0.35, 0.72),
                ha="center", color="#555555", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2))
    ax.annotate(f"有阻力落点\n{x_d:.2f} m",
                xy=(x_d, 0), xytext=(x_d + 1.15, 0.50),
                ha="center", color="#c0392b", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

    # ---- 射程缩短 ----
    ax.annotate("", xy=(x_d, 0.11), xytext=(x_n, 0.11),
                arrowprops=dict(arrowstyle="<->", color="#2c3e50", lw=1.4))
    ax.text((x_d + x_n) / 2, 0.15, f"射程缩短 {drop:.1f}%",
            ha="center", va="bottom", color="#2c3e50", fontsize=10.5)

    # ---- 指出"后半段更陡" ----
    idx = int(np.argmin(np.abs(t_d[:, 2] - 0.55)))
    ax.annotate("后半段明显更陡",
                xy=(t_d[idx, 0], t_d[idx, 2]), xytext=(3.05, 1.02),
                color="#c0392b", fontsize=10.5, ha="left",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))

    # ---- 原因说明框（验收第3条：向老师解释原因）----
    ax.text(0.18, 0.62,
            "为什么后半段更陡？\n"
            "球飞行中不断减速 → 速度越低，阻力系数 Cd 越大\n"
            f"（{param['cd_low']} → {param['cd_high']}）→ 阻力占比持续升高 → 下坠越来越快",
            fontsize=9.5, color="#333333", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fdf6e3", ec="#d9b96a"))

    ax.set_xlabel("水平位移 x (m)")
    ax.set_ylabel("高度 z (m)")
    ax.set_title("羽毛球飞行轨迹：有阻力 vs 无阻力")
    ax.set_xlim(-0.4, 10.7)
    ax.set_ylim(-0.09, 2.28)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()

    out = PROJECT_ROOT / "tests" / "trajectory_M2.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out}")


if __name__ == "__main__":
    sys.exit(main())
