"""sim vs real(command) measured-trajectory 비교 플로터 (Hysteresis 전용).

AML hysteresis 세션에서 수집된 세 CSV 를 joint 별로 한 장씩 PNG 로 겹쳐
저장한다.
  - command: 실제 로봇에 들어간 target (real 시계 기준, ramp 포함)
  - real   : 실제 로봇의 엔코더 측정값
  - sim    : Isaac Sim 에서 기록한 측정값 (이미 t=0 기준)
실측 2종 은 sampling rate 가 다르므로(real ~100 Hz, sim 200 Hz) 각자의 time
컬럼으로 plot 하고, command/real 은 첫 timestamp 를 0 으로 리베이스해
sim 과 시간축을 맞춘다.

의존성: numpy + matplotlib (Isaac Sim 내장 python 에 이미 포함). pandas 는
사용하지 않는다.

사용 예:
    python src/hysteresis/compare_sim_real.py \\
        --command trajectory/aml_hysteresis_group/real_data/command_joint_deg_0.csv \\
        --real    trajectory/aml_hysteresis_group/real_data/measured_joint_deg_0.csv \\
        --sim     trajectory/aml_hysteresis_group/sim_measured_20260427_055052.csv \\
        --out     trajectory/aml_hysteresis_group/sim_real_compare_0 \\
        --joints  abad mcp pip
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

# Figure 은 정사각형, y축은 촘촘하고 길게. 가로 축 눈금은 일반적인 밀도를 유지.
_FIGSIZE = (9, 9)          # inches, square
_DPI = 130
_Y_MAJOR_NBINS = 15         # ~15 major y-ticks irrespective of range
_TICK_LEN_MAJOR = 10.0      # long tick marks on y
_TICK_LEN_MINOR = 5.0

# 실제 로봇의 관절 ROM. command 가 이 범위를 벗어나면 실기가 물리적으로
# 포화되므로 MSE 에서 제외한다 (sim 은 해당 구간을 추종 가능해도 real 과의
# 비교 기준을 맞추기 위해 동일 마스크를 적용).
# None = 범위 필터 없음.
_JOINT_RANGES_DEG: dict[str, tuple[float, float] | None] = {
    "abad": None,
    "mcp": (-10.0, 90.0),
    "pip": (0.0, 100.0),
}


def load_csv(csv_path: Path) -> dict[str, np.ndarray]:
    """헤더 기반으로 {col_name: float array} 반환."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    arr = np.asarray(rows, dtype=np.float64)
    return {name: arr[:, i] for i, name in enumerate(header)}


def _rebase_to_zero(data: dict[str, np.ndarray]) -> None:
    data["t"] = data["t"] - data["t"][0]


def plot_joint(
    ax,
    command: dict[str, np.ndarray] | None,
    real: dict[str, np.ndarray],
    sim: dict[str, np.ndarray],
    joint: str,
) -> None:
    col = f"{joint}_deg"
    if command is not None:
        ax.plot(
            command["t"], command[col],
            label="command", linewidth=1.2, alpha=0.9,
            linestyle="--", color="tab:gray",
        )
    ax.plot(real["t"], real[col],
            label="real", linewidth=1.4, alpha=0.95, color="tab:blue")
    ax.plot(sim["t"], sim[col],
            label="sim", linewidth=1.2, alpha=0.95, color="tab:red")

    ax.set_title(f"R_Index {joint.upper()}  (command / real / sim)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(f"{joint}_deg")

    # Dense + long y ticks so the hysteresis curve is readable on a square axis.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=_Y_MAJOR_NBINS, integer=False))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="y", which="major",
                   length=_TICK_LEN_MAJOR, width=1.2, labelsize=9)
    ax.tick_params(axis="y", which="minor",
                   length=_TICK_LEN_MINOR, width=0.8)
    ax.tick_params(axis="x", which="major", length=5)

    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15, linestyle=":")
    ax.legend(loc="best")


def _mse_vs_command(
    series: dict[str, np.ndarray],
    command: dict[str, np.ndarray],
    joint: str,
) -> tuple[float, float, int, int]:
    """series(sim/real) 를 command 시계에 선형 보간해 MSE/RMSE 를 계산.

    두 시계의 공통 구간만 사용하고, `_JOINT_RANGES_DEG[joint]` 가 설정되어
    있으면 보간된 command 값이 ROM 안에 있는 샘플만 남긴다 (실기 포화
    구간 제거). Returns (mse, rmse, n_used, n_total_in_window).
    """
    col = f"{joint}_deg"
    t_series = series["t"]
    y_series = series[col]
    t_cmd = command["t"]
    y_cmd = command[col]

    t_lo = max(t_series[0], t_cmd[0])
    t_hi = min(t_series[-1], t_cmd[-1])
    time_mask = (t_series >= t_lo) & (t_series <= t_hi)
    t_eval = t_series[time_mask]
    y_eval = y_series[time_mask]
    y_cmd_on_series = np.interp(t_eval, t_cmd, y_cmd)

    n_total = int(t_eval.size)
    rng = _JOINT_RANGES_DEG.get(joint)
    if rng is not None:
        lo, hi = rng
        rom_mask = (y_cmd_on_series >= lo) & (y_cmd_on_series <= hi)
        y_eval = y_eval[rom_mask]
        y_cmd_on_series = y_cmd_on_series[rom_mask]

    err = y_eval - y_cmd_on_series
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    return mse, rmse, int(y_eval.size), n_total


def _format_mse_table(
    joints: list[str],
    sim: dict[str, np.ndarray],
    real: dict[str, np.ndarray],
    command: dict[str, np.ndarray],
) -> str:
    """sim-vs-command / real-vs-command MSE 표를 ASCII 로 포맷.

    각 joint 행의 `range` 열은 적용된 ROM 필터(또는 `-`)를 보여주고,
    `n_sim`/`n_real` 은 필터 적용 후 유효 샘플 수 / 공통 구간 샘플 수.
    """
    rows = []
    for joint in joints:
        sim_mse, sim_rmse, n_sim, n_sim_tot = _mse_vs_command(sim, command, joint)
        real_mse, real_rmse, n_real, n_real_tot = _mse_vs_command(real, command, joint)
        rng = _JOINT_RANGES_DEG.get(joint)
        rng_str = f"[{rng[0]:g}, {rng[1]:g}]" if rng is not None else "-"
        rows.append((
            joint, rng_str,
            sim_mse, sim_rmse, real_mse, real_rmse,
            n_sim, n_sim_tot, n_real, n_real_tot,
        ))

    header = (
        f"{'joint':<6} | {'range':>13} | "
        f"{'sim_MSE[deg^2]':>15} | {'sim_RMSE[deg]':>14} | "
        f"{'real_MSE[deg^2]':>16} | {'real_RMSE[deg]':>15} | "
        f"{'n_sim(used/tot)':>17} | {'n_real(used/tot)':>17}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for (joint, rng_str, sim_mse, sim_rmse, real_mse, real_rmse,
         n_sim, n_sim_tot, n_real, n_real_tot) in rows:
        n_sim_str = f"{n_sim}/{n_sim_tot}"
        n_real_str = f"{n_real}/{n_real_tot}"
        lines.append(
            f"{joint:<6} | {rng_str:>13} | "
            f"{sim_mse:>15.4f} | {sim_rmse:>14.4f} | "
            f"{real_mse:>16.4f} | {real_rmse:>15.4f} | "
            f"{n_sim_str:>17} | {n_real_str:>17}"
        )
    return "\n".join(lines)


def run(
    real_csv: Path,
    sim_csv: Path,
    out_dir: Path,
    joints: list[str],
    command_csv: Path | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    real = load_csv(real_csv)
    sim = load_csv(sim_csv)
    _rebase_to_zero(real)

    command = None
    if command_csv is not None:
        command = load_csv(command_csv)
        _rebase_to_zero(command)

    print(
        f"[compare] real rows={real['t'].size} span={real['t'][-1]:.2f}s  "
        f"sim rows={sim['t'].size} span={sim['t'][-1]:.2f}s"
        + (f"  cmd rows={command['t'].size} span={command['t'][-1]:.2f}s"
           if command is not None else "")
    )

    saved: list[Path] = []
    for joint in joints:
        fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
        plot_joint(ax, command, real, sim, joint)
        fig.tight_layout()
        out_path = out_dir / f"{joint}.png"
        fig.savefig(out_path)
        plt.close(fig)
        saved.append(out_path)
        print(f"[compare] wrote {out_path}")

    if command is not None:
        print("\n[compare] MSE / RMSE vs command "
              "(command 을 각 시계에 선형보간 후 공통 구간만 사용)")
        table = _format_mse_table(joints, sim, real, command)
        print(table)
        table_path = out_dir / "mse_vs_command.txt"
        table_path.write_text(table + "\n", encoding="utf-8")
        print(f"[compare] wrote {table_path}")
        saved.append(table_path)

    return saved


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", type=Path, required=True)
    p.add_argument("--sim", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--command", type=Path, default=None,
                   help="optional command CSV to overlay as a dashed reference")
    p.add_argument("--joints", nargs="+", default=["abad", "mcp", "pip"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.real, args.sim, args.out, args.joints, command_csv=args.command)
