"""ALLEX line-plot subprocess entry point.

Launched via ``subprocess.Popen`` against an *external* Python interpreter
(system ``python3``) from a Kit-side client (see ``torque_plotter.py``).
Isaac Sim's bundled Python does not ship ``_tkinter.so`` and Qt backends
can clash with Kit's own Qt event loop, so the only supported path is
TkAgg running inside a standalone process.

Frame schema is channel-keyed: each frame may carry ``y_torque_sim``,
``y_torque_real``, ``y_pos_sim``, ``y_pos_real`` (all optional flat lists of
length = total joints across groups). Per-group ``channel`` in the init
handshake decides which key feeds that subplot. Y-axis label is derived
from channel ("torque [N m]" / "position [deg]"). Position payload is rad;
the reader converts to deg before appending to deques.

IPC
---
Line-delimited JSON on stdin. First frame must be an ``init`` handshake,
then a stream of ``sample`` frames, terminated by either a ``quit`` frame
or stdin EOF.

Frames::

    {"cmd":"init", "title":"...", "window_sec":5.0, "hz":200,
     "groups":[{"name":"Arm L","joints":["..."],"channel":"torque"}, ...]}

    {"t": 12.34, "y_torque_sim":[...], "y_torque_real":[...],
     "y_pos_sim":[...], "y_pos_real":[...]}   # all keys optional

    {"cmd":"quit"}

The main thread drives Tk's mainloop via ``plt.show()``. A background
daemon thread ingests stdin and pushes parsed samples into a rolling
deque per subplot. ``FuncAnimation`` polls the deques every 50 ms.
"""
from __future__ import annotations

# CRITICAL: select backend before any pyplot import.
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.widgets import Button  # noqa: E402

# 어두운 배경 — figure/axes facecolor + tick/label/legend 색을 한 번에 전환.
plt.style.use("dark_background")

# Force/torque vector viz 와 동일한 real(red) / sim(cyan-blue) convention.
# viz_config.colors.real / colors.sim 와 같은 값. plot_proc 는 독립 subprocess 라
# viz_config import 가 안 되므로 hard-code.
_REAL_COLOR = (1.0, 0.0, 0.0)
# _SIM_COLOR = (0.0, 0.8, 0.0)  # green override — plot 만 적용 (subprocess hardcode)
_SIM_COLOR = (0.0, 0.8, 1.0)


def _shade(base_color, idx: int, total: int) -> tuple:
    """Per-joint lightness variation around base color so 같은 hue 안에서도
    여러 joint 가 한 subplot 에 겹쳐 있을 때 구분 가능. idx=0 → 가장 saturated,
    idx 증가 → 흰색 쪽으로 점진 lighten."""
    if total <= 1:
        return mcolors.to_rgb(base_color)
    # mix in [0, 0.55] — 너무 흰색에 가까우면 hue 정체성 사라지므로 상한 0.55.
    mix = 0.55 * (idx / (total - 1))
    r, g, b = mcolors.to_rgb(base_color)
    return (r + (1.0 - r) * mix, g + (1.0 - g) * mix, b + (1.0 - b) * mix)

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402


_RAD_TO_DEG = 180.0 / math.pi


_POLL_INTERVAL_MS = 50
_DEFAULT_HZ = 200.0


def _log(msg: str) -> None:
    sys.stderr.write(f"[ALLEX][PlotProc] {msg}\n")
    sys.stderr.flush()


def _read_init(stdin) -> Optional[dict]:
    """Block until the first valid ``init`` line arrives or EOF."""
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            _log(f"bad init line (skip): {exc}")
            continue
        if obj.get("cmd") == "init":
            return obj
        _log(f"ignoring pre-init frame: {obj!r}")
    return None


def _stdin_reader_loop(
    stdin,
    groups_joint_count: list[int],
    groups_channel: list[str],
    shared_t: deque,
    group_joint_dqs: list[list[deque]],
    group_real_dqs: Optional[list[list[deque]]],
    stop_event: threading.Event,
) -> None:
    """Read stdin lines, route channel-keyed samples into per-group deques.

    Frame keys (all optional, per channel):
      - ``y_torque_sim`` / ``y_torque_real`` — flat list (N m), length = total joints
      - ``y_pos_sim``    / ``y_pos_real``    — flat list (rad)

    Each group is fed only from the key matching ``groups_channel[g]``
    (``"torque"`` or ``"pos"``). 누락 / 길이 불일치 시 NaN 패딩으로 sim/real/t
    parity 유지. rad → deg 변환은 _update 표시 직전이 아니라 reader 에서 한 번만
    수행 (deque 안에 deg 가 들어가도록) — _update 의 autoscale 로직이 단위
    변환을 다시 신경쓰지 않아도 됨.
    """
    has_real = group_real_dqs is not None
    received = 0
    for raw in stdin:
        if stop_event.is_set():
            break
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            _log(f"bad sample (skip): {exc}")
            continue
        cmd = obj.get("cmd")
        if cmd == "quit":
            stop_event.set()
            break
        if cmd == "init":
            continue
        if cmd == "reset":
            shared_t.clear()
            for jdq_list in group_joint_dqs:
                for dq in jdq_list:
                    dq.clear()
            if group_real_dqs is not None:
                for rdq_list in group_real_dqs:
                    for dq in rdq_list:
                        dq.clear()
            continue
        t = obj.get("t")
        if t is None:
            continue

        # Per-channel payloads. Apply rad → deg only to pos channels.
        y_torque_sim = obj.get("y_torque_sim")
        y_torque_real = obj.get("y_torque_real")
        y_pos_sim_rad = obj.get("y_pos_sim")
        y_pos_real_rad = obj.get("y_pos_real")

        def _to_deg(lst):
            if not isinstance(lst, list):
                return None
            out: list[float] = []
            for v in lst:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    out.append(float("nan"))
                    continue
                if fv != fv:  # NaN
                    out.append(fv)
                else:
                    out.append(fv * _RAD_TO_DEG)
            return out

        y_pos_sim = _to_deg(y_pos_sim_rad) if y_pos_sim_rad is not None else None
        y_pos_real = _to_deg(y_pos_real_rad) if y_pos_real_rad is not None else None

        # Frame valid only if at least one channel payload has correct length.
        total_expected = sum(groups_joint_count)

        def _validate(payload):
            return (isinstance(payload, list) and len(payload) == total_expected)

        ok_torque_sim = _validate(y_torque_sim)
        ok_torque_real = _validate(y_torque_real)
        ok_pos_sim = _validate(y_pos_sim)
        ok_pos_real = _validate(y_pos_real)
        if not (ok_torque_sim or ok_torque_real or ok_pos_sim or ok_pos_real):
            continue

        shared_t.append(float(t))
        cursor = 0
        nan = float("nan")
        for g, count in enumerate(groups_joint_count):
            ch = groups_channel[g] if g < len(groups_channel) else "torque"
            sim_dqs = group_joint_dqs[g]
            real_dqs = group_real_dqs[g] if has_real else None
            # Pick the (sim_src, real_src) pair for this group's channel.
            if ch == "pos":
                sim_src = y_pos_sim if ok_pos_sim else None
                real_src = y_pos_real if ok_pos_real else None
            else:
                sim_src = y_torque_sim if ok_torque_sim else None
                real_src = y_torque_real if ok_torque_real else None
            for j in range(count):
                if sim_src is not None:
                    try:
                        sim_dqs[j].append(float(sim_src[cursor + j]))
                    except (TypeError, ValueError):
                        sim_dqs[j].append(nan)
                else:
                    sim_dqs[j].append(nan)
                if real_dqs is not None:
                    if real_src is not None:
                        try:
                            real_dqs[j].append(float(real_src[cursor + j]))
                        except (TypeError, ValueError):
                            real_dqs[j].append(nan)
                    else:
                        real_dqs[j].append(nan)
            cursor += count
        received += 1
        if received % 200 == 0:
            _log(f"recv={received} t_last={t:.3f}")
    stop_event.set()


def _resolve_save_dir() -> Path:
    env = os.environ.get("ALLEX_PLOT_SAVE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / "allex_twin_plots"


def main() -> int:
    init = _read_init(sys.stdin)
    if init is None:
        _log("no init received; exiting")
        return 1

    title = str(init.get("title", "ALLEX Plot"))
    window_sec = float(init.get("window_sec", 5.0))
    hz = float(init.get("hz", _DEFAULT_HZ))
    plot_mode = str(init.get("plot_mode", "rolling")).lower()
    if plot_mode not in ("rolling", "cumulative"):
        plot_mode = "rolling"
    save_on_exit = bool(init.get("save_on_exit", False))
    has_real = bool(init.get("has_real", False))
    # Fixed y-limit (subset 전체 fallback). None → autoscale.
    # group["y_lim"] 가 있는 subplot 은 그걸 우선 사용 (axis_specs 빌드 시 결정).
    def _parse_ylim_pair(v) -> Optional[tuple]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            return None
        try:
            a = float(v[0])
            b = float(v[1])
        except (TypeError, ValueError):
            return None
        return (a, b) if a < b else None

    y_lim: Optional[tuple] = _parse_ylim_pair(init.get("y_lim"))
    groups = init.get("groups", [])
    if not isinstance(groups, list) or not groups:
        _log("init has no groups; exiting")
        return 1

    # Filter empty groups.
    groups = [g for g in groups if g.get("joints")]
    if not groups:
        _log("init has only empty groups; exiting")
        return 1

    # Buffers: maxlen=None (unbounded) for both modes — visualization 차이는
    # _update 의 visible window 처리에서만. 저장 기능은 항상 전체 history 가
    # 있어야 하므로 통일.
    shared_t: deque = deque()
    group_joint_dqs: list[list[deque]] = []
    group_real_dqs: Optional[list[list[deque]]] = [] if has_real else None
    groups_joint_count: list[int] = []
    groups_channel: list[str] = []
    for g in groups:
        joints = list(g.get("joints", []))
        groups_joint_count.append(len(joints))
        ch = str(g.get("channel", "torque"))
        if ch not in ("torque", "pos"):
            ch = "torque"
        groups_channel.append(ch)
        group_joint_dqs.append([deque() for _ in joints])
        if group_real_dqs is not None:
            group_real_dqs.append([deque() for _ in joints])

    # Flat joint name list for CSV header.
    flat_joint_names: list[str] = []
    for g in groups:
        flat_joint_names.extend(str(j) for j in g.get("joints", []))

    # Build figure + subplots. subplot 당 최소 3.5 inch 확보해서 legend 와 plot
    # 데이터 모두 충분한 면적 가지도록.
    n_rows = len(groups)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(12, max(4.0, 3.5 * n_rows)),
        sharex=True, num=title,
    )
    if n_rows == 1:
        axes = [axes]

    # Per-axis line objects. Visual convention:
    #   - real channel: solid red 계열, labeled "real/<jname>"
    #   - sim channel:  dashed blue 계열, labeled "sim/<jname>"
    # has_real=False 면 sim 만 존재, solid + "<jname>" label.
    #
    # Legend 안 모든 entry 는 pickable — 클릭하면 해당 라인 visibility 가 토글되며
    # legend marker 가 dim 처리됨. real/sim 는 각각 독립적으로 토글 가능.
    # axes 더블클릭 시 그 axes 의 모든 라인이 다시 visible.
    #
    # axis_specs entry: (ax, sim_lines, real_lines_or_None, per_joint_legend, ax_y_lim, channel).
    # ax_y_lim 은 그 subplot 전용 (tuple[float, float] | None). data_plotter 가
    # init["groups"][i]["y_lim"] 으로 미리 resolve 해서 보낸다 (group > subset > None).
    # channel == "torque" → y_label "torque [N m]"; "pos" → "position [deg]".
    axis_specs: list[tuple] = []
    leg_to_orig: dict = {}
    for ax, grp in zip(axes, groups):
        sim_lines = []
        real_lines: Optional[list] = [] if has_real else None
        joints = list(grp.get("joints", []))
        n_joints = len(joints)
        # 같은 axes 안에서 real/sim 각각 따로 legend entry 가 생기도록 두 라인 모두
        # label 부여. labeled_lines 순서는 legend 표시 순서 (real_i, sim_i 페어가
        # 연속되게 plot 순서를 유지).
        labeled_lines_ax: list = []
        for j_idx, jname in enumerate(joints):
            # Force/torque vector viz 와 동일 convention: real=red 계열,
            # sim=cyan-blue 계열. 한 subplot 안에서 여러 joint 가 겹칠 때는
            # 같은 hue 안에서 lightness 로 구분.
            real_col = _shade(_REAL_COLOR, j_idx, n_joints)
            sim_col = _shade(_SIM_COLOR, j_idx, n_joints)
            if has_real:
                (ln_real,) = ax.plot(
                    [], [], color=real_col, label=f"Real/{jname}", linewidth=0.8,
                )
                real_lines.append(ln_real)
                labeled_lines_ax.append(ln_real)
                (ln_sim,) = ax.plot(
                    [], [], color=sim_col, linestyle="--", linewidth=0.8,
                    label=f"Sim/{jname}",
                )
                sim_lines.append(ln_sim)
                labeled_lines_ax.append(ln_sim)
            else:
                # No real channel — sim is the sole solid line with the label.
                (ln_sim,) = ax.plot(
                    [], [], color=sim_col, label=str(jname), linewidth=0.8,
                )
                sim_lines.append(ln_sim)
                labeled_lines_ax.append(ln_sim)
        ax.set_title(str(grp.get("name", "")))
        ax_channel = str(grp.get("channel", "torque"))
        if ax_channel not in ("torque", "pos"):
            ax_channel = "torque"
        ax.set_ylabel("position [deg]" if ax_channel == "pos" else "torque [N m]")
        ax.grid(True, alpha=0.3)
        # ncol=2 로 real/sim 페어가 같은 row 에 묶여 보이게. Legend 가 너무 커서
        # plot 영역을 가리면 viz_config.json::torque_plot.subsets 에서 joint 수를
        # 줄이는 식으로 해결.
        leg = ax.legend(loc="upper left", fontsize="x-small", ncol=2)
        for leg_line, orig_line in zip(leg.get_lines(), labeled_lines_ax):
            leg_line.set_picker(True)
            leg_line.set_pickradius(5)
            leg_to_orig[id(leg_line)] = orig_line
        # group["y_lim"] 우선, 없으면 init 의 subset-level y_lim fallback.
        ax_y_lim = _parse_ylim_pair(grp.get("y_lim")) or y_lim
        # 사전 등록된 시간 구간 강조 띠 (axvspan). torque_plotter 가
        # init["groups"][i]["highlights"] 로 그대로 넘긴 list 사용. 안전하게
        # 형식 검증 후 추가; cleanup 은 subprocess 종료 시 자연 정리.
        highlights = grp.get("highlights")
        if isinstance(highlights, list):
            for h in highlights:
                if not isinstance(h, dict):
                    continue
                t0 = h.get("t0")
                t1 = h.get("t1")
                if not isinstance(t0, (int, float)) or not isinstance(t1, (int, float)):
                    continue
                kw = {
                    "facecolor": str(h.get("color", "#ff8c00")),
                    "alpha": float(h.get("alpha", 0.15)),
                    "zorder": 0,
                }
                if "edgecolor" in h:
                    kw["edgecolor"] = str(h["edgecolor"])
                    kw["linewidth"] = float(h.get("linewidth", 0.8))
                else:
                    kw["linewidth"] = 0.0
                ax.axvspan(float(t0), float(t1), **kw)
        axis_specs.append((ax, sim_lines, real_lines, leg, ax_y_lim, ax_channel))
    axes[-1].set_xlabel("time [s]")
    # rect=[left, bottom, right, top] — 위 strip 은 Hide/Show 버튼, 아래는 x-label,
    # 왼쪽은 y-label, 오른쪽도 숨 좀 트이게 작은 여백. 0(=가장자리 붙임) 이면 figure
    # 가 좁을 때 라벨이나 마지막 tick 이 잘림.
    fig.tight_layout(rect=[0.05, 0.02, 0.99, 0.92], h_pad=1.5)

    def _set_all_visible(visible: bool):
        for _ax, _sl, _rl, leg, _yl, _ch in axis_specs:
            if leg is None:
                continue
            for leg_line in leg.get_lines():
                orig = leg_to_orig.get(id(leg_line))
                if orig is not None:
                    orig.set_visible(visible)
                leg_line.set_alpha(1.0 if visible else 0.25)
        fig.canvas.draw_idle()

    btn_ax_hide = fig.add_axes([0.80, 0.94, 0.08, 0.04])
    btn_hide = Button(btn_ax_hide, "Hide all")
    btn_ax_show = fig.add_axes([0.89, 0.94, 0.08, 0.04])
    btn_show = Button(btn_ax_show, "Show all")
    btn_hide.on_clicked(lambda _e: _set_all_visible(False))
    btn_show.on_clicked(lambda _e: _set_all_visible(True))

    def _save_data(_evt=None):
        if not shared_t:
            _log("save: no samples to write")
            return
        save_dir = _resolve_save_dir()
        _log(f"save: target dir = {save_dir}")
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            _log(f"save: mkdir failed: {exc}")
            return
        # 파일명: <title 의 안전화 prefix>_<timestamp>
        safe_title = "".join(c if c.isalnum() or c in "._-" else "_" for c in title)[:60]
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem = f"{safe_title}_{ts}"
        csv_path = save_dir / f"{stem}.csv"
        png_path = save_dir / f"{stem}.png"

        # CSV: time + flat joint columns. When has_real, each joint gets a
        # paired ``<joint>_real`` column right after its sim column.
        try:
            t_arr_save = np.fromiter(shared_t, dtype=np.float32, count=len(shared_t))
            n = t_arr_save.size
            cols = [t_arr_save.reshape(-1, 1)]
            header = ["time"]

            def _align(arr: np.ndarray, target_n: int) -> np.ndarray:
                if arr.size < target_n:
                    pad = np.full(target_n - arr.size, np.nan, dtype=np.float32)
                    return np.concatenate([pad, arr])
                if arr.size > target_n:
                    return arr[-target_n:]
                return arr

            for g_idx, (grp_dqs, grp) in enumerate(zip(group_joint_dqs, groups)):
                grp_real_dqs = (
                    group_real_dqs[g_idx]
                    if (group_real_dqs is not None) else None
                )
                joint_names = grp.get("joints", [])
                for j, (jdq, jname) in enumerate(zip(grp_dqs, joint_names)):
                    arr = np.fromiter(jdq, dtype=np.float32, count=len(jdq))
                    arr = _align(arr, n)
                    cols.append(arr.reshape(-1, 1))
                    header.append(str(jname))
                    if grp_real_dqs is not None:
                        rdq = grp_real_dqs[j]
                        rarr = np.fromiter(rdq, dtype=np.float32, count=len(rdq))
                        rarr = _align(rarr, n)
                        cols.append(rarr.reshape(-1, 1))
                        header.append(f"{jname}_real")
            data = np.hstack(cols)
            np.savetxt(
                csv_path, data, delimiter=",",
                header=",".join(header), comments="", fmt="%.6f",
            )
        except Exception as exc:
            _log(f"save CSV failed: {exc}")
            csv_path = None

        try:
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
        except Exception as exc:
            _log(f"save PNG failed: {exc}")
            png_path = None

        _log(f"saved: {csv_path}  +  {png_path}")

    # Strong refs — matplotlib widgets are GC'd if unreferenced.
    fig._allex_btns = (btn_hide, btn_show)

    def _on_pick(event):
        leg_line = event.artist
        orig = leg_to_orig.get(id(leg_line))
        if orig is None:
            return
        visible = not orig.get_visible()
        orig.set_visible(visible)
        leg_line.set_alpha(1.0 if visible else 0.25)
        fig.canvas.draw_idle()

    def _on_dblclick(event):
        if event.dblclick:
            for _leg_id, orig in leg_to_orig.items():
                orig.set_visible(True)
            for _ax, _sl, _rl, leg, _yl, _ch in axis_specs:
                if leg is not None:
                    for ll in leg.get_lines():
                        ll.set_alpha(1.0)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", _on_pick)
    fig.canvas.mpl_connect("button_press_event", _on_dblclick)

    stop_event = threading.Event()

    # Launch stdin reader thread.
    reader = threading.Thread(
        target=_stdin_reader_loop,
        args=(sys.stdin, groups_joint_count, groups_channel, shared_t,
              group_joint_dqs, group_real_dqs, stop_event),
        name="ALLEXPlotStdin",
        daemon=True,
    )
    reader.start()

    def _on_close(_evt):
        stop_event.set()
        try:
            plt.close(fig)
        except Exception:
            pass

    fig.canvas.mpl_connect("close_event", _on_close)

    def _update(_frame_idx):
        if stop_event.is_set():
            try:
                plt.close(fig)
            except Exception:
                pass
            return []

        if not shared_t:
            return []
        t_arr = np.fromiter(shared_t, dtype=np.float32, count=len(shared_t))
        t_last = float(t_arr[-1])
        if plot_mode == "cumulative":
            # 시작부터 모든 데이터, x 는 첫 sample ~ now.
            t_min = float(t_arr[0])
        else:
            t_min = t_last - window_sec
        updated = []
        for spec_idx, ((ax, sim_lines, real_lines, _leg, ax_y_lim, _ch), joint_dqs) in enumerate(
            zip(axis_specs, group_joint_dqs)
        ):
            real_jdqs = (
                group_real_dqs[spec_idx]
                if (group_real_dqs is not None) else None
            )
            y_min = float("inf")
            y_max = float("-inf")
            for j, (ln, jdq) in enumerate(zip(sim_lines, joint_dqs)):
                if not jdq:
                    continue
                y_arr = np.fromiter(jdq, dtype=np.float32, count=len(jdq))
                n = min(t_arr.size, y_arr.size)
                if n == 0:
                    continue
                t_tail = t_arr[-n:]
                y_tail = y_arr[-n:]
                # Optional paired real series.
                ln_real = (
                    real_lines[j] if (real_lines is not None and j < len(real_lines))
                    else None
                )
                yr_tail = None
                if ln_real is not None and real_jdqs is not None:
                    rdq = real_jdqs[j]
                    if rdq:
                        yr_arr = np.fromiter(rdq, dtype=np.float32, count=len(rdq))
                        nr = min(n, yr_arr.size)
                        if nr > 0:
                            # Re-slice with same n (sim/real stay length-aligned
                            # because reader appends in lockstep, but be defensive).
                            yr_tail = yr_arr[-n:] if yr_arr.size >= n else yr_arr[-nr:]
                if plot_mode == "rolling":
                    # window 밖 샘플은 화면/autoscale 모두에서 제외.
                    mask = t_tail >= t_min
                    if not mask.any():
                        continue
                    t_vis = t_tail[mask]
                    y_vis = y_tail[mask]
                    if yr_tail is not None and yr_tail.size == t_tail.size:
                        yr_vis = yr_tail[mask]
                    else:
                        yr_vis = None
                else:
                    # cumulative: 전체 history 그대로.
                    t_vis = t_tail
                    y_vis = y_tail
                    yr_vis = yr_tail if (yr_tail is not None and
                                         yr_tail.size == t_tail.size) else None
                ln.set_data(t_vis, y_vis)
                updated.append(ln)
                if ln_real is not None:
                    if yr_vis is not None:
                        ln_real.set_data(t_vis, yr_vis)
                    else:
                        # No real samples yet — keep line empty (not 0).
                        ln_real.set_data([], [])
                    updated.append(ln_real)
                # Exclude hidden lines from y-autoscale.
                if not ln.get_visible():
                    continue
                # nanmin/nanmax — NaN 패딩된 real 샘플을 자동스케일에서 무시.
                if y_vis.size > 0:
                    y_min = min(y_min, float(np.nanmin(y_vis)))
                    y_max = max(y_max, float(np.nanmax(y_vis)))
                if yr_vis is not None and yr_vis.size > 0:
                    yr_min = float(np.nanmin(yr_vis))
                    yr_max = float(np.nanmax(yr_vis))
                    if yr_min == yr_min:  # not NaN
                        y_min = min(y_min, yr_min)
                        y_max = max(y_max, yr_max)
            ax.set_xlim(t_min, t_last + 1e-3)
            if ax_y_lim is not None:
                # 고정 y-axis — autoscale 끔. group["y_lim"] 또는 subset fallback.
                ax.set_ylim(ax_y_lim[0], ax_y_lim[1])
            elif y_min != float("inf") and y_max != float("-inf"):
                if y_min == y_max:
                    y_min -= 1.0
                    y_max += 1.0
                margin = 0.1 * (y_max - y_min)
                ax.set_ylim(y_min - margin, y_max + margin)
        return updated

    anim = FuncAnimation(  # noqa: F841
        fig, _update, interval=_POLL_INTERVAL_MS,
        blit=False, cache_frame_data=False,
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        # save_on_exit: subprocess 종료 직전 마지막 buffer 를 dump.
        if save_on_exit:
            try:
                _save_data()
            except Exception as exc:
                _log(f"save_on_exit failed: {exc}")
        try:
            plt.close("all")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
