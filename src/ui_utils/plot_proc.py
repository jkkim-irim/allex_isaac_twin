"""ALLEX line-plot subprocess entry point.

Launched via ``subprocess.Popen`` against an *external* Python interpreter
(system ``python3``) from a Kit-side client (see ``torque_plotter.py``).
Isaac Sim's bundled Python does not ship ``_tkinter.so`` and Qt backends
can clash with Kit's own Qt event loop, so the only supported path is
TkAgg running inside a standalone process.

This module is intentionally signal-agnostic: the y-axis label and frame
payload are supplied by the client. Currently used for joint torque, but
force/any per-joint scalar can reuse it by sending a different
``y_label`` in the init handshake.

IPC
---
Line-delimited JSON on stdin. First frame must be an ``init`` handshake,
then a stream of ``sample`` frames, terminated by either a ``quit`` frame
or stdin EOF.

Frames::

    {"cmd":"init", "title":"...", "window_sec":5.0, "hz":200,
     "y_label":"torque [N m]",
     "groups":[{"name":"Arm L","joints":["..."]}, ...]}

    {"t": 12.34, "y": [0.1, -0.2, ...]}   # y follows flat joint order

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
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.widgets import Button  # noqa: E402

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402


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
    shared_t: deque,
    group_joint_dqs: list[list[deque]],
    stop_event: threading.Event,
) -> None:
    """Read stdin lines, route samples into shared t + per-group joint deques.

    ``shared_t`` holds one timestamp per sample.
    ``group_joint_dqs[g][j]`` holds the y-value stream for joint ``j`` of
    group ``g``. The incoming ``y`` list is flat across all groups in the
    order given by the init handshake; ``groups_joint_count[g]`` slices
    that flat list.
    """
    total_expected = sum(groups_joint_count)
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
        y = obj.get("y")
        t = obj.get("t")
        if y is None or t is None:
            continue
        if not isinstance(y, list) or len(y) != total_expected:
            _log(f"y length mismatch: got {len(y) if isinstance(y, list) else '?'} "
                 f"expected {total_expected}; drop")
            continue
        shared_t.append(float(t))
        cursor = 0
        for g, count in enumerate(groups_joint_count):
            joint_dqs = group_joint_dqs[g]
            for j in range(count):
                try:
                    joint_dqs[j].append(float(y[cursor + j]))
                except (TypeError, ValueError):
                    joint_dqs[j].append(0.0)
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
    y_label = str(init.get("y_label", "value"))
    plot_mode = str(init.get("plot_mode", "rolling")).lower()
    if plot_mode not in ("rolling", "cumulative"):
        plot_mode = "rolling"
    save_on_exit = bool(init.get("save_on_exit", False))
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
    groups_joint_count: list[int] = []
    for g in groups:
        joints = list(g.get("joints", []))
        groups_joint_count.append(len(joints))
        group_joint_dqs.append([deque() for _ in joints])

    # Flat joint name list for CSV header.
    flat_joint_names: list[str] = []
    for g in groups:
        flat_joint_names.extend(str(j) for j in g.get("joints", []))

    # Build figure + subplots.
    n_rows = len(groups)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(10, max(3.0, 2.5 * n_rows)),
        sharex=True, num=title,
    )
    if n_rows == 1:
        axes = [axes]

    # Per-axis line objects. Legend entries are pickable — clicking one
    # toggles the corresponding line's visibility (dimmed legend marker
    # indicates hidden). Double-click anywhere on an axes resets all lines.
    axis_specs: list[tuple[object, list]] = []
    leg_to_orig: dict = {}
    for ax, grp in zip(axes, groups):
        lines = []
        for jname in grp.get("joints", []):
            (ln,) = ax.plot([], [], label=str(jname), linewidth=1.0)
            lines.append(ln)
        ax.set_title(str(grp.get("name", "")))
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        leg = ax.legend(loc="upper left", fontsize="x-small", ncol=2)
        for leg_line, orig_line in zip(leg.get_lines(), lines):
            leg_line.set_picker(True)
            leg_line.set_pickradius(5)
            leg_to_orig[id(leg_line)] = orig_line
        axis_specs.append((ax, lines))
    axes[-1].set_xlabel("time [s]")
    # Reserve a strip at the top for Hide/Show all buttons.
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    def _set_all_visible(visible: bool):
        for ax in axes:
            leg = ax.get_legend()
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

        # CSV: time + flat joint columns
        try:
            t_arr_save = np.fromiter(shared_t, dtype=np.float32, count=len(shared_t))
            n = t_arr_save.size
            cols = [t_arr_save.reshape(-1, 1)]
            header = ["time"]
            for grp_dqs, grp in zip(group_joint_dqs, groups):
                for jdq, jname in zip(grp_dqs, grp.get("joints", [])):
                    arr = np.fromiter(jdq, dtype=np.float32, count=len(jdq))
                    if arr.size < n:
                        pad = np.full(n - arr.size, np.nan, dtype=np.float32)
                        arr = np.concatenate([pad, arr])
                    elif arr.size > n:
                        arr = arr[-n:]
                    cols.append(arr.reshape(-1, 1))
                    header.append(str(jname))
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
            for leg_line, orig in leg_to_orig.items():
                orig.set_visible(True)
            for ax in axes:
                leg = ax.get_legend()
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
        args=(sys.stdin, groups_joint_count, shared_t, group_joint_dqs, stop_event),
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
        for (ax, lines), joint_dqs in zip(axis_specs, group_joint_dqs):
            y_min = float("inf")
            y_max = float("-inf")
            for ln, jdq in zip(lines, joint_dqs):
                if not jdq:
                    continue
                y_arr = np.fromiter(jdq, dtype=np.float32, count=len(jdq))
                n = min(t_arr.size, y_arr.size)
                if n == 0:
                    continue
                t_tail = t_arr[-n:]
                y_tail = y_arr[-n:]
                if plot_mode == "rolling":
                    # window 밖 샘플은 화면/autoscale 모두에서 제외.
                    mask = t_tail >= t_min
                    if not mask.any():
                        continue
                    t_vis = t_tail[mask]
                    y_vis = y_tail[mask]
                else:
                    # cumulative: 전체 history 그대로.
                    t_vis = t_tail
                    y_vis = y_tail
                ln.set_data(t_vis, y_vis)
                updated.append(ln)
                # Exclude hidden lines from y-autoscale.
                if not ln.get_visible():
                    continue
                y_min = min(y_min, float(y_vis.min()))
                y_max = max(y_max, float(y_vis.max()))
            ax.set_xlim(t_min, t_last + 1e-3)
            if y_min != float("inf") and y_max != float("-inf"):
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
