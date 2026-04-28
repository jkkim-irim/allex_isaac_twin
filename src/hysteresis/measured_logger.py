"""Measured Logger — Hysteresis 모드 전용.

매 physics step 마다 articulation의 지정 관절 위치를 rad 로 읽어 deg 로
변환해 버퍼링한 뒤, `flush()` 시 `t, abad_deg, mcp_deg, pip_deg` 헤더의
CSV 로 저장한다. HANDOFF_hysteresis_test.md §6 Step 3 명세 그대로.
Isaac Sim 내장 Python 환경에 pandas 가 없으므로 표준 `csv` 모듈을 사용한다.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np


class MeasuredLogger:
    """선택된 관절을 매 step deg 단위로 버퍼링 후 CSV 덤프."""

    _HEADER = ["t", "abad_deg", "mcp_deg", "pip_deg"]

    def __init__(self, articulation, joint_names: list[str],
                 out_path: Path, hz: float, timestamp: bool = True):
        """timestamp=True 면 out_path.stem 뒤에 `_YYYYMMDD_HHMMSS` 를 붙여
        매 실행마다 독립 파일로 저장한다 (덮어쓰기 방지)."""
        self._articulation = articulation
        self._joint_names = list(joint_names)
        base = Path(out_path)
        if timestamp:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = base.with_name(f"{base.stem}_{stamp}{base.suffix}")
        self._out_path = base
        self._hz = float(hz)
        self._dt = 1.0 / self._hz

        dof_names = list(articulation.dof_names or [])
        name_to_idx = {n: i for i, n in enumerate(dof_names)}
        self._indices = [name_to_idx[n] for n in self._joint_names]

        self._buffer: list[tuple[float, float, float, float]] = []
        self._t: float = 0.0

        if len(self._joint_names) != len(self._HEADER) - 1:
            raise ValueError(
                f"MeasuredLogger expects {len(self._HEADER) - 1} joints but got "
                f"{len(self._joint_names)}"
            )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def on_step(self, sim_time_s: float | None = None) -> None:
        """매 physics step에서 호출. sim_time_s 생략 시 내부 시계 사용."""
        positions = self._articulation.get_joint_positions()
        if hasattr(positions, "detach"):
            positions = positions.detach().cpu().numpy()
        positions = np.asarray(positions, dtype=np.float64).reshape(-1)
        deg = np.rad2deg(positions[self._indices])

        t = sim_time_s if sim_time_s is not None else self._t
        self._buffer.append((float(t), float(deg[0]), float(deg[1]), float(deg[2])))
        self._t += self._dt

    def flush(self) -> Path:
        """버퍼를 CSV로 덤프하고 출력 경로 반환."""
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self._HEADER)
            for t, abad, mcp, pip in self._buffer:
                writer.writerow([
                    f"{t:.6f}", f"{abad:.6f}", f"{mcp:.6f}", f"{pip:.6f}",
                ])
        return self._out_path

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def sample_count(self) -> int:
        return len(self._buffer)

    @property
    def out_path(self) -> Path:
        return self._out_path
