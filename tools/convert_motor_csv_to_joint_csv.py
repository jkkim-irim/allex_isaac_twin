#!/usr/bin/env python3
"""Motor-ref trajectory CSV → Joint-단 trajectory CSV 일괄 변환.

각 CSV의 `trq_lim_*` / `K_pos_*` / `K_vel_*` 컬럼이 모터-단 값이라고 가정하고
shoulder/elbow/wrist/finger/thumb 변환을 적용해 joint-단 값으로 덮어쓴다.
joint position(deg) 컬럼은 그대로 보존. comment(`-1,...`) / 빈 줄 / multi-section
헤더 구조도 보존.

사용법:
    python tools/convert_motor_csv_to_joint_csv.py \
        --src trajectory/demo1_dynamic_group_motor_ref_data \
        --dst trajectory/demo1_dynamic_group \
        [--dry-run]

기본적으로 `--dst` 의 동명 파일은 `<filename>.bak` 으로 백업 후 새로 작성.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np

# motor_joint_transform 을 site-packages 경로 의존 없이 직접 로드 (Isaac Sim 미실행 환경에서도 동작).
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_MJT_PATH = _REPO / "src" / "allex" / "trajectory_generate" / "motor_joint_transform.py"

_spec = importlib.util.spec_from_file_location("_mjt", _MJT_PATH)
mjt = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec and _spec.loader
_spec.loader.exec_module(mjt)  # type: ignore[union-attr]


# --------------------------------------------------------------------
# Section schema (헤더 라인에서 추출)
# --------------------------------------------------------------------
class SectionSchema:
    """현재 섹션의 컬럼 인덱스 매핑."""

    def __init__(self) -> None:
        self.joint_cols: list[int] = []
        self.trq_cols: list[int] = []
        self.kp_cols: list[int] = []
        self.kd_cols: list[int] = []

    def parse_header(self, parts: list[str]) -> None:
        self.joint_cols = []
        self.trq_cols = []
        self.kp_cols = []
        self.kd_cols = []
        for i, name in enumerate(parts):
            if name.startswith("joint_"):
                self.joint_cols.append(i)
            elif name.startswith("trq_lim_"):
                self.trq_cols.append(i)
            elif name.startswith("K_pos_"):
                self.kp_cols.append(i)
            elif name.startswith("K_vel_"):
                self.kd_cols.append(i)

    @property
    def n_joints(self) -> int:
        return len(self.joint_cols)


# --------------------------------------------------------------------
# 셀 단위 유틸
# --------------------------------------------------------------------
def _format_value(x: float) -> str:
    """관절-단 변환 결과 셀 포맷. NaN 은 빈 문자열."""
    if np.isnan(x):
        return ""
    # 일반적으로 큰 값(K_j는 수만 수준)도 있으니 6 sig fig 면 충분.
    # 정수면 정수처럼, 분수면 유효숫자 보존.
    return f"{x:.6g}"


def _parse_cell_or_nan(s: str) -> float:
    s = s.strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _read_motor_vec(parts: list[str], cols: list[int], n: int) -> np.ndarray | None:
    """행에서 모터-단 N-vector 추출. 모든 셀 비어있으면 None."""
    if not cols:
        return None
    vals = []
    any_present = False
    for c in cols:
        if c < len(parts) and parts[c].strip():
            vals.append(_parse_cell_or_nan(parts[c]))
            any_present = True
        else:
            vals.append(float("nan"))
    if not any_present:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) != n:
        # 컬럼 수 불일치 (이상 케이스).
        return None
    return arr


# --------------------------------------------------------------------
# 라인 단위 변환
# --------------------------------------------------------------------
def _transform_data_row(
    parts: list[str],
    schema: SectionSchema,
    layout: list[tuple[str, int]],
) -> list[str]:
    """데이터 행 한 줄을 변환. 원본 cell 리스트를 in-place 갱신해 반환."""
    n = schema.n_joints
    if n == 0:
        return parts  # 헤더 없음 — 그대로

    # 1) joint position (deg → rad)
    pos_deg = np.zeros(n, dtype=np.float64)
    for i, c in enumerate(schema.joint_cols):
        if c >= len(parts):
            return parts  # 행이 jointed cols 보다 짧음 — 변환 불가, 원본 보존
        pos_deg[i] = _parse_cell_or_nan(parts[c])
    if np.any(np.isnan(pos_deg)):
        return parts  # 위치 결측 — 보통 hermite_spline 도 skip
    q_rad_full = np.deg2rad(pos_deg)

    # 2) 모터-단 입력 추출 (있는 카테고리만)
    trq_full = _read_motor_vec(parts, schema.trq_cols, n)
    kp_full = _read_motor_vec(parts, schema.kp_cols, n)
    kd_full = _read_motor_vec(parts, schema.kd_cols, n)

    if trq_full is None and kp_full is None and kd_full is None:
        return parts  # 변환할 카테고리 없음

    # 3) 그룹별 슬라이스 변환
    trq_out = trq_full.copy() if trq_full is not None else None
    kp_out = kp_full.copy() if kp_full is not None else None
    kd_out = kd_full.copy() if kd_full is not None else None

    cursor = 0
    for group_kind, count in layout:
        sl = slice(cursor, cursor + count)
        kwargs: dict[str, np.ndarray] = {}
        if trq_full is not None:
            kwargs["trq_lim"] = trq_full[sl]
        if kp_full is not None:
            kwargs["k_pos"] = kp_full[sl]
        if kd_full is not None:
            kwargs["k_vel"] = kd_full[sl]
        out = mjt.transform_group(group_kind, q_rad_full[sl], **kwargs)
        if "trq_lim" in out and trq_out is not None:
            trq_out[sl] = out["trq_lim"]
        if "k_pos" in out and kp_out is not None:
            kp_out[sl] = out["k_pos"]
        if "k_vel" in out and kd_out is not None:
            kd_out[sl] = out["k_vel"]
        cursor += count

    # 4) 변환 결과를 원래 자리에 splice
    new_parts = list(parts)

    def _splice(cols: list[int], vec: np.ndarray | None) -> None:
        if vec is None:
            return
        for i, c in enumerate(cols):
            # 원본 셀이 비어있던 자리는 그대로 비워둠 (NaN→빈 문자열)
            if c >= len(new_parts):
                # 행 길이 확장 (구분자 채우기)
                while len(new_parts) <= c:
                    new_parts.append("")
            new_parts[c] = _format_value(float(vec[i]))

    _splice(schema.trq_cols, trq_out)
    _splice(schema.kp_cols, kp_out)
    _splice(schema.kd_cols, kd_out)

    return new_parts


def _layout_total_dofs(layout: list[tuple[str, int]]) -> int:
    return sum(c for _, c in layout)


def _is_data_line(parts: list[str]) -> bool:
    """첫 셀이 양수면 데이터 행. 빈/주석/-1 등은 아님."""
    if not parts or not parts[0].strip():
        return False
    try:
        d = float(parts[0])
    except ValueError:
        return False
    return d > 0


def convert_csv(
    src_path: Path,
    dst_path: Path,
    layout: list[tuple[str, int]],
    *,
    dry_run: bool = False,
    backup_suffix: str = ".bak",
) -> dict[str, int]:
    """단일 CSV 변환. 통계 dict 반환."""
    stats = {"rows_total": 0, "rows_transformed": 0, "rows_skipped": 0, "sections": 0}
    src_text = src_path.read_text(encoding="utf-8")
    out_lines: list[str] = []

    schema = SectionSchema()
    n_joints_first: int | None = None

    for raw_line in src_text.splitlines():
        # 빈 줄: 보존
        if not raw_line.strip():
            out_lines.append(raw_line)
            continue

        # CSV split (공백 trim)
        parts = [p.strip() for p in raw_line.split(",")]

        # 헤더 라인 (re-header for new section)
        if parts and parts[0] == "duration":
            schema.parse_header(parts)
            stats["sections"] += 1
            if n_joints_first is None:
                n_joints_first = schema.n_joints
                expected = _layout_total_dofs(layout)
                if n_joints_first != expected:
                    raise ValueError(
                        f"{src_path.name}: joint columns ({n_joints_first}) "
                        f"!= layout total ({expected}) — fix GROUP_LAYOUT"
                    )
            out_lines.append(raw_line)  # 헤더는 그대로
            continue

        # 데이터 행 vs 주석
        if _is_data_line(parts):
            stats["rows_total"] += 1
            new_parts = _transform_data_row(parts, schema, layout)
            if new_parts is parts:
                stats["rows_skipped"] += 1
                out_lines.append(raw_line)
            else:
                stats["rows_transformed"] += 1
                # trailing empty cells 가 추가되었을 수 있으니 ',' 로 합치되,
                # 모든 trailing 공백 셀을 제거 (원본도 trailing comma 없는 형태).
                while new_parts and new_parts[-1] == "":
                    new_parts.pop()
                out_lines.append(",".join(new_parts))
        else:
            # comment / -1 marker / 기타
            out_lines.append(raw_line)

    new_text = "\n".join(out_lines)
    # 원본이 trailing newline 으로 끝났으면 보존
    if src_text.endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"

    if dry_run:
        print(f"  [dry-run] would write {dst_path} ({len(new_text)} bytes)")
        return stats

    # 백업
    if dst_path.exists():
        bak = dst_path.with_suffix(dst_path.suffix + backup_suffix)
        if bak.exists():
            bak.unlink()
        shutil.move(str(dst_path), str(bak))
        print(f"  backed up: {dst_path.name} → {bak.name}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(new_text, encoding="utf-8")
    return stats


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", type=Path, required=True, help="motor-ref CSV 디렉토리")
    p.add_argument("--dst", type=Path, required=True, help="joint CSV 출력 디렉토리")
    p.add_argument("--dry-run", action="store_true", help="파일 쓰지 않고 통계만 출력")
    p.add_argument(
        "--backup-suffix",
        default=".bak",
        help="출력 디렉토리의 기존 파일 백업 접미사 (default: .bak)",
    )
    args = p.parse_args(argv)

    src: Path = args.src
    dst: Path = args.dst

    if not src.is_dir():
        print(f"ERROR: --src not a directory: {src}", file=sys.stderr)
        return 2

    layout_map = mjt.GROUP_LAYOUT
    files = sorted(src.glob("*.csv"))
    if not files:
        print(f"WARNING: no CSV files in {src}")
        return 0

    print(f"Source: {src}")
    print(f"Dest:   {dst}{' (DRY-RUN)' if args.dry_run else ''}")
    print(f"Files:  {len(files)}")
    print()

    converted = 0
    skipped = 0
    for src_file in files:
        stem = src_file.stem
        if stem not in layout_map:
            print(f"[skip] {src_file.name}: no GROUP_LAYOUT entry")
            skipped += 1
            continue
        layout = layout_map[stem]
        print(f"[convert] {src_file.name} → {dst / src_file.name}")
        try:
            stats = convert_csv(
                src_file,
                dst / src_file.name,
                layout,
                dry_run=args.dry_run,
                backup_suffix=args.backup_suffix,
            )
            print(
                f"  sections={stats['sections']} rows={stats['rows_total']} "
                f"transformed={stats['rows_transformed']} skipped={stats['rows_skipped']}"
            )
            converted += 1
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            skipped += 1

    print()
    print(f"Converted: {converted}, Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
