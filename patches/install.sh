#!/usr/bin/env bash
# ALLEX Twin — Isaac Sim Newton force-access patches installer
#
# Usage:
#   ISAACSIM_ROOT=/path/to/isaacsim ./install.sh [--dry-run] [--backup]
#
# 네 개의 Isaac Sim 소스 파일을 Newton 의 MuJoCo force 분해 항들
# (qfrc_actuator / qfrc_applied / qfrc_gravcomp) 에 접근 가능하게
# 수정한 버전으로 교체한다. 자세한 내용은 ../docs/newton_force_access.md 참고.

set -euo pipefail

DRY_RUN=0
BACKUP=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --backup)  BACKUP=1 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

if [[ -z "${ISAACSIM_ROOT:-}" ]]; then
  echo "[ERROR] ISAACSIM_ROOT env var is required."
  echo "        e.g. ISAACSIM_ROOT=~/isaacsim $0"
  exit 1
fi

if [[ ! -d "$ISAACSIM_ROOT" ]]; then
  echo "[ERROR] $ISAACSIM_ROOT is not a directory."
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"

# (src_relative_in_patches, dst_relative_in_ISAACSIM_ROOT)
declare -a MAPPING=(
  "isaacsim.physics.newton/python/impl/newton_stage.py|source/extensions/isaacsim.physics.newton/python/impl/newton_stage.py"
  "isaacsim.physics.newton/python/impl/tensors/articulation_view.py|source/extensions/isaacsim.physics.newton/python/impl/tensors/articulation_view.py"
  "isaacsim.core.prims/isaacsim/core/prims/impl/articulation.py|_build/linux-x86_64/release/exts/isaacsim.core.prims/isaacsim/core/prims/impl/articulation.py"
  "isaacsim.core.prims/isaacsim/core/prims/impl/single_articulation.py|_build/linux-x86_64/release/exts/isaacsim.core.prims/isaacsim/core/prims/impl/single_articulation.py"
)

echo "[ALLEX patches] ISAACSIM_ROOT = $ISAACSIM_ROOT"
echo "[ALLEX patches] dry-run = $DRY_RUN, backup = $BACKUP"
echo ""

for entry in "${MAPPING[@]}"; do
  SRC_REL="${entry%%|*}"
  DST_REL="${entry##*|}"
  SRC="$HERE/$SRC_REL"
  DST="$ISAACSIM_ROOT/$DST_REL"

  if [[ ! -f "$SRC" ]]; then
    echo "[SKIP] missing source: $SRC"
    continue
  fi
  if [[ ! -f "$DST" ]]; then
    echo "[WARN] dst not found (path differs?): $DST"
    continue
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] would copy $SRC_REL → $DST_REL"
    continue
  fi

  if [[ "$BACKUP" == "1" ]]; then
    cp -a "$DST" "$DST.orig"
    echo "[BACKUP]  $DST.orig"
  fi

  cp -a "$SRC" "$DST"
  echo "[INSTALL] $DST_REL"
done

echo ""
echo "[ALLEX patches] done. Restart Isaac Sim to pick up changes."
