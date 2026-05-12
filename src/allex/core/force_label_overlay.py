"""ForceLabelOverlay — viewport 위에 force vector magnitude 텍스트 라벨 표시.

각 force vector prim 의 head tip world 좌표에 anchor 된 2D billboard label 을
omni.ui.scene.SceneView 로 active viewport overlay 에 렌더. arrow 가 hidden
이거나 master toggle 이 OFF 면 라벨도 hidden.

force_torque_visualizer 가 add/update/remove/clear 훅에서 호출하는 singleton.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger("allex.viz")


# ----------------------------------------------------------------------------
# source -> RGBA color (omni.ui.scene.Label color 는 0..1 float tuple).
# ----------------------------------------------------------------------------
_SOURCE_COLOR = {
    "real": (1.0, 0.2, 0.2, 1.0),
    "sim":  (1.0, 0.2, 0.2, 1.0),
    "user": (1.0, 0.2, 0.2, 1.0),
}
_DEFAULT_COLOR = (1.0, 0.2, 0.2, 1.0)
_LABEL_SIZE_PX = 16
# world Z up offset (m) — anchor 를 살짝 올려 arrow head 와 글자가 겹치지 않게.
_LABEL_Z_OFFSET = 0.03


class ForceLabelOverlay:
    """active viewport 위에 magnitude label 을 오버레이하는 singleton.

    Isaac Sim 외 환경 (headless / kit 없음) 에서는 setup 이 실패해도 모든
    메서드가 silent no-op. 로그 WARNING 은 1회만.
    """

    def __init__(self) -> None:
        self._enabled: bool = False
        self._scene_view = None      # omni.ui.scene.SceneView
        self._viewport_frame = None  # ui.Frame inside viewport window
        self._viewport_window = None # 카메라 model bind 해제용 참조
        self._unavailable_warned: bool = False

        # key -> {"source", "transform", "label", "color"}
        self._entries: dict = {}
        # key -> last (text, color, visible, pos) — dirty-check 용
        self._last_state: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """master toggle. OFF 면 모든 entry 제거 + scene_view 해제."""
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        if not enabled:
            self._teardown_scene()
            logger.info("[viz] force_label_overlay disabled")
        else:
            self._ensure_scene()
            logger.info("[viz] force_label_overlay enabled")

    def add(self, key: str, source: str) -> None:
        """라벨 entry 등록. enabled 가 아니어도 metadata 만 기록.

        실제 scene.Label 생성은 첫 update 호출에서 lazy 로.
        """
        if not key:
            return
        src = str(source).lower()
        if src not in _SOURCE_COLOR:
            src = "user"
        rec = self._entries.get(key)
        if rec is None:
            self._entries[key] = {
                "source": src,
                "color": _SOURCE_COLOR.get(src, _DEFAULT_COLOR),
                "transform": None,
                "label": None,
            }
        else:
            rec["source"] = src
            rec["color"] = _SOURCE_COLOR.get(src, _DEFAULT_COLOR)
            # 색 변경 시 다음 update 가 라벨 재생성하도록 stale 처리
            if rec.get("label") is not None:
                self._last_state.pop(key, None)

    def update(self, key: str, world_pos, magnitude_n: float,
               visible: bool) -> None:
        """라벨 위치/텍스트/가시성 갱신. master enabled=False 면 no-op."""
        if not self._enabled:
            return
        if not key:
            return
        rec = self._entries.get(key)
        if rec is None:
            return
        if not self._ensure_scene():
            return

        text = f"{float(magnitude_n):.1f} [N]"
        try:
            px, py, pz = float(world_pos[0]), float(world_pos[1]), float(world_pos[2])
        except Exception:
            return

        state = (text, bool(visible), round(px, 4), round(py, 4), round(pz, 4))
        if self._last_state.get(key) == state:
            return

        try:
            self._rebuild_entry(key, rec, (px, py, pz), text, bool(visible))
            self._last_state[key] = state
        except Exception as exc:
            # SceneView 가 stale 됐을 가능성 — 한 번 재생성 시도.
            if not self._unavailable_warned:
                logger.debug(f"[viz] force_label_overlay update warn: {exc}")
            self._teardown_scene()
            try:
                if self._ensure_scene():
                    self._rebuild_entry(key, rec, (px, py, pz), text, bool(visible))
                    self._last_state[key] = state
            except Exception as exc2:
                if not self._unavailable_warned:
                    logger.warning(f"[viz] force_label_overlay disabled: {exc2}")
                    self._unavailable_warned = True
                self._enabled = False

    def remove(self, key: str) -> None:
        if not key:
            return
        rec = self._entries.pop(key, None)
        self._last_state.pop(key, None)
        if rec is None:
            return
        # Scene 에서 라벨 노드 제거는 전체 scene rebuild 가 가장 간단.
        # entry dict 가 이미 비어있다면 그냥 두고, 아니면 다음 update 가 rebuild.
        self._rebuild_all_from_state()

    def clear(self, source: str | None = None) -> None:
        if source is None:
            self._entries.clear()
            self._last_state.clear()
            self._rebuild_all_from_state()
            return
        src = str(source).lower()
        keys = [k for k, r in self._entries.items() if r.get("source") == src]
        for k in keys:
            self._entries.pop(k, None)
            self._last_state.pop(k, None)
        self._rebuild_all_from_state()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _ensure_scene(self) -> bool:
        """active viewport 위에 SceneView 가 attach 돼 있게 보장. 실패 시 False."""
        if self._scene_view is not None:
            return True
        try:
            from omni.kit.viewport.utility import get_active_viewport_window
            import omni.ui as ui
            import omni.ui.scene as sc
        except Exception as exc:
            if not self._unavailable_warned:
                logger.warning(f"[viz] force_label_overlay deps unavailable: {exc}")
                self._unavailable_warned = True
            return False

        try:
            vp_window = get_active_viewport_window()
            if vp_window is None:
                if not self._unavailable_warned:
                    logger.warning("[viz] force_label_overlay: no active viewport")
                    self._unavailable_warned = True
                return False
            frame = vp_window.get_frame("allex_force_label_overlay")
            with frame:
                self._scene_view = sc.SceneView(
                    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
                )
            self._viewport_frame = frame
            self._viewport_window = vp_window
            # SceneView 를 viewport camera model 에 bind. 이게 없으면 projection 이
            # identity 라 3D world pos → screen pixel 변환이 안 돼 라벨이 화면 밖에 찍힘.
            try:
                vp_window.viewport_api.add_scene_view(self._scene_view)
            except Exception as exc:
                logger.warning(
                    f"[viz] force_label_overlay: viewport_api.add_scene_view failed: {exc}"
                )
            # 등록된 entry 가 있으면 즉시 rebuild.
            self._rebuild_all_from_state()
            return True
        except Exception as exc:
            if not self._unavailable_warned:
                logger.warning(f"[viz] force_label_overlay setup failed: {exc}")
                self._unavailable_warned = True
            self._scene_view = None
            self._viewport_frame = None
            self._viewport_window = None
            return False

    def _teardown_scene(self) -> None:
        """SceneView / frame 해제. entry metadata 는 유지."""
        # viewport_api 에서 SceneView 등록 해제 (없이 destroy 하면 stale ref).
        if self._scene_view is not None and self._viewport_window is not None:
            try:
                self._viewport_window.viewport_api.remove_scene_view(self._scene_view)
            except Exception:
                pass
        try:
            if self._scene_view is not None:
                self._scene_view.scene.clear()
        except Exception:
            pass
        try:
            if self._viewport_frame is not None:
                self._viewport_frame.clear()
        except Exception:
            pass
        self._scene_view = None
        self._viewport_frame = None
        self._viewport_window = None
        # transform/label 참조 무효화
        for rec in self._entries.values():
            rec["transform"] = None
            rec["label"] = None
        self._last_state.clear()

    def _rebuild_entry(self, key: str, rec: dict, world_pos, text: str,
                       visible: bool) -> None:
        """단일 entry 의 scene 노드 (재)생성. transform/label 캐시 갱신."""
        if self._scene_view is None:
            return
        import omni.ui.scene as sc

        # 기존 노드는 그대로 두고 dict 만 갱신하는 패턴은 scene.Transform 의
        # translation 을 set 하는 API 가 버전마다 다름 → 안전하게 entry 전체를
        # rebuild. 단 dirty check 가 위에서 걸러주므로 update 호출 빈도는 낮음.
        self._rebuild_all_from_state(override={
            key: {"pos": world_pos, "text": text, "visible": visible,
                  "color": rec.get("color", _DEFAULT_COLOR)},
        })

    def _rebuild_all_from_state(self, override: dict | None = None) -> None:
        """scene 전체를 last_state + override 로 재구성.

        scene.Transform 의 translation mutate API 가 버전 의존이라, dirty
        검출 후 1회 full rebuild 가 가장 robust. entry 수가 적으니 (~14) 부담 없음.
        """
        if self._scene_view is None:
            return
        import omni.ui as ui
        import omni.ui.scene as sc

        try:
            self._scene_view.scene.clear()
        except Exception:
            pass

        try:
            alignment_center = ui.Alignment.CENTER
        except Exception:
            alignment_center = None

        with self._scene_view.scene:
            for key, rec in self._entries.items():
                state = None
                if override is not None and key in override:
                    state = override[key]
                else:
                    # 기존 last_state 에서 복원.
                    last = self._last_state.get(key)
                    if last is None:
                        continue
                    text, visible, px, py, pz = last
                    state = {"pos": (px, py, pz), "text": text,
                             "visible": visible,
                             "color": rec.get("color", _DEFAULT_COLOR)}
                if not state.get("visible", False):
                    continue
                pos = state["pos"]
                try:
                    tr_mat = sc.Matrix44.get_translation_matrix(
                        float(pos[0]),
                        float(pos[1]),
                        float(pos[2]) + _LABEL_Z_OFFSET,
                    )
                    transform = sc.Transform(transform=tr_mat)
                    with transform:
                        label_kwargs = {
                            "color": state.get("color", _DEFAULT_COLOR),
                            "size": _LABEL_SIZE_PX,
                        }
                        if alignment_center is not None:
                            label_kwargs["alignment"] = alignment_center
                        sc.Label(state["text"], **label_kwargs)
                    rec["transform"] = transform
                except Exception as exc:
                    logger.debug(f"[viz] force_label_overlay node build warn: {exc}")
                    rec["transform"] = None
                    rec["label"] = None


# ----------------------------------------------------------------------------
# module-level singleton accessor
# ----------------------------------------------------------------------------
_overlay_singleton: ForceLabelOverlay | None = None


def get_force_label_overlay() -> ForceLabelOverlay:
    """싱글톤 getter — 최초 호출 시 인스턴스 생성. setup 실패해도 객체는 반환."""
    global _overlay_singleton
    if _overlay_singleton is None:
        _overlay_singleton = ForceLabelOverlay()
    return _overlay_singleton
