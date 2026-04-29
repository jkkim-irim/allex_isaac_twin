"""Contact-force vector visualization (ALLEX self-collision only).

Reads MuJoCo Warp contact data each physics step from Newton's stage,
filters to contacts where both geoms belong to the ALLEX articulation,
and draws yellow shaft + orange tip arrows in the Isaac Sim viewport via
``isaacsim.util.debug_draw`` immediate-mode primitives.

Toggled from the Visualizer panel; zero cost when disabled.
"""
from __future__ import annotations

from .ui_settings_utils import UIComponentFactory, UILayout


class ContactForceVisualizer:
    """Toggleable per-step viewport overlay of self-collision force arrows."""

    def __init__(self):
        self._enabled: bool = False
        self._status_label = None
        self._draw_iface = None
        # shape_count snapshot, used to invalidate cached pair / default state
        # when the scene changes (e.g. asset reload).
        self._cached_shape_count: int = -1
        self._first_render_logged: bool = False
        self._step_count: int = 0
        # Allowlist of contact pairs (frozenset of frozenset({shape_idx_a, shape_idx_b})).
        # Resolved lazily from config/contact_config.json once shape_label is known.
        self._allowed_pair_idx = None
        # Aggregate groups loaded from config/contact_config.json (aggregate_groups[]).
        self._aggregate_groups: list[dict] = []
        # Visualization defaults (force_scale, min_force_n, colors, sizes) loaded
        # from config/contact_config.json `defaults`. Populated lazily.
        self._defaults: dict = {}
        # body_label → body index cache for aggregate-group shape lookups.
        self._body_id_cache: dict = {}

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------
    def build(self) -> None:
        UIComponentFactory.create_styled_button(
            "Contact Forces",
            callback=self._toggle,
            color_scheme="blue",
            height=UILayout.BUTTON_HEIGHT,
        )
        self._status_label = UIComponentFactory.create_status_label(
            "Contact Forces: OFF", UILayout.LABEL_WIDTH_LARGE,
        )

    def cleanup(self) -> None:
        self._enabled = False
        self._clear_draw()
        if self._draw_iface is not None:
            try:
                from isaacsim.util.debug_draw import _debug_draw
                _debug_draw.release_debug_draw_interface(self._draw_iface)
            except Exception:
                pass
            self._draw_iface = None
        self._cached_shape_count = -1
        self._allowed_pair_idx = None
        self._aggregate_groups = []
        self._defaults = {}
        self._body_id_cache.clear()

    # ------------------------------------------------------------------
    # Toggle
    # ------------------------------------------------------------------
    def _toggle(self) -> None:
        if not self._enabled:
            try:
                from isaacsim.util.debug_draw import _debug_draw
                self._draw_iface = _debug_draw.acquire_debug_draw_interface()
            except Exception as exc:
                print(f"[ALLEX][ContactViz] debug_draw unavailable: {exc}")
                self._draw_iface = None
                return
            self._enabled = True
            self._first_render_logged = False
            if self._status_label is not None:
                self._status_label.text = "Contact Forces: ON"
            print("[ALLEX][ContactViz] enabled")
        else:
            self._enabled = False
            self._clear_draw()
            if self._status_label is not None:
                self._status_label.text = "Contact Forces: OFF"
            print("[ALLEX][ContactViz] disabled")

    def set_enabled(self, enabled: bool) -> None:
        """Idempotent on/off entry point for external callers (UI 패널 등)."""
        if bool(enabled) != self._enabled:
            self._toggle()

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _force_to_shaft_color(self, force_n: float, fmax: float | None = None) -> tuple:
        """Linear gradient from shaft_color_min → shaft_color_max as force_n grows
        from 0 → fmax. fmax defaults to defaults.shaft_color_max_force_n; per-pair
        and per-aggregate-group overrides are passed in explicitly. Clamped."""
        cmin = self._defaults["shaft_color_min"]
        cmax = self._defaults["shaft_color_max"]
        if fmax is None:
            fmax = float(self._defaults.get("shaft_color_max_force_n", 100.0))
        t = 0.0 if fmax <= 0.0 else max(0.0, min(1.0, float(force_n) / float(fmax)))
        return (
            cmin[0] + t * (cmax[0] - cmin[0]),
            cmin[1] + t * (cmax[1] - cmin[1]),
            cmin[2] + t * (cmax[2] - cmin[2]),
            cmin[3] + t * (cmax[3] - cmin[3]),
        )

    def _clear_draw(self) -> None:
        if self._draw_iface is None:
            return
        try:
            self._draw_iface.clear_lines()
            self._draw_iface.clear_points()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Per-step entry point
    # ------------------------------------------------------------------
    def on_physics_step(self, step_dt: float) -> None:
        if not self._enabled or self._draw_iface is None:
            return
        try:
            self._render_contacts()
        except Exception as exc:
            # Don't kill the physics callback on a viz hiccup; log once.
            if not self._first_render_logged:
                print(f"[ALLEX][ContactViz] render error (logged once): {exc}")
                self._first_render_logged = True

    # ------------------------------------------------------------------
    # Live shape transform via mjw_data.geom_xpos / geom_xmat. Used because
    # body-level xpos can collapse onto the parent body when fixed-joint
    # children are merged (e.g., R_Palm_Link returns R_Wrist_Pitch_Link's
    # transform). Shapes always have their own world transforms in MuJoCo.
    # ------------------------------------------------------------------
    def _resolve_shape_geom(self, shape_idx: int, geom_to_shape_np):
        """Find (world_id, geom_id) for a Newton shape index. Cached."""
        cached = self._body_id_cache.get(("shape", shape_idx), "MISS")
        if cached != "MISS":
            return cached
        import numpy as np
        matches = np.where(geom_to_shape_np == shape_idx)
        wg = None
        if matches[0].size > 0:
            wg = (int(matches[0][0]), int(matches[1][0]))
        self._body_id_cache[("shape", shape_idx)] = wg
        return wg

    def _get_shape_origin_axis(self, shape_idx, axis_local, geom_to_shape_np, mjw_data):
        """Return (origin_world, axis_world_unit) from mjw_data geom transforms."""
        try:
            import numpy as np
            wg = self._resolve_shape_geom(int(shape_idx), geom_to_shape_np)
            if wg is None:
                if not self._first_render_logged:
                    print(f"[ALLEX][ContactViz] shape index {shape_idx} not found in geom_to_shape map")
                return None, None
            w, g = wg
            xpos_arr = getattr(mjw_data, "geom_xpos", None)
            xmat_arr = getattr(mjw_data, "geom_xmat", None)
            if xpos_arr is None or xmat_arr is None:
                if not self._first_render_logged:
                    print("[ALLEX][ContactViz] mjw_data has no geom_xpos/geom_xmat")
                return None, None
            xpos = xpos_arr.numpy()
            xmat = xmat_arr.numpy()
            if xpos.ndim == 3:   pos = xpos[w, g]
            else:                pos = xpos[g]
            if xmat.ndim == 4:                              rot = xmat[w, g]
            elif xmat.ndim == 3 and xmat.shape[-1] == 9:    rot = xmat[w, g].reshape(3, 3)
            elif xmat.ndim == 3:                            rot = xmat[g]
            elif xmat.ndim == 2 and xmat.shape[-1] == 9:    rot = xmat[g].reshape(3, 3)
            else:                                           return None, None
            origin = np.asarray(pos, dtype=np.float32)
            axis_w = np.asarray(rot, dtype=np.float32) @ np.asarray(axis_local, dtype=np.float32)
            n = float(np.linalg.norm(axis_w))
            if n < 1e-9:
                return origin, np.array([0.0, 0.0, 1.0], dtype=np.float32)
            return origin, (axis_w / n).astype(np.float32)
        except Exception as exc:
            if not self._first_render_logged:
                print(f"[ALLEX][ContactViz] _get_shape_origin_axis failed: {exc}")
            return None, None

    # ------------------------------------------------------------------
    # Live body transform (kept for any future per-body needs; aggregate
    # groups now use shape transforms via origin_shape).
    # ------------------------------------------------------------------
    def _resolve_body_id(self, prim_path: str, model) -> int | None:
        cached = self._body_id_cache.get(prim_path, "MISS")
        if cached != "MISS":
            return cached
        body_label = getattr(model, "body_label", None) or []
        body_id: int | None = None
        # 1) Exact match (preferred).
        for i, lbl in enumerate(body_label):
            if lbl == prim_path:
                body_id = i
                break
        # No suffix fallback — too unreliable when fixed-joint links are merged
        # into a parent body. Diagnose by dumping candidates that share a token
        # with prim_path so the user can pick the correct origin_prim.
        if body_id is None:
            tokens = [t for t in prim_path.split("/") if t]
            tail = tokens[-1] if tokens else prim_path
            related = [(i, l) for i, l in enumerate(body_label) if tail.split("_")[0] in l or "Palm" in l or "Wrist" in l]
            print(f"[ALLEX][ContactViz] body_label has no exact entry for '{prim_path}'")
            print(f"  total body_label entries: {len(body_label)}")
            print(f"  candidates that mention a related token (showing up to 20):")
            for i, l in related[:20]:
                print(f"    [{i}] {l}")
            if not related:
                print(f"  (no candidates found; first 10 entries:)")
                for i, l in enumerate(body_label[:10]):
                    print(f"    [{i}] {l}")
        else:
            print(f"[ALLEX][ContactViz] body_id resolved: '{prim_path}' → [{body_id}]")
        self._body_id_cache[prim_path] = body_id
        return body_id

    def _get_prim_origin_axis(self, prim_path: str, axis_local, model, mjw_data):
        """Return (origin_world, axis_world_unit) for prim_path's CURRENT pose.

        Reads from MuJoCo Warp's live body state (`mjw_data.xpos` / `xmat`)
        rather than USD, because Newton does not write transforms back to the
        USD stage — UsdGeom.XformCache returns the initial authored pose.
        """
        try:
            import numpy as np
            body_id = self._resolve_body_id(prim_path, model)
            if body_id is None:
                return None, None
            xpos_arr = getattr(mjw_data, "xpos", None)
            xmat_arr = getattr(mjw_data, "xmat", None)
            if xpos_arr is None or xmat_arr is None:
                if not self._first_render_logged:
                    print("[ALLEX][ContactViz] mjw_data has no xpos/xmat; aggregate origin will be missing")
                return None, None
            xpos = xpos_arr.numpy()  # (n_world, n_body, 3) or (n_body, 3)
            xmat = xmat_arr.numpy()  # rotation, flattened 9 or (3,3)
            # Index into world 0 if multi-world; otherwise direct.
            if xpos.ndim == 3:
                pos = xpos[0, body_id]
            else:
                pos = xpos[body_id]
            if xmat.ndim == 4:        # (n_world, n_body, 3, 3)
                rot = xmat[0, body_id]
            elif xmat.ndim == 3 and xmat.shape[-1] == 9:  # (n_world, n_body, 9)
                rot = xmat[0, body_id].reshape(3, 3)
            elif xmat.ndim == 3:      # (n_body, 3, 3)
                rot = xmat[body_id]
            elif xmat.ndim == 2 and xmat.shape[-1] == 9:  # (n_body, 9)
                rot = xmat[body_id].reshape(3, 3)
            else:
                return None, None
            origin = np.asarray(pos, dtype=np.float32)
            axis_w = np.asarray(rot, dtype=np.float32) @ np.asarray(axis_local, dtype=np.float32)
            n = float(np.linalg.norm(axis_w))
            if n < 1e-9:
                return origin, np.array([0.0, 0.0, 1.0], dtype=np.float32)
            return origin, (axis_w / n).astype(np.float32)
        except Exception as exc:
            if not self._first_render_logged:
                print(f"[ALLEX][ContactViz] _get_prim_origin_axis({prim_path}) failed: {exc}")
            return None, None

    # ------------------------------------------------------------------
    # Allowlist (config/contact_config.json) cache
    # ------------------------------------------------------------------
    def _ensure_allowed_pairs(self, shape_label: list[str]):
        """Resolve configured pair paths → set of frozenset({sh_a, sh_b}).

        Also resolves `aggregate_groups[]` into self._aggregate_groups.
        Cached per shape_count (rebuilt on scene reload).
        """
        if self._allowed_pair_idx is not None and len(shape_label) == self._cached_shape_count:
            return self._allowed_pair_idx

        import json, os
        path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "config", "contact_config.json",
        ))
        try:
            with open(path) as f:
                data = json.load(f)
            raw_pairs = data.get("pairs", [])
            raw_groups = data.get("aggregate_groups", [])
            raw_defaults = data.get("defaults", {}) or {}
        except Exception as exc:
            print(f"[ALLEX][ContactViz] failed to load {path}: {exc}")
            self._allowed_pair_idx = set()
            self._aggregate_groups = []
            return self._allowed_pair_idx

        # Defaults are sourced from JSON only; colors come in as lists, normalize
        # to tuples for the debug_draw API.
        defaults: dict = dict(raw_defaults)
        for k in ("shaft_color_min", "shaft_color_max", "tip_color"):
            v = defaults.get(k)
            if isinstance(v, list):
                defaults[k] = tuple(float(x) for x in v)
        self._defaults = defaults
        self._cached_shape_count = len(shape_label)
        print(
            f"[ALLEX][ContactViz] defaults: force_scale={self._defaults['force_scale']}, "
            f"min_force_n={self._defaults['min_force_n']}"
        )

        name_to_idx = {name: i for i, name in enumerate(shape_label)}
        # pair_key (frozenset of shape indices) → {"flip": bool, "color_max": float}
        resolved: dict = {}
        missing = []
        global_cmax = float(self._defaults.get("shaft_color_max_force_n", 100.0))
        for entry in raw_pairs:
            a = entry["a"]; b = entry["b"]
            flip = bool(entry.get("flip_direction", False))
            color_max = float(entry.get("shaft_color_max_force_n", global_cmax))
            ia = name_to_idx.get(a); ib = name_to_idx.get(b)
            if ia is None or ib is None:
                if ia is None: missing.append(a)
                if ib is None: missing.append(b)
                continue
            resolved[frozenset((ia, ib))] = {"flip": flip, "color_max": color_max}
        if missing:
            print(f"[ALLEX][ContactViz] {len(missing)} configured shape(s) not found in stage:")
            for m in missing:
                print(f"    - {m}")
        print(f"[ALLEX][ContactViz] {len(resolved)} contact pair(s) loaded from contact_config.json")
        self._allowed_pair_idx = resolved

        # Resolve aggregate groups by EXPLICIT pair listing (same shape as
        # top-level `pairs[]`). Substring matching was removed because it
        # caused over-inclusion (e.g., palm vs torso/arm contacts).
        groups: list[dict] = []
        for entry in raw_groups:
            raw_pair_list = entry.get("pairs") or []
            pair_idx: set = set()
            agg_missing: list[str] = []
            for p in raw_pair_list:
                a = p.get("a"); b = p.get("b")
                if not a or not b:
                    continue
                ia = name_to_idx.get(a); ib = name_to_idx.get(b)
                if ia is None or ib is None:
                    if ia is None: agg_missing.append(a)
                    if ib is None: agg_missing.append(b)
                    continue
                pair_idx.add(frozenset((ia, ib)))
            if agg_missing:
                gname = entry.get("name", "unnamed")
                print(f"[ALLEX][ContactViz] aggregate '{gname}' {len(agg_missing)} shape(s) not found in stage:")
                for m in agg_missing:
                    print(f"    - {m}")
            origin_shape_path = entry.get("origin_shape")
            origin_shape_idx = name_to_idx.get(origin_shape_path) if origin_shape_path else None
            if origin_shape_path and origin_shape_idx is None:
                gname = entry.get("name", "unnamed")
                print(f"[ALLEX][ContactViz] aggregate '{gname}' origin_shape not in shape_label: {origin_shape_path}")
            groups.append({
                "name": entry.get("name", "unnamed"),
                "origin_shape_idx": origin_shape_idx,
                "origin_shape_path": origin_shape_path or "",
                "axis": tuple(float(x) for x in (entry.get("axis") or [0.0, 0.0, 1.0])),
                "force_scale": float(entry.get("force_scale", self._defaults["force_scale"])),
                "color_max": float(entry.get("shaft_color_max_force_n", global_cmax)),
                "pair_idx": pair_idx,
            })
            print(
                f"[ALLEX][ContactViz] aggregate group '{groups[-1]['name']}' resolved to "
                f"{len(pair_idx)} pair(s); origin_shape={groups[-1]['origin_shape_path']} (idx={origin_shape_idx})"
            )
        self._aggregate_groups = groups

        return resolved

    # ------------------------------------------------------------------
    # Contact data → arrow geometry
    # ------------------------------------------------------------------
    def _render_contacts(self) -> None:
        import numpy as np
        import warp as wp
        from isaacsim.physics.newton import acquire_stage

        stage = acquire_stage()
        if stage is None:
            return
        solver = getattr(stage, "solver", None)
        model = getattr(stage, "model", None)
        if solver is None or model is None:
            return
        mjw_data = getattr(solver, "mjw_data", None)
        mjc_geom_to_newton_shape = getattr(solver, "mjc_geom_to_newton_shape", None)
        if mjw_data is None or mjc_geom_to_newton_shape is None:
            return

        # Active contact count (one int host sync, ~μs).
        nacon_arr = getattr(mjw_data, "nacon", None)
        if nacon_arr is None:
            return
        try:
            nacon = int(nacon_arr.numpy()[0])
        except Exception:
            return
        if nacon <= 0:
            self._clear_draw()
            return

        shape_label = getattr(model, "shape_label", None) or []
        if not shape_label:
            self._clear_draw()
            return
        allowed_pairs = self._ensure_allowed_pairs(shape_label)
        # Bail only when neither individual nor aggregate viz is configured.
        if not allowed_pairs and not self._aggregate_groups:
            self._clear_draw()
            return

        # Pull mjw contact arrays as torch (zero-copy) → numpy (Python filter).
        contact = mjw_data.contact
        try:
            geom_t      = wp.to_torch(contact.geom)[:nacon]
            pos_t       = wp.to_torch(contact.pos)[:nacon]
            frame_t     = wp.to_torch(contact.frame)[:nacon]
            efc_addr_t  = wp.to_torch(contact.efc_address)[:nacon, 0]
            worldid_t   = wp.to_torch(contact.worldid)[:nacon]
            efc_force_t = wp.to_torch(mjw_data.efc.force)
            geom_to_shape_t = wp.to_torch(mjc_geom_to_newton_shape)
        except Exception:
            return

        geom_np         = geom_t.cpu().numpy()           # (nacon, 2) int
        pos_np          = pos_t.cpu().numpy()            # (nacon, 3) float
        normal_np       = frame_t[:, 0, :].cpu().numpy() # (nacon, 3) world frame
        efc_addr_np     = efc_addr_t.cpu().numpy()       # (nacon,)
        worldid_np      = worldid_t.cpu().numpy()        # (nacon,)
        efc_force_np    = efc_force_t.cpu().numpy()      # (nworld, njmax)
        geom_to_shape_np = geom_to_shape_t.cpu().numpy() # (nworld, ngeom)

        # Map MuJoCo geom → Newton shape per row.
        sh0 = geom_to_shape_np[worldid_np, geom_np[:, 0]]
        sh1 = geom_to_shape_np[worldid_np, geom_np[:, 1]]

        # Drop unmapped geom rows; the JSON allowlist (pairs[] / aggregate_groups
        # pair_idx) does the actual scope filtering further down.
        valid = (sh0 >= 0) & (sh1 >= 0)
        # Compute reaction force on the surviving rows; classification
        # (individual vs aggregate-group) happens after force-threshold filter.
        keep = np.where(valid)[0]
        if keep.size == 0:
            self._clear_draw()
            return
        addr = efc_addr_np[keep]
        wids = worldid_np[keep]
        force_scalar = np.where(addr >= 0, efc_force_np[wids, np.maximum(addr, 0)], 0.0)
        force_vec = (-force_scalar)[:, None] * normal_np[keep]   # (k, 3) world frame

        # Magnitude noise floor.
        mags = np.linalg.norm(force_vec, axis=1)
        keep2 = mags >= self._defaults["min_force_n"]
        if not keep2.any():
            self._clear_draw()
            return

        sh0_kept = sh0[keep][keep2]
        sh1_kept = sh1[keep][keep2]
        force_scalar_kept = force_scalar[keep2]
        positions_kept = pos_np[keep][keep2]
        force_vec_kept = force_vec[keep2]

        # Classify each surviving row: individual (allowlist) or aggregate group.
        # A row in an aggregate group is consumed by that group and NOT drawn
        # as an individual arrow. Pairs in neither set are ignored.
        indiv_starts: list[tuple] = []
        indiv_ends: list[tuple] = []
        group_sums: list[float] = [0.0] * len(self._aggregate_groups)

        indiv_colors: list[tuple] = []
        for i in range(sh0_kept.shape[0]):
            a = int(sh0_kept[i]); b = int(sh1_kept[i])
            pair_key = frozenset((a, b))
            fmag = abs(float(force_scalar_kept[i]))
            # Aggregate first (replaces individual draw).
            consumed = False
            for gi, g in enumerate(self._aggregate_groups):
                if pair_key in g["pair_idx"]:
                    group_sums[gi] += fmag
                    consumed = True
                    break
            if consumed:
                continue
            if pair_key in allowed_pairs:
                info = allowed_pairs[pair_key]
                sign = -1.0 if info["flip"] else 1.0
                s = positions_kept[i]
                e = s + force_vec_kept[i] * (sign * self._defaults["force_scale"])
                indiv_starts.append((float(s[0]), float(s[1]), float(s[2])))
                indiv_ends.append((float(e[0]), float(e[1]), float(e[2])))
                indiv_colors.append(self._force_to_shaft_color(fmag, info["color_max"]))

        # Aggregate group arrows: bidirectional along origin_shape's local axis.
        agg_starts: list[tuple] = []
        agg_ends: list[tuple] = []
        agg_colors: list[tuple] = []
        for gi, g in enumerate(self._aggregate_groups):
            total = group_sums[gi]
            if total <= 0.0:
                continue
            shape_idx = g.get("origin_shape_idx")
            if shape_idx is None:
                continue
            origin, axis_w = self._get_shape_origin_axis(shape_idx, g["axis"], geom_to_shape_np, mjw_data)
            if origin is None:
                continue
            length = total * g["force_scale"]
            up = origin + axis_w * length
            dn = origin - axis_w * length
            o = (float(origin[0]), float(origin[1]), float(origin[2]))
            color = self._force_to_shaft_color(total, g["color_max"])
            agg_starts.append(o); agg_ends.append((float(up[0]), float(up[1]), float(up[2]))); agg_colors.append(color)
            agg_starts.append(o); agg_ends.append((float(dn[0]), float(dn[1]), float(dn[2]))); agg_colors.append(color)

        all_starts = indiv_starts + agg_starts
        all_ends = indiv_ends + agg_ends
        all_colors = indiv_colors + agg_colors
        n_total = len(all_starts)

        # Submit (clear + draw replaces previous frame).
        self._draw_iface.clear_lines()
        self._draw_iface.clear_points()
        if n_total == 0:
            return
        sizes = [self._defaults["shaft_width"]] * n_total
        self._draw_iface.draw_lines(all_starts, all_ends, all_colors, sizes)
        tip_colors = [self._defaults["tip_color"]] * n_total
        tip_sizes = [self._defaults["tip_size"]] * n_total
        self._draw_iface.draw_points(all_ends, tip_colors, tip_sizes)

        if not self._first_render_logged:
            print(
                f"[ALLEX][ContactViz] drawn: {len(indiv_starts)} individual + "
                f"{len(agg_starts)//2} aggregate group(s); "
                f"max |F| = {float(mags[keep2].max()):.2f} N; "
                f"scale = {self._defaults['force_scale']} m/N"
            )
            self._first_render_logged = True
