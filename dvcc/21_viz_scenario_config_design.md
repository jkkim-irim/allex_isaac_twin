# Viz Scenario Config — 통합 설계 노트 (구현 보류)

`viz_scenario_config.json` 한 파일에 force trigger + torque ring trigger + graph plot
시나리오를 모두 정의. 기존 `force_trigger.json` 은 폐기 (마이그레이션 없이 새 이름으로
대체).

상태: 미구현. 재개 시 이 문서부터 읽고 schema/결정사항 검토 후 진행.

## 핵심 정책

- 파일 위치: `trajectory/<group>/viz_scenario_config.json`
- **파일 없음 → 기존 default 동작 유지** (force 는 sim-active gate, torque ring 은 user
  토글, graph plot 은 없음). 기존 그룹들 그대로 작동.
- 시간 단위 = real CSV `time` 컬럼 (rosbag 첫 sample 0 기준 상대시간 [s])
- `force_trigger.json` 은 더 이상 인식 안 함. 별도 마이그레이션 도구 불필요 (기존 사용자가
  손으로 옮겨도 됨, 5줄짜리 파일).

## 통합 스키마

```json
{
  "_notes": "Real/Sim Replay viz 통합 설정",

  "force_triggers": {
    "ext_force_arm_r":        [[33.0, 38.5]],
    "ext_force_hand_l_thumb": [[33.0, 35.5]]
  },

  "torque_ring_triggers": {
    "real": [[10.0, 50.0]],
    "sim":  [[10.0, 50.0]]
  },

  "graph_plots": [
    {
      "title": "Hand R Index Force",
      "columns": [
        {"name": "ext_force_hand_r_index_x", "source": "real", "label": "Fx"},
        {"name": "ext_force_hand_r_index_y", "source": "real", "label": "Fy"},
        {"name": "ext_force_hand_r_index_z", "source": "real", "label": "Fz"}
      ],
      "active_ranges": [[33.0, 38.5]],
      "window": { "size": [800, 400], "position": [100, 100] },
      "fade":   { "in_s": 0.5, "out_s": 0.5 },
      "y_label": "Force [N]",
      "x_window_s": 5.0
    }
  ]
}
```

`active_ranges` 는 list of `[t_on, t_off]` — 한 plot 이 여러 disjoint 구간에 visible 가능
(force_triggers 와 같은 의미).

## 결정 동결 사항

- **Plot 1개 = subprocess 1개**: 기존 torque_plotter 패턴 그대로 확장. 격리 깔끔, 메모리
  ~30–50MB/proc. 5개까진 부담 없음.
- **plot_proc 일반화 X**: 기존 `plot_proc.py` 는 torque 전용 유지. 새 `plot_proc_generic.py`
  를 column-based time series 용으로 작성. torque 의 dof grouping 특수 로직과 분리.
- **Plot X axis = absolute t** (real CSV `time` 그대로). active_range 시작 시점으로 reset
  하지 않음.
- **Plot data buffer 유지 안 함**: active_range 이탈 시 fade out + withdraw, 재진입 시
  buffer fresh start. 즉 새 active_range 마다 plot 이 빈 상태로 시작.
- **창 lifecycle = "hide" (withdraw) 패턴**: replay start 시 모든 plot subprocess 를
  alpha=0 + withdrawn 으로 spawn. active_range 진입 시 deiconify + fade in. 이탈 시
  fade out + withdraw. replay stop 시 전부 kill. transition 비용 낮음 (즉시 re-show).
- **Multiple active_ranges 지원**: 한 plot 의 `active_ranges` 리스트에 여러 [t_on, t_off]
  쌍 가능. 매 tick "현재 t 가 어떤 range 에라도 들어 있나" 검사 (force_triggers 와 동일
  로직). enter/exit 이벤트는 ranges union 의 boundary 마다 발화 — 즉 각 range 진입/이탈
  마다 show/hide IPC + buffer reset.
- **Fade 옵션 default = 비활성**: per-plot `fade` 필드 optional. 생략 시
  `{"in_s": 0, "out_s": 0}` 으로 적용 — animation 없이 즉시 show/hide. 명시적으로 페이드
  주려면 `{"in_s": 0.5, "out_s": 0.5}` 같이 양수 지정. init 시 fade 시간이 active_ranges
  의 최단 구간 절반보다 길면 WARNING (페이드 미완료로 깔끔하지 않음).
- **시간 비교**: replayer 가 매 step `real_reader.t[idx] - real_reader.t[0]` 로 현재 t 계산,
  active_ranges 와 비교해서 enter/exit 이벤트 발화.
- **Sim/real auto-detect**:
  - real-only prefix: `ext_force_*`, `contact_pos_*`
  - sim-only prefix: `force_vec_*`, `force_aggregate_*`, `aggregate_origin_*`, `normal_*`
  - 양쪽 공존 (`pos_*`, `torque_*`, `time`): JSON 의 `source` 필드 명시 필수
- **`_global` 키 없음**: 채널별 (`real`/`sim`) 만. 동시 토글 필요하면 양쪽 같은 ranges 쓰면 됨.

## 기술적 가능성 (확정)

| 요구             | 가능? | 근거                                                         |
|------------------|-------|--------------------------------------------------------------|
| Window 위치/크기 | ✅    | TkAgg `fig.canvas.manager.window.geometry("WxH+X+Y")`        |
| Fade in/out      | ✅*   | Tk `wm_attributes("-alpha", v)`; *X11 compositor 필요         |
| Show/hide popup  | ✅    | Tk `wm_withdraw()` / `wm_deiconify()`                        |
| Sim/real 구분    | ✅    | column prefix auto-detect + 모호 시 explicit `source`         |

User 환경 (Ubuntu 24.04 + GNOME) 의 compositor 는 mutter — alpha OK. WM 이 alpha 무시하면
즉시 visible/invisible 폴백 (기능 안전, fade 효과만 사라짐).

## 단계별 구현 순서

1. **Phase 1 — force_triggers 통합**
   - `viz_scenario_config.json` 로더 신설 (`_load_viz_scenario`)
   - 기존 `_load_force_triggers` 제거 (force_trigger.json 인식 안 함)
   - JSON 없으면 빈 dict 반환 — 기존 sim-active fallback gate 동작 유지
2. **Phase 2 — torque_ring_triggers** (✅ region 단위 지원 + exclusive 시멘틱)
   - replayer 가 매 step 현재 t 비교 → `set_torque_region_gate(src, region, active)`
     동적 토글 (transition 시에만)
   - 키 형식: `"<source>"` (전체) 또는 `"<source>.<region>"` (예: `real.Hand_L`,
     `real.Hand_L_thumb`, `real.Arm_R`)
   - region 정의 (3-tier arm + 2-tier hand):
     - Arm full: `Arm_<L|R>` (7개)
     - Arm sub-group: `Arm_<L|R>_shoulder` (3), `Arm_<L|R>_wrist` (3)
     - Arm 개별: `Arm_<L|R>_<shoulder_pitch|shoulder_roll|shoulder_yaw|elbow|wrist_yaw|wrist_roll|wrist_pitch>`
     - Hand full: `Hand_<L|R>` (20개)
     - Hand 손가락: `Hand_<L|R>_<thumb|index|middle|ring|little>` (4개씩)
   - dof_abbr → region 매핑은 `_dof_abbr_to_regions` 정적 메서드로 derive
   - **Visibility (exclusive 시멘틱)**:
     - source 에 trigger 없음 → mode-only default
     - source 에 trigger 있음, dof 매칭됨 → 매칭 중 OR (하나라도 True 면 visible)
     - source 에 trigger 있음, dof 매칭 안 됨 → **자동 hide** (다른 region 만 명시한
       trigger 의 부수효과 — user 직관에 맞춤)
   - sim CSV 없으면 sim 채널 트리거 자동 무시
3. **Phase 3 — graph_plots**
   - 새 `plot_proc_generic.py` 작성: init handshake 에 geometry/fade 옵션, show/hide IPC
   - replayer 측 dispatcher: 활성 plot 별 column 값 추출/전송, active_range 진입/이탈 이벤트
   - sim/real auto-detect + explicit source 처리
   - subprocess lifecycle: spawn-on-replay-start / hide-deiconify-fade / kill-on-replay-stop
   - Multiple active_ranges: plot 별 state machine (`out` / `in_range_<idx>`) 으로 ranges
     순회하며 enter/exit 이벤트 발화. ranges 정렬·overlap 검증 init 시 1회.

## 코드 진입점 (재개 시)

- 통합 로더 위치: `src/allex/utils/showcase_replay.py` 의 `_load_force_triggers` 를
  `_load_viz_scenario` 로 교체. 반환 dict 구조 = `{"force_triggers": ..., "torque_ring_triggers":
  ..., "graph_plots": [...]}`.
- Replayer 인자: `force_triggers=...` → `viz_scenario=...` (전체 dict 전달)
- Torque ring runtime gate: `csv_replayer.py::advance` 에서 매 step 시간 비교
- Plot subprocess 매니저: 별도 `viz_plot_manager.py` 신설 (csv_replayer 와 분리)
- 사용법 문서: `dvcc/20_force_trigger_usage.md` 를 `20_viz_scenario_usage.md` 로 확장 + rename

## Fade 구현 detail (Phase 3 시 참고)

```python
# plot_proc_generic.py 안
def fade_to(target_alpha: float, duration_s: float):
    start_alpha = float(root.wm_attributes("-alpha"))
    start_t = time.monotonic()
    def step():
        elapsed = (time.monotonic() - start_t) / duration_s
        if elapsed >= 1.0:
            root.wm_attributes("-alpha", target_alpha)
            if target_alpha == 0.0:
                root.withdraw()
            return
        a = start_alpha + (target_alpha - start_alpha) * elapsed
        root.wm_attributes("-alpha", a)
        root.after(30, step)
    if target_alpha > 0.0 and root.state() == "withdrawn":
        root.wm_attributes("-alpha", 0.0)   # 첫 frame flash 방지
        root.deiconify()
    step()
```

IPC 메시지 (replayer → plot subprocess):
```json
{"cmd": "show", "fade_s": 0.5}
{"cmd": "hide", "fade_s": 0.5}
```

## 미해결 / 추후 결정

(현재 없음 — 이전의 미해결 질문들은 위 "결정 동결" 로 전부 정리됨.)
