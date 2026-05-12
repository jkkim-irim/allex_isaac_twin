"""Extract joint→motor FK polynomial coefficients from allex_control C++.

Parses the polynomial bodies of:
  - CArm{L,R}MJMapper::calcWristJoint2MotorAngle(qr, qp) → 2 outputs
  - CJanghwanFingerMJMapper::cal_motorAngles_janghwan(q1, q2, q3) → 3 outputs
  - CThumb{R,L}MJMapper::cal_motorAngles_thumb(q1, q2, q3) → 3 outputs

Output: `src/allex/trajectory_generate/fk_coeffs.py` containing per-part dicts
        of the form
            FK_<PART> = {
              "inputs":  ["q1", "q2", "q3"],
              "outputs": [
                  [ ((e1, e2, e3), coef), ... ],    # a(0)
                  [ ... ],                          # a(1)
                  [ ... ],                          # a(2)
              ],
            }

Re-run whenever allex_control polynomials change.
"""
from __future__ import annotations

import re
from pathlib import Path

ALLEX = Path("/home/hancheol/wirobotics-rih/allex_control/src/hardware_model/motor_joint")
OUT = Path(__file__).parent.parent / "src/allex/trajectory_generate/fk_coeffs.py"


# ---------------------------------------------------------------------------
# Generic parser
# ---------------------------------------------------------------------------
_NUM_RE = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_TERM_RE = re.compile(
    rf"^\s*a\((\d+)\)\s*\+=\s*({_NUM_RE})\s*((?:\*\s*[A-Za-z][A-Za-z0-9_]*\s*)*);"
)


def _parse_var_token(tok: str, vars_known: list[str]) -> tuple[str, int]:
    """Resolve a token like 'q1', 'q1_3', 'qr', 'qr2' into (var_name, power).

    Supports two conventions:
        - underscored:   q1, q1_2, q1_3, ...           (janghwan / thumb)
        - concatenated:  qr, qr2, qp3, ...             (wrist)
    """
    tok = tok.strip()
    # Underscored: var_<power>
    if "_" in tok:
        var, p = tok.split("_", 1)
        if var in vars_known and p.isdigit():
            return var, int(p)
    # Concatenated: longest matching prefix in vars_known
    for v in sorted(vars_known, key=len, reverse=True):
        if tok == v:
            return v, 1
        if tok.startswith(v):
            suf = tok[len(v):]
            if suf.isdigit():
                return v, int(suf)
    if tok in vars_known:
        return tok, 1
    raise ValueError(f"unknown token {tok!r}, vars={vars_known}")


def parse_function_body(body: str, vars_known: list[str]
                        ) -> dict[int, list[tuple[tuple[int, ...], float]]]:
    """Return {output_idx: [((exp_v1, exp_v2, ...), coef), ...]}."""
    out: dict[int, list[tuple[tuple[int, ...], float]]] = {}
    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        m = _TERM_RE.match(s)
        if not m:
            continue
        idx = int(m.group(1))
        coef = float(m.group(2))
        rest = m.group(3).strip()

        # Build exponent vector
        exps = [0] * len(vars_known)
        if rest:
            # split on '*' (we already stripped leading '*' between coef & factors)
            # rest looks like: "* q1_3 * q2 * q3_2"
            tokens = [t.strip() for t in rest.split("*") if t.strip()]
            for tok in tokens:
                v, p = _parse_var_token(tok, vars_known)
                exps[vars_known.index(v)] += p

        out.setdefault(idx, []).append((tuple(exps), coef))
    return out


def extract_function(src: Path, fn_signature: str) -> str:
    """Return the body of the matching function definition (between {} braces)."""
    text = src.read_text()
    idx = text.find(fn_signature)
    if idx < 0:
        raise RuntimeError(f"signature {fn_signature!r} not found in {src}")
    # Find opening brace after sig
    brace_open = text.find("{", idx)
    if brace_open < 0:
        raise RuntimeError("no opening brace after signature")
    # Find matching closing brace
    depth = 0
    end = brace_open
    while end < len(text):
        c = text[end]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace_open + 1:end]
        end += 1
    raise RuntimeError("unmatched braces")


# ---------------------------------------------------------------------------
# Per-part extraction
# ---------------------------------------------------------------------------
def extract_wrist(side: str):
    src = ALLEX / f"arm_{side.lower()}_mj_map.cpp"
    sig = f"CArm{side}MJMapper::calcWristJoint2MotorAngle"
    body = extract_function(src, sig)
    return parse_function_body(body, ["qr", "qp"])


def extract_janghwan():
    src = ALLEX / "janghwan_mj_map.cpp"
    sig = "CJanghwanFingerMJMapper::cal_motorAngles_janghwan"
    body = extract_function(src, sig)
    return parse_function_body(body, ["q1", "q2", "q3"])


def extract_thumb(side: str):
    src = ALLEX / f"thumb_{side.lower()}_mj_map.cpp"
    sig = f"CThumb{side}MJMapper::cal_motorAngles_thumb"
    body = extract_function(src, sig)
    return parse_function_body(body, ["q1", "q2", "q3"])


# ---------------------------------------------------------------------------
# Output generator
# ---------------------------------------------------------------------------
def fmt_polynomial(name: str, inputs: list[str],
                   terms_per_output: dict[int, list]) -> str:
    n_out = max(terms_per_output) + 1
    out = []
    out.append(f"{name} = {{")
    out.append(f'    "inputs": {inputs!r},')
    out.append('    "outputs": [')
    for k in range(n_out):
        terms = terms_per_output.get(k, [])
        out.append(f"        # output a({k}):  {len(terms)} terms")
        out.append("        [")
        for (exps, coef) in terms:
            out.append(f"            ({tuple(exps)!r}, {coef!r}),")
        out.append("        ],")
    out.append("    ],")
    out.append("}")
    return "\n".join(out)


def main():
    sections: list[str] = []
    sections.append('"""Auto-generated by tools/extract_fk_polynomials.py — do not edit.\n\n'
                    'joint→motor angle polynomial coefficients sourced from allex_control."""')
    sections.append("from __future__ import annotations\n")

    # Wrist L/R
    for side in ("R", "L"):
        terms = extract_wrist(side)
        sections.append(fmt_polynomial(f"FK_WRIST_{side}", ["qr", "qp"], terms))

    # Janghwan finger (shared by 8 fingers)
    terms = extract_janghwan()
    sections.append(fmt_polynomial("FK_FINGER_JANGHWAN", ["q1", "q2", "q3"], terms))

    # Thumb L/R
    for side in ("R", "L"):
        terms = extract_thumb(side)
        sections.append(fmt_polynomial(f"FK_THUMB_{side}", ["q1", "q2", "q3"], terms))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(sections) + "\n")
    print(f"wrote {OUT}")

    # Print summary
    for tag, terms in [
        ("WRIST_R", extract_wrist("R")),
        ("WRIST_L", extract_wrist("L")),
        ("FINGER_JANGHWAN", extract_janghwan()),
        ("THUMB_R", extract_thumb("R")),
        ("THUMB_L", extract_thumb("L")),
    ]:
        sizes = [len(terms.get(k, [])) for k in sorted(terms)]
        print(f"  {tag:20s}  outputs={len(sizes)}  terms/output={sizes}  total={sum(sizes)}")


if __name__ == "__main__":
    main()
