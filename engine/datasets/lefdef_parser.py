"""
Lightweight LEF/DEF parsers for NanGate45 MacroPlacement flow datasets.

Goals:
- Avoid heavyweight EDA dependencies
- Be robust enough for typical generated LEF/DEF (OpenROAD/Cadence)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable


@dataclass
class DefDie:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    row_height: float
    site_width: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass
class DefComponent:
    name: str
    macro: str
    x: float  # lower-left x (microns)
    y: float  # lower-left y (microns)
    orient: str
    fixed: bool


@dataclass
class DefPin:
    name: str
    x: float  # absolute x (microns)
    y: float  # absolute y (microns)


@dataclass
class DefNet:
    name: str
    pins: List[Tuple[str, str]]  # (inst_or_PIN, pin_name)


_RE_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def collect_lef_files(
    design_root: str,
    extra_roots: Optional[List[str]] = None,
) -> List[str]:
    """Collect .lef files from design_root and optional extra_roots (recursive)."""
    roots = [design_root] + (extra_roots or [])
    lefs: List[str] = []
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        for dirpath, _, filenames in os.walk(r):
            for fn in filenames:
                if fn.lower().endswith(".lef"):
                    lefs.append(os.path.join(dirpath, fn))
    # de-dup, keep stable order
    seen = set()
    out = []
    for p in lefs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def parse_lef_macro_sizes(lef_paths: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    """
    Parse LEF files to extract MACRO size:
      MACRO <name> ... SIZE <w> BY <h> ;

    Returns:
      dict: macro_name -> (w, h) in microns (LEF units).
    """
    sizes: Dict[str, Tuple[float, float]] = {}
    re_macro = re.compile(r"^\s*MACRO\s+(\S+)\s*$", re.IGNORECASE)
    re_size = re.compile(rf"^\s*SIZE\s+({_RE_FLOAT})\s+BY\s+({_RE_FLOAT})\s*;\s*$", re.IGNORECASE)
    re_end = re.compile(r"^\s*END\s+(\S+)\s*$", re.IGNORECASE)

    current: Optional[str] = None
    for path in lef_paths:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", errors="ignore") as f:
                for line in f:
                    m = re_macro.match(line)
                    if m:
                        current = m.group(1)
                        continue
                    if current is not None:
                        ms = re_size.match(line)
                        if ms:
                            w = float(ms.group(1))
                            h = float(ms.group(2))
                            sizes[current] = (w, h)
                            continue
                        me = re_end.match(line)
                        if me and me.group(1) == current:
                            current = None
        except Exception:
            # best-effort parser
            continue

    return sizes


def _parse_def_units(lines: List[str]) -> float:
    # UNITS DISTANCE MICRONS <dbu_per_micron> ;
    re_units = re.compile(r"^\s*UNITS\s+DISTANCE\s+MICRONS\s+(\d+)\s*;\s*$", re.IGNORECASE)
    for ln in lines:
        m = re_units.match(ln)
        if m:
            return float(m.group(1))
    # Default to 1000 dbu per micron if missing (common)
    return 1000.0


def _to_microns(v: float, dbu_per_micron: float) -> float:
    return float(v) / float(dbu_per_micron)


def parse_def(
    def_path: str,
) -> Tuple[DefDie, List[DefComponent], List[DefPin], List[DefNet]]:
    """
    Parse a DEF (Cadence/OpenROAD) for:
    - DIEAREA
    - ROWs (infer row_height, site_width)
    - COMPONENTS with placements (lower-left)
    - PINS with fixed/placed locations
    - NETS with (instance, pin) tuples when present
    """
    with open(def_path, "r", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    dbu_per_micron = _parse_def_units(lines)

    # DIEAREA ( x1 y1 ) ( x2 y2 ) ;
    re_die = re.compile(r"^\s*DIEAREA\s*\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)\s*;\s*$", re.IGNORECASE)
    die_ll = (0.0, 0.0)
    die_ur = (0.0, 0.0)
    for ln in lines:
        m = re_die.match(ln)
        if m:
            die_ll = (_to_microns(float(m.group(1)), dbu_per_micron), _to_microns(float(m.group(2)), dbu_per_micron))
            die_ur = (_to_microns(float(m.group(3)), dbu_per_micron), _to_microns(float(m.group(4)), dbu_per_micron))
            break

    # ROW ... x y ... STEP stepX stepY ;
    re_row = re.compile(r"^\s*ROW\s+\S+\s+\S+\s+(\d+)\s+(\d+)\s+\S+\s+DO\s+\d+\s+BY\s+\d+\s+STEP\s+(\d+)\s+(\d+)\s*;.*$", re.IGNORECASE)
    row_ys: List[float] = []
    site_width = None
    for ln in lines:
        m = re_row.match(ln)
        if m:
            x_dbu = float(m.group(1))
            y_dbu = float(m.group(2))
            step_x = float(m.group(3))
            # step_y = float(m.group(4))  # often 0
            row_ys.append(_to_microns(y_dbu, dbu_per_micron))
            if site_width is None:
                site_width = _to_microns(step_x, dbu_per_micron)

    row_height = 1.0
    if len(row_ys) >= 2:
        ys = sorted(set(row_ys))
        diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        diffs = [d for d in diffs if d > 1e-9]
        if diffs:
            row_height = min(diffs)

    if site_width is None:
        site_width = row_height  # fallback

    die = DefDie(
        x_min=die_ll[0],
        y_min=die_ll[1],
        x_max=die_ur[0],
        y_max=die_ur[1],
        row_height=row_height,
        site_width=site_width,
    )

    # COMPONENTS parsing (best-effort)
    comps_by_name: Dict[str, DefComponent] = {}
    in_components = False
    # Example:
    # - inst macro + PLACED ( x y ) N ;
    re_comp = re.compile(r"^\s*-\s+(\S+)\s+(\S+)\s+.*$", re.IGNORECASE)
    re_place = re.compile(r"^\s*\+\s+(FIXED|PLACED)\s*\(\s*(\d+)\s+(\d+)\s*\)\s+(\S+)\s*;\s*$", re.IGNORECASE)
    # Some DEFs put placement on the same line as the component header.
    re_place_inline = re.compile(r"(?i)\+\s*(FIXED|PLACED)\s*\(\s*(\d+)\s+(\d+)\s*\)\s+(\S+)\s*;")
    current_name: Optional[str] = None
    current_macro: Optional[str] = None
    for ln in lines:
        if ln.strip().upper().startswith("COMPONENTS"):
            in_components = True
            continue
        if in_components and ln.strip().upper().startswith("END COMPONENTS"):
            in_components = False
            current_name = None
            current_macro = None
            continue
        if not in_components:
            continue

        m = re_comp.match(ln)
        if m:
            current_name = m.group(1)
            current_macro = m.group(2)
            # Create a placeholder component immediately; some DEFs omit placement for components.
            if current_name not in comps_by_name:
                comps_by_name[current_name] = DefComponent(
                    name=current_name,
                    macro=current_macro or "",
                    x=0.0,
                    y=0.0,
                    orient="N",
                    fixed=False,
                )
            mi = re_place_inline.search(ln)
            if mi:
                kind = mi.group(1).upper()
                x = _to_microns(float(mi.group(2)), dbu_per_micron)
                y = _to_microns(float(mi.group(3)), dbu_per_micron)
                orient = mi.group(4)
                comps_by_name[current_name] = DefComponent(
                    name=current_name,
                    macro=current_macro or "",
                    x=x,
                    y=y,
                    orient=orient,
                    fixed=(kind == "FIXED"),
                )
                current_name = None
                current_macro = None
            continue
        if current_name is not None:
            mp = re_place.match(ln)
            if mp:
                kind = mp.group(1).upper()
                x = _to_microns(float(mp.group(2)), dbu_per_micron)
                y = _to_microns(float(mp.group(3)), dbu_per_micron)
                orient = mp.group(4)
                comps_by_name[current_name] = DefComponent(
                    name=current_name,
                    macro=current_macro or "",
                    x=x,
                    y=y,
                    orient=orient,
                    fixed=(kind == "FIXED"),
                )
                current_name = None
                current_macro = None
                continue

    comps: List[DefComponent] = list(comps_by_name.values())

    # PINS parsing (fixed/placed)
    pins: List[DefPin] = []
    in_pins = False
    re_pin_start = re.compile(r"^\s*-\s+(\S+)\s+.*$", re.IGNORECASE)
    re_pin_place = re.compile(r"^\s*\+\s+(FIXED|PLACED)\s*\(\s*(\d+)\s+(\d+)\s*\)\s+\S+\s*;\s*$", re.IGNORECASE)
    cur_pin = None
    for ln in lines:
        if ln.strip().upper().startswith("PINS"):
            in_pins = True
            continue
        if in_pins and ln.strip().upper().startswith("END PINS"):
            in_pins = False
            cur_pin = None
            continue
        if not in_pins:
            continue
        ms = re_pin_start.match(ln)
        if ms:
            cur_pin = ms.group(1)
            continue
        if cur_pin is not None:
            mp = re_pin_place.match(ln)
            if mp:
                x = _to_microns(float(mp.group(2)), dbu_per_micron)
                y = _to_microns(float(mp.group(3)), dbu_per_micron)
                pins.append(DefPin(name=cur_pin, x=x, y=y))
                cur_pin = None

    # NETS parsing (optional; many floorplan-only DEFs don't have this)
    nets: List[DefNet] = []
    in_nets = False
    re_nets_header = re.compile(r"^\s*NETS\s+(\d+)\s*;\s*$", re.IGNORECASE)
    re_net_start = re.compile(r"^\s*-\s+(\S+)\s+(.*)$", re.IGNORECASE)
    re_conn = re.compile(r"^\s*\(\s*(\S+)\s+(\S+)\s*\)\s*$")

    cur_net_name = None
    cur_pins: List[Tuple[str, str]] = []

    def _flush_net():
        nonlocal cur_net_name, cur_pins
        if cur_net_name is not None and len(cur_pins) >= 2:
            nets.append(DefNet(name=cur_net_name, pins=cur_pins))
        cur_net_name = None
        cur_pins = []

    for ln in lines:
        if re_nets_header.match(ln):
            in_nets = True
            continue
        if in_nets and ln.strip().upper().startswith("END NETS"):
            _flush_net()
            in_nets = False
            continue
        if not in_nets:
            continue

        ms = re_net_start.match(ln)
        if ms:
            _flush_net()
            cur_net_name = ms.group(1)
            rest = ms.group(2)
            # Parse any ( inst pin ) tuples on the same line
            for tup in re.findall(r"\(\s*\S+\s+\S+\s*\)", rest):
                mc = re_conn.match(tup)
                if mc:
                    cur_pins.append((mc.group(1), mc.group(2)))
            # Net might end on same line with ';'
            if ";" in rest:
                _flush_net()
            continue

        # Continuation lines: may contain more ( inst pin ) tuples, and end with ';'
        for tup in re.findall(r"\(\s*\S+\s+\S+\s*\)", ln):
            mc = re_conn.match(tup)
            if mc:
                cur_pins.append((mc.group(1), mc.group(2)))
        if ";" in ln:
            _flush_net()

    return die, comps, pins, nets

