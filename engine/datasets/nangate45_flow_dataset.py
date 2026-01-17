"""
NanGate45 MacroPlacement flow dataset -> DiffPlace PyG Data.

Supports:
- OpenROAD CodeElement clustered DEF+LEF (preferred): provides NETS/PINS and cluster sizes.
- Cadence/OpenROAD floorplan DEF without NETS: fallback to macro-only connectivity via Verilog (best-effort).
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .lefdef_parser import (
    collect_lef_files,
    parse_lef_macro_sizes,
    parse_def,
    DefComponent,
    DefPin,
    DefNet,
)


def _normalize_positions_centers(
    centers_xy: torch.Tensor,
    die_min: torch.Tensor,
    die_size: torch.Tensor,
) -> torch.Tensor:
    # centers_xy: (V,2) microns
    return (centers_xy - die_min) / die_size * 2.0 - 1.0


def _normalize_sizes_for_model(
    sizes_wh: torch.Tensor,
    die_size: torch.Tensor,
    model_size_scale: float,
) -> torch.Tensor:
    # sizes_wh: (V,2) microns
    # return scaled sizes in same normalized coordinate system as pos ([-1,1]) * scale
    s = sizes_wh / die_size * 2.0
    return s * float(model_size_scale)


def _build_star_edges_from_nets(
    nets: List[DefNet],
    name_to_idx: Dict[str, int],
) -> torch.Tensor:
    src: List[int] = []
    dst: List[int] = []
    for net in nets:
        # resolve nodes participating in this net
        nodes = []
        for inst, _pin in net.pins:
            if inst == "PIN":
                # pin name will be in _pin; in DEF syntax it's ( PIN <pinname> )
                pin_name = _pin
                if pin_name in name_to_idx:
                    nodes.append(name_to_idx[pin_name])
            else:
                if inst in name_to_idx:
                    nodes.append(name_to_idx[inst])
        nodes = list(dict.fromkeys(nodes))  # de-dup keep order
        if len(nodes) < 2:
            continue
        root = nodes[0]
        for i in range(1, len(nodes)):
            u = root
            v = nodes[i]
            src.extend([u, v])
            dst.extend([v, u])
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def _build_net_structure(
    nets: List[DefNet],
    name_to_idx: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build net_to_pin, pin_to_macro, pin_offsets compatible with Trinity guidance.
    Pin offsets are 0 (no LEF pin geometry available from DEF in this pipeline).
    """
    all_pins: List[Tuple[int, float, float]] = []
    net_pins: List[List[int]] = []
    max_pins = 0

    for net in nets:
        pins = []
        for inst, pin in net.pins:
            if inst == "PIN":
                node_name = pin
            else:
                node_name = inst
            if node_name in name_to_idx:
                pins.append(name_to_idx[node_name])

        pins = list(dict.fromkeys(pins))
        if len(pins) < 2:
            continue

        start_idx = len(all_pins)
        for macro_idx in pins:
            all_pins.append((macro_idx, 0.0, 0.0))
        net_pins.append(list(range(start_idx, start_idx + len(pins))))
        max_pins = max(max_pins, len(pins))

    if not net_pins:
        # dummy structure
        return (
            torch.zeros((1, 2), dtype=torch.long),
            torch.zeros((2,), dtype=torch.long),
            torch.zeros((2, 2), dtype=torch.float32),
        )

    num_nets = len(net_pins)
    net_to_pin = torch.full((num_nets, max_pins), -1, dtype=torch.long)
    for i, pins in enumerate(net_pins):
        net_to_pin[i, : len(pins)] = torch.tensor(pins, dtype=torch.long)

    num_pins = len(all_pins)
    pin_to_macro = torch.zeros((num_pins,), dtype=torch.long)
    pin_offsets = torch.zeros((num_pins, 2), dtype=torch.float32)
    for i, (macro_idx, offx, offy) in enumerate(all_pins):
        pin_to_macro[i] = int(macro_idx)
        pin_offsets[i, 0] = float(offx)
        pin_offsets[i, 1] = float(offy)

    return net_to_pin, pin_to_macro, pin_offsets


def _infer_design_inputs(design_dir: str) -> Tuple[str, Optional[str]]:
    """
    Returns (def_path, optional_lef_dir_hint).
    Prefer clustered netlist (has NETS). Fall back to *_fp_placed_macros.def.
    """
    clustered_def = os.path.join(
        design_dir, "netlist", "output_CodeElement", "OpenROAD", "clustered_netlist.def"
    )
    if os.path.isfile(clustered_def):
        return clustered_def, os.path.join(design_dir, "netlist", "output_CodeElement", "OpenROAD")

    # Common floorplan defs (no nets): generic search
    cand: List[str] = []
    def_dir = os.path.join(design_dir, "def")
    if os.path.isdir(def_dir):
        for fn in os.listdir(def_dir):
            if fn.endswith("_fp_placed_macros.def") or fn.endswith("_fp.def"):
                cand.append(os.path.join(def_dir, fn))
    for p in cand:
        if os.path.isfile(p):
            return p, None
    raise FileNotFoundError(f"Could not find DEF in {design_dir}")


def _strip_verilog_comments(lines: List[str]) -> List[str]:
    out: List[str] = []
    in_block = False
    for ln in lines:
        s = ln
        if in_block:
            if "*/" in s:
                s = s.split("*/", 1)[1]
                in_block = False
            else:
                continue
        if "/*" in s:
            before, after = s.split("/*", 1)
            s = before
            in_block = True
        s = re.sub(r"//.*$", "", s)
        if s.strip():
            out.append(s)
    return out


def _parse_verilog_macro_nets(
    verilog_paths: List[str],
    macro_celltypes: Optional[set] = None,
) -> List[DefNet]:
    """
    Best-effort structural Verilog parser to extract macro-to-macro netlist.
    Returns DefNet-like objects with pins=(inst, pinname) where pinname is placeholder.
    """
    macro_celltypes = macro_celltypes or set()
    keyword = {
        "module", "endmodule", "assign", "wire", "reg", "input", "output", "inout",
        "parameter", "localparam", "generate", "endgenerate", "if", "else", "for",
        "begin", "end", "always", "initial", "case", "endcase",
    }

    net_to_insts: Dict[str, List[str]] = {}

    # inst header: <celltype> <instname> (
    re_inst_start = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s+(\S+)\s*\(\s*$")
    # accept both normal and escaped identifiers for net names
    re_conn = re.compile(r"\.\s*([A-Za-z_][A-Za-z0-9_$]*)\s*\(\s*(\\?[A-Za-z_][A-Za-z0-9_\\\[\]\./$]*)\s*\)")

    def is_macro(celltype: str) -> bool:
        if celltype in macro_celltypes:
            return True
        if celltype.lower().startswith("fakeram45_"):
            return True
        return False

    for vp in verilog_paths:
        if not os.path.isfile(vp):
            continue
        try:
            with open(vp, "r", errors="ignore") as f:
                raw = f.readlines()
        except Exception:
            continue
        lines = _strip_verilog_comments(raw)

        in_inst = False
        celltype = ""
        instname = ""
        buf: List[str] = []
        pending_celltype: Optional[str] = None
        pending_inst: Optional[str] = None

        for ln in lines:
            s = ln.strip()
            if not in_inst:
                # Pattern A: single-line header "celltype instname ("
                m = re_inst_start.match(s)
                if m:
                    ct = m.group(1)
                    if ct not in keyword:
                        celltype = ct
                        instname = m.group(2)
                        if instname.startswith("\\"):
                            instname = instname[1:]
                        buf = []
                        in_inst = True
                        continue

                # Pattern B: 3-line header:
                #   celltype
                #   instname
                #   (
                if pending_celltype is None:
                    # candidate celltype line: a single token
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", s) and (s not in keyword):
                        pending_celltype = s
                    continue

                if pending_celltype is not None and pending_inst is None:
                    # candidate instname line: single token (may be escaped)
                    if re.fullmatch(r"\\?\S+", s) and s not in ("(", ")", ");"):
                        pending_inst = s
                    else:
                        pending_celltype = None
                    continue

                if pending_celltype is not None and pending_inst is not None:
                    if s == "(":
                        celltype = pending_celltype
                        instname = pending_inst[1:] if pending_inst.startswith("\\") else pending_inst
                        pending_celltype = None
                        pending_inst = None
                        buf = []
                        in_inst = True
                        continue
                    # reset if unexpected
                    pending_celltype = None
                    pending_inst = None
                    continue

            else:
                buf.append(s)
                if ");" not in s:
                    continue
                text = " ".join(buf)
                in_inst = False

                if not is_macro(celltype):
                    continue
                for _pm, net in re_conn.findall(text):
                    if net.startswith("1'") or net.startswith("0'"):
                        continue
                    if net.startswith("\\"):
                        net = net[1:]
                    net_to_insts.setdefault(net, []).append(instname)

    nets: List[DefNet] = []
    for net, insts in net_to_insts.items():
        uniq = list(dict.fromkeys(insts))
        if len(uniq) < 2:
            continue
        pins = [(inst, "P") for inst in uniq]
        nets.append(DefNet(name=net, pins=pins))
    return nets


class NanGate45FlowParser:
    def __init__(
        self,
        design_dir: str,
        enablements_lef_dir: str,
        model_size_scale: float = 0.1,
        macro_height_threshold_in_rows: float = 1.0,
    ):
        self.design_dir = design_dir
        self.enablements_lef_dir = enablements_lef_dir
        self.model_size_scale = float(model_size_scale)
        self.macro_height_threshold_in_rows = float(macro_height_threshold_in_rows)

    def to_pyg_data(self, max_nodes: Optional[int] = None) -> Data:
        def_path, lef_hint = _infer_design_inputs(self.design_dir)

        extra_roots = [self.enablements_lef_dir]
        if lef_hint is not None:
            extra_roots.append(lef_hint)
        lefs = collect_lef_files(self.design_dir, extra_roots=extra_roots)
        lef_sizes = parse_lef_macro_sizes(lefs)

        die, comps, pins, nets = parse_def(def_path)

        # If DEF has no NETS (common for floorplan-only DEFs), fall back to
        # macro-only connectivity from structural Verilog.
        if not nets:
            verilog_candidates: List[str] = []
            for sub in ("netlist", "rtl", os.path.join("scripts", "cadence", "rtl")):
                p = os.path.join(self.design_dir, sub)
                if not os.path.isdir(p):
                    continue
                for dirpath, _, filenames in os.walk(p):
                    for fn in filenames:
                        if fn.lower().endswith(".v"):
                            verilog_candidates.append(os.path.join(dirpath, fn))
            # Restrict to macro modules we know sizes for (plus fakeram fallback)
            macro_celltypes = set(lef_sizes.keys())
            nets = _parse_verilog_macro_nets(verilog_candidates, macro_celltypes=macro_celltypes)

        # Decide node set: macros (from COMPONENTS) + top-level PINS.
        # Filter macros by LEF size existence and height threshold vs row_height.
        macro_nodes: List[Tuple[str, str, float, float, str, bool]] = []
        for c in comps:
            if c.macro not in lef_sizes:
                continue
            w, h = lef_sizes[c.macro]
            if h <= die.row_height * self.macro_height_threshold_in_rows:
                continue
            macro_nodes.append((c.name, c.macro, c.x, c.y, c.orient, c.fixed))

        # Add ports as nodes
        port_nodes: List[DefPin] = pins

        if max_nodes is not None and len(macro_nodes) > max_nodes:
            macro_nodes = macro_nodes[: int(max_nodes)]

        # Build mapping
        names: List[str] = [n[0] for n in macro_nodes] + [p.name for p in port_nodes]
        name_to_idx = {n: i for i, n in enumerate(names)}
        V = len(names)
        if V == 0:
            raise ValueError(f"No macros/pins parsed from {def_path}")

        # Build sizes and centers in microns
        sizes = torch.zeros((V, 2), dtype=torch.float32)
        centers = torch.zeros((V, 2), dtype=torch.float32)
        is_ports = torch.zeros((V,), dtype=torch.bool)
        rot_label = torch.zeros((V,), dtype=torch.long)

        orient_map = {"N": 0, "E": 1, "S": 2, "W": 3, "FN": 0, "FS": 2, "FE": 1, "FW": 3}

        for i, (inst, macro, x, y, orient, _fixed) in enumerate(macro_nodes):
            w, h = lef_sizes[macro]
            sizes[i] = torch.tensor([w, h], dtype=torch.float32)
            centers[i] = torch.tensor([x + w / 2.0, y + h / 2.0], dtype=torch.float32)
            rot_label[i] = orient_map.get(orient, 0)

        base = len(macro_nodes)
        for j, p in enumerate(port_nodes):
            idx = base + j
            sizes[idx] = torch.tensor([0.0, 0.0], dtype=torch.float32)
            centers[idx] = torch.tensor([p.x, p.y], dtype=torch.float32)
            is_ports[idx] = True

        die_min = torch.tensor([die.x_min, die.y_min], dtype=torch.float32)
        die_size = torch.tensor([die.width, die.height], dtype=torch.float32).clamp(min=1e-6)

        pos = _normalize_positions_centers(centers, die_min, die_size)
        x_sizes = _normalize_sizes_for_model(sizes, die_size, self.model_size_scale)
        overlap_sizes = _normalize_sizes_for_model(sizes, die_size, 1.0)

        # Connectivity
        edge_index = _build_star_edges_from_nets(nets, name_to_idx)
        edge_attr = torch.zeros((edge_index.shape[1], 4), dtype=torch.float32)

        net_to_pin, pin_to_macro, pin_offsets = _build_net_structure(nets, name_to_idx)

        data = Data(
            pos=pos,
            x=x_sizes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            is_ports=is_ports,
            rot_label=rot_label,
            net_to_pin=net_to_pin,
            pin_to_macro=pin_to_macro,
            pin_offsets=pin_offsets,
            num_nodes=V,
            benchmark=os.path.basename(self.design_dir.rstrip("/")),
        )
        data.overlap_sizes = overlap_sizes
        data.node_names = names  # list[str], same order as tensors
        data.is_macro = (~is_ports).clone()
        data.sizes_microns = sizes
        data.die_info = torch.tensor(
            [die.x_min, die.y_min, die.x_max, die.y_max, die.row_height, die.site_width],
            dtype=torch.float32,
        )
        data.source_def = def_path
        return data


class NanGate45FlowDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        designs: Optional[List[str]] = None,
        enablements_lef_dir: Optional[str] = None,
        max_nodes: Optional[int] = None,
        model_size_scale: float = 0.1,
        macro_height_threshold_in_rows: float = 1.0,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.designs = designs or []
        self.max_nodes = max_nodes
        self.enablements_lef_dir = enablements_lef_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(root_dir))),
            "Enablements",
            "NanGate45",
            "lef",
        )
        self.model_size_scale = model_size_scale
        self.macro_height_threshold_in_rows = macro_height_threshold_in_rows

        if not self.designs:
            # infer designs as subdirectories under root_dir
            self.designs = [
                d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
            ]

    def __len__(self) -> int:
        return len(self.designs)

    def __getitem__(self, idx: int) -> Data:
        design = self.designs[idx]
        design_dir = os.path.join(self.root_dir, design)
        parser = NanGate45FlowParser(
            design_dir=design_dir,
            enablements_lef_dir=self.enablements_lef_dir,
            model_size_scale=self.model_size_scale,
            macro_height_threshold_in_rows=self.macro_height_threshold_in_rows,
        )
        return parser.to_pyg_data(max_nodes=self.max_nodes)

