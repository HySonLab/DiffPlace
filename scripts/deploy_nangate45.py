#!/usr/bin/env python

import os
import argparse
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.diffplace import DiffPlace
from engine.diffusion.overlap_loss import OverlapLoss
from engine.datasets.nangate45_flow_dataset import NanGate45FlowParser


@dataclass
class Macro:
    name: str
    x: float
    y: float
    w: float
    h: float
    orient: str = "N"


@dataclass
class DieBounds:
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


def calculate_overlap_area(macros: List[Macro]) -> float:
    total = 0.0
    for i, m1 in enumerate(macros):
        for m2 in macros[i + 1 :]:
            ox = max(0.0, min(m1.x + m1.w, m2.x + m2.w) - max(m1.x, m2.x))
            oy = max(0.0, min(m1.y + m1.h, m2.y + m2.h) - max(m1.y, m2.y))
            total += ox * oy
    return total


class TetrisLegalizer:
    def __init__(
        self,
        die_bounds: DieBounds,
        grid_resolution: float,
        max_radius: Optional[int] = None,
        fallback_scan: bool = True,
        pick_best: bool = True,
        strict: bool = True,
    ):
        self.die = die_bounds
        self.grid_res = float(grid_resolution)
        self.grid_w = int(np.ceil(die_bounds.width / self.grid_res))
        self.grid_h = int(np.ceil(die_bounds.height / self.grid_res))
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        self.max_radius = max_radius if max_radius is not None else max(self.grid_w, self.grid_h)
        self.fallback_scan = fallback_scan
        self.pick_best = pick_best
        self.strict = strict

    def _to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x - self.die.x_min) / self.grid_res)
        gy = int((y - self.die.y_min) / self.grid_res)
        gx = max(0, min(self.grid_w - 1, gx))
        gy = max(0, min(self.grid_h - 1, gy))
        return gx, gy

    def _mark_occupied(self, x: float, y: float, w: float, h: float):
        x1, y1 = self._to_grid(x, y)
        x2, y2 = self._to_grid(x + w, y + h)
        self.grid[y1 : y2 + 1, x1 : x2 + 1] = True

    def _check_overlap(self, x: float, y: float, w: float, h: float) -> bool:
        if x < self.die.x_min or x + w > self.die.x_max:
            return True
        if y < self.die.y_min or y + h > self.die.y_max:
            return True
        x1, y1 = self._to_grid(x, y)
        x2, y2 = self._to_grid(x + w, y + h)
        return self.grid[y1 : y2 + 1, x1 : x2 + 1].any()

    def _snap(self, x: float, y: float) -> Tuple[float, float]:
        x_snap = self.die.x_min + round((x - self.die.x_min) / self.die.site_width) * self.die.site_width
        y_snap = self.die.y_min + round((y - self.die.y_min) / self.die.row_height) * self.die.row_height
        return x_snap, y_snap

    def _find_valid_position(self, x: float, y: float, w: float, h: float) -> Tuple[float, float]:
        x, y = self._snap(x, y)
        x = max(self.die.x_min, min(self.die.x_max - w, x))
        y = max(self.die.y_min, min(self.die.y_max - h, y))

        if not self._check_overlap(x, y, w, h):
            return x, y

        step_x = max(self.die.site_width, 1e-6)
        step_y = max(self.die.row_height, 1e-6)
        best = None
        best_d2 = float("inf")

        for radius in range(1, int(self.max_radius) + 1):
            for dx in range(-radius, radius + 1):
                dy_abs = radius - abs(dx)
                for dy in (-dy_abs, dy_abs) if dy_abs != 0 else (0,):
                    nx = x + dx * step_x
                    ny = y + dy * step_y
                    nx, ny = self._snap(nx, ny)
                    nx = max(self.die.x_min, min(self.die.x_max - w, nx))
                    ny = max(self.die.y_min, min(self.die.y_max - h, ny))
                    if not self._check_overlap(nx, ny, w, h):
                        if not self.pick_best:
                            return nx, ny
                        d2 = (nx - x) ** 2 + (ny - y) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best = (nx, ny)
            if best is not None:
                return best

        if self.fallback_scan:
            gx0, gy0 = self._to_grid(x, y)
            best = None
            best_d2 = float("inf")
            max_r = max(self.grid_w, self.grid_h)
            for r in range(0, max_r + 1):
                for gx in range(max(0, gx0 - r), min(self.grid_w, gx0 + r + 1)):
                    for gy in (gy0 - r, gy0 + r):
                        if gy < 0 or gy >= self.grid_h:
                            continue
                        nx = self.die.x_min + gx * self.grid_res
                        ny = self.die.y_min + gy * self.grid_res
                        nx, ny = self._snap(nx, ny)
                        nx = max(self.die.x_min, min(self.die.x_max - w, nx))
                        ny = max(self.die.y_min, min(self.die.y_max - h, ny))
                        if not self._check_overlap(nx, ny, w, h):
                            d2 = (nx - x) ** 2 + (ny - y) ** 2
                            if d2 < best_d2:
                                best_d2 = d2
                                best = (nx, ny)
                for gy in range(max(0, gy0 - r + 1), min(self.grid_h, gy0 + r)):
                    for gx in (gx0 - r, gx0 + r):
                        if gx < 0 or gx >= self.grid_w:
                            continue
                        nx = self.die.x_min + gx * self.grid_res
                        ny = self.die.y_min + gy * self.grid_res
                        nx, ny = self._snap(nx, ny)
                        nx = max(self.die.x_min, min(self.die.x_max - w, nx))
                        ny = max(self.die.y_min, min(self.die.y_max - h, ny))
                        if not self._check_overlap(nx, ny, w, h):
                            d2 = (nx - x) ** 2 + (ny - y) ** 2
                            if d2 < best_d2:
                                best_d2 = d2
                                best = (nx, ny)
                if best is not None:
                    return best

        return x, y

    def legalize(self, macros: List[Macro]) -> List[Macro]:
        sorted_macros = sorted(macros, key=lambda m: m.w * m.h, reverse=True)
        out: List[Macro] = []
        failed = 0
        for m in sorted_macros:
            x, y = self._find_valid_position(m.x, m.y, m.w, m.h)
            if not self._check_overlap(x, y, m.w, m.h):
                self._mark_occupied(x, y, m.w, m.h)
            else:
                failed += 1
                if self.strict:
                    raise RuntimeError(f"TetrisLegalizer failed macro={m.name}")
            out.append(Macro(name=m.name, x=x, y=y, w=m.w, h=m.h, orient=m.orient))
        return out


def load_model(checkpoint_path: str, device: str) -> DiffPlace:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    model = DiffPlace(
        hidden_size=model_cfg.get("hidden_size", 256),
        num_blocks=model_cfg.get("num_blocks", 8),
        layers_per_block=model_cfg.get("layers_per_block", 2),
        num_heads=model_cfg.get("num_heads", 8),
        global_context_every=model_cfg.get("global_context_every", 2),
        gradient_checkpointing=False,
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--design_dir", type=str, required=True)
    p.add_argument("--enablements_lef_dir", type=str, default="/home/hinhnv/kienle/chip/MacroPlacement/Enablements/NanGate45/lef")
    p.add_argument("--output_pkl", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--ddim_steps", type=int, default=100)
    p.add_argument("--eta", type=float, default=0.0)

    p.add_argument("--overlap_guidance_scale_max", type=float, default=200.0)
    p.add_argument("--overlap_guidance_power", type=float, default=2.0)
    p.add_argument("--overlap_guidance_start_frac", type=float, default=0.85)
    p.add_argument("--overlap_guidance_grad_norm", type=float, default=0.03)
    p.add_argument("--overlap_guidance_grad_clip", type=float, default=0.08)

    p.add_argument("--overlap_refine_steps", type=int, default=500)
    p.add_argument("--overlap_refine_lr", type=float, default=0.04)
    p.add_argument("--overlap_refine_anchor_weight_start", type=float, default=0.01)
    p.add_argument("--overlap_refine_anchor_weight_end", type=float, default=0.30)

    p.add_argument("--tetris_grid_resolution", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = load_model(args.checkpoint, device)

    parser = NanGate45FlowParser(
        design_dir=args.design_dir,
        enablements_lef_dir=args.enablements_lef_dir,
        model_size_scale=0.1,
        macro_height_threshold_in_rows=1.0,
    )
    data = parser.to_pyg_data()
    data = data.to(device)

    with torch.no_grad():
        x, rot_onehot, _ = model.sample_ddim(
            cond=data,
            batch_size=1,
            num_inference_steps=args.ddim_steps,
            eta=args.eta,
            guidance_scale=0.0,
            overlap_guidance=True,
            overlap_guidance_scale_max=args.overlap_guidance_scale_max,
            overlap_guidance_power=args.overlap_guidance_power,
            overlap_sizes=data.overlap_sizes,
            overlap_guidance_start_frac=args.overlap_guidance_start_frac,
            overlap_guidance_grad_norm=args.overlap_guidance_grad_norm,
            overlap_guidance_grad_clip=args.overlap_guidance_grad_clip,
            return_intermediates=False,
        )

    # Optional anchored refinement (still silent)
    if args.overlap_refine_steps > 0:
        overlap_loss_fn = OverlapLoss().to(device)
        x0 = x.detach()
        x_ref = x0.clone().detach()

        fixed_pos = data.pos.unsqueeze(0)[:, :, :2]
        fixed_mask_3d = data.is_ports.unsqueeze(0).unsqueeze(-1).expand_as(x_ref)
        macro_mask = data.is_macro  # (V,)

        for k in range(int(args.overlap_refine_steps)):
            x_ref = x_ref.detach().requires_grad_(True)
            loss_ov = overlap_loss_fn(x_ref, data.overlap_sizes, macro_mask=macro_mask)
            loss_anchor = F.mse_loss(x_ref, x0)
            if args.overlap_refine_steps <= 1:
                aw = float(args.overlap_refine_anchor_weight_end)
            else:
                p = float(k) / float(args.overlap_refine_steps - 1)
                aw = (1.0 - p) * float(args.overlap_refine_anchor_weight_start) + p * float(args.overlap_refine_anchor_weight_end)
            loss = loss_ov + aw * loss_anchor
            g = torch.autograd.grad(loss, x_ref, retain_graph=False, create_graph=False)[0]
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            x_ref = (x_ref - float(args.overlap_refine_lr) * g).detach().clamp(-1.0, 1.0)
            x_ref = torch.where(fixed_mask_3d, fixed_pos, x_ref)

        x = x_ref

    # Convert to die coords (centers -> lower-left) and strict tetris
    die_info = data.die_info.detach().cpu().numpy().tolist()
    die = DieBounds(
        x_min=die_info[0],
        y_min=die_info[1],
        x_max=die_info[2],
        y_max=die_info[3],
        row_height=die_info[4],
        site_width=die_info[5],
    )

    node_names: List[str] = list(data.node_names)
    sizes_microns = data.sizes_microns.detach().cpu().numpy()
    is_macro = data.is_macro.detach().cpu().numpy().astype(bool)

    x_np = x.squeeze(0).detach().cpu().numpy()  # normalized centers
    centers = np.zeros_like(x_np)
    centers[:, 0] = (x_np[:, 0] + 1.0) / 2.0 * die.width + die.x_min
    centers[:, 1] = (x_np[:, 1] + 1.0) / 2.0 * die.height + die.y_min

    macros: List[Macro] = []
    for i, name in enumerate(node_names):
        if not is_macro[i]:
            continue
        w = float(sizes_microns[i, 0])
        h = float(sizes_microns[i, 1])
        cx = float(centers[i, 0])
        cy = float(centers[i, 1])
        llx = max(die.x_min, min(die.x_max - w, cx - w / 2.0))
        lly = max(die.y_min, min(die.y_max - h, cy - h / 2.0))
        macros.append(Macro(name=name, x=llx, y=lly, w=w, h=h, orient="N"))

    tetris = TetrisLegalizer(
        die_bounds=die,
        grid_resolution=float(args.tetris_grid_resolution),
        fallback_scan=True,
        pick_best=True,
        strict=True,
    )
    macros_legal = tetris.legalize(macros)
    if calculate_overlap_area(macros_legal) > 0.0:
        raise RuntimeError("Non-zero overlap after strict legalization")

    out: Dict = {
        "name": os.path.basename(args.design_dir.rstrip("/")),
        "die_area": {"width": die.width, "height": die.height, "x_min": die.x_min, "y_min": die.y_min},
        "macros": {m.name: {"x": m.x, "y": m.y, "w": m.w, "h": m.h, "orient": m.orient} for m in macros_legal},
        "num_macros": len(macros_legal),
        "source_def": getattr(data, "source_def", ""),
    }

    os.makedirs(os.path.dirname(args.output_pkl) or ".", exist_ok=True)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()

