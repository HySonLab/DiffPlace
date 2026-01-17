#!/usr/bin/env python
import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import warnings

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("cvxpy not installed. SequencePairLegalizer will use fallback mode.")


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


class SequencePairLegalizer:
    
    def __init__(
        self,
        die_bounds: DieBounds,
        num_iterations: int = 1,
        gap_margin: float = 0.0,
        verbose: bool = True,
        lp_timeout: float = 60.0,
    ):
        self.die = die_bounds
        self.num_iterations = num_iterations
        self.gap_margin = gap_margin
        self.verbose = verbose
        self.lp_timeout = lp_timeout
    
    def legalize(self, macros: List[Macro]) -> List[Macro]:
        N = len(macros)
        if N == 0:
            return []
        
        if N == 1:
            m = macros[0]
            x = np.clip(m.x, self.die.x_min, self.die.x_max - m.w)
            y = np.clip(m.y, self.die.y_min, self.die.y_max - m.h)
            return [Macro(m.name, x, y, m.w, m.h, m.orient)]
        
        if self.verbose:
            pass
        
        positions = np.array([[m.x, m.y] for m in macros], dtype=np.float64)
        sizes = np.array([[m.w, m.h] for m in macros], dtype=np.float64)
        names = [m.name for m in macros]
        orients = [m.orient for m in macros]
        
        for iteration in range(self.num_iterations):
            if self.verbose:
                pass
            
            sp_plus, sp_minus = self._extract_sequence_pair(positions, sizes)
            
            h_graph = self._build_tcg_horizontal(sp_plus, sp_minus, N)
            v_graph = self._build_tcg_vertical(sp_plus, sp_minus, N)
            
            h_graph = self._transitive_reduction(h_graph, N)
            v_graph = self._transitive_reduction(v_graph, N)
            
            if CVXPY_AVAILABLE:
                try:
                    positions[:, 0] = self._solve_lp_axis(
                        h_graph, positions[:, 0], sizes[:, 0], 
                        self.die.x_min, self.die.x_max
                    )
                    positions[:, 1] = self._solve_lp_axis(
                        v_graph, positions[:, 1], sizes[:, 1],
                        self.die.y_min, self.die.y_max
                    )
                except Exception as e:
                    if self.verbose:
                        pass
                    positions[:, 0] = self._solve_longest_path(h_graph, positions[:, 0], sizes[:, 0], self.die.x_min, self.die.x_max)
                    positions[:, 1] = self._solve_longest_path(v_graph, positions[:, 1], sizes[:, 1], self.die.y_min, self.die.y_max)
            else:
                positions[:, 0] = self._solve_longest_path(h_graph, positions[:, 0], sizes[:, 0], self.die.x_min, self.die.x_max)
                positions[:, 1] = self._solve_longest_path(v_graph, positions[:, 1], sizes[:, 1], self.die.y_min, self.die.y_max)
            
            overlap_count = self._count_overlaps(positions, sizes)
            if self.verbose:
                pass
            
            if overlap_count == 0:
                break
        
        positions = self._snap_to_grid(positions, sizes)
        
        result = []
        for i in range(N):
            result.append(Macro(
                name=names[i],
                x=positions[i, 0],
                y=positions[i, 1],
                w=sizes[i, 0],
                h=sizes[i, 1],
                orient=orients[i],
            ))
        
        if self.verbose:
            final_overlaps = self._count_overlaps(positions, sizes)
            pass
        
        return result
    
    def _extract_sequence_pair(
        self, 
        positions: np.ndarray, 
        sizes: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Extract Sequence Pair from current positions"""
        N = len(positions)
        centers = positions + sizes / 2
        
        sum_coords = centers[:, 0] + centers[:, 1]
        diff_coords = centers[:, 0] - centers[:, 1]
        
        sp_plus = list(np.argsort(sum_coords))
        sp_minus = list(np.argsort(diff_coords))
        
        return sp_plus, sp_minus
    
    def _build_tcg_horizontal(
        self, 
        sp_plus: List[int], 
        sp_minus: List[int],
        N: int
    ) -> Dict[int, Set[int]]:
        """
        Build Horizontal TCG (Left-of constraints)
        
        Rule: i Left-of j if:
          S+(i) < S+(j) AND S-(i) < S-(j)
        """
        graph = defaultdict(set)
        
        pos_in_sp_plus = {macro: idx for idx, macro in enumerate(sp_plus)}
        pos_in_sp_minus = {macro: idx for idx, macro in enumerate(sp_minus)}
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                if (pos_in_sp_plus[i] < pos_in_sp_plus[j] and 
                    pos_in_sp_minus[i] < pos_in_sp_minus[j]):
                    graph[i].add(j)
        
        return graph
    
    def _build_tcg_vertical(
        self, 
        sp_plus: List[int], 
        sp_minus: List[int],
        N: int
    ) -> Dict[int, Set[int]]:
        """
        Build Vertical TCG (Below-of constraints)
        
        Rule: i Below j if:
          S+(i) > S+(j) AND S-(i) < S-(j)
        """
        graph = defaultdict(set)
        
        pos_in_sp_plus = {macro: idx for idx, macro in enumerate(sp_plus)}
        pos_in_sp_minus = {macro: idx for idx, macro in enumerate(sp_minus)}
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                if (pos_in_sp_plus[i] > pos_in_sp_plus[j] and 
                    pos_in_sp_minus[i] < pos_in_sp_minus[j]):
                    graph[i].add(j)
        
        return graph
    
    def _transitive_reduction(
        self, 
        graph: Dict[int, Set[int]],
        N: int
    ) -> Dict[int, Set[int]]:
        """
        Transitive Reduction: Remove redundant edges
        
        If i -> k exists and there's a path i -> j -> k, remove i -> k
        This reduces graph density from O(N^2) to O(N)
        """
        reduced = {i: set(graph[i]) for i in range(N)}
        
        for i in range(N):
            reachable = set()
            for j in graph[i]:
                reachable.update(self._dfs_reachable(graph, j))
            
            for k in reachable:
                reduced[i].discard(k)
        
        return reduced
    
    def _dfs_reachable(
        self, 
        graph: Dict[int, Set[int]], 
        start: int
    ) -> Set[int]:
        """DFS to find all reachable nodes from start"""
        visited = set()
        stack = [start]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(graph.get(node, []))
        
        return visited
    
    def _solve_lp_axis(
        self,
        graph: Dict[int, Set[int]],
        current_coords: np.ndarray,
        widths: np.ndarray,
        die_min: float,
        die_max: float,
    ) -> np.ndarray:
        """
        Solve LP for one axis
        
        Minimize: sum |x_new - x_old|  (L1 displacement)
        Subject to:
          - die_min <= x_new[i] <= die_max - width[i]
          - x_new[j] >= x_new[i] + width[i] + gap  (for edge i->j)
        """
        N = len(current_coords)
        
        x = cp.Variable(N)
        
        displacement = cp.sum(cp.abs(x - current_coords))
        objective = cp.Minimize(displacement)
        
        constraints = []
        
        constraints.append(x >= die_min)
        constraints.append(x + widths <= die_max)
        
        for i in range(N):
            for j in graph.get(i, []):
                constraints.append(x[j] >= x[i] + widths[i] + self.gap_margin)
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-4, eps_rel=1e-4, max_iter=10000)
        except:
            try:
                problem.solve(solver=cp.ECOS, verbose=False)
            except:
                problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError(f"LP failed: {problem.status}")
        
        result = x.value
        if result is None:
            raise RuntimeError("LP returned None")
        
        return np.clip(result, die_min, die_max - widths)
    
    def _solve_longest_path(
        self,
        graph: Dict[int, Set[int]],
        current_coords: np.ndarray,
        widths: np.ndarray,
        die_min: float,
        die_max: float,
    ) -> np.ndarray:
        """
        Fallback: Solve using Longest Path on DAG
        
        This satisfies all constraints but may have higher displacement than LP
        """
        N = len(current_coords)
        
        in_degree = np.zeros(N, dtype=np.int32)
        for i in range(N):
            for j in graph.get(i, []):
                in_degree[j] += 1
        
        pos = current_coords.copy()
        queue = deque([i for i in range(N) if in_degree[i] == 0])
        
        while queue:
            u = queue.popleft()
            
            for v in graph.get(u, []):
                required_pos = pos[u] + widths[u] + self.gap_margin
                pos[v] = max(pos[v], required_pos)
                
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        for i in range(N):
            pos[i] = np.clip(pos[i], die_min, die_max - widths[i])
        
        if pos.max() + widths[pos.argmax()] > die_max:
            total_width = np.sum(widths) + (N - 1) * self.gap_margin
            available_space = die_max - die_min
            
            if total_width < available_space:
                scale = available_space / (pos.max() - pos.min() + widths[pos.argmax()])
                pos = die_min + (pos - pos.min()) * scale * 0.95
        
        return np.clip(pos, die_min, die_max - widths)
    
    def _count_overlaps(self, positions: np.ndarray, sizes: np.ndarray) -> int:
        N = len(positions)
        count = 0
        
        for i in range(N):
            for j in range(i + 1, N):
                x_overlap = max(0, min(positions[i, 0] + sizes[i, 0], positions[j, 0] + sizes[j, 0]) - 
                               max(positions[i, 0], positions[j, 0]))
                y_overlap = max(0, min(positions[i, 1] + sizes[i, 1], positions[j, 1] + sizes[j, 1]) - 
                               max(positions[i, 1], positions[j, 1]))
                
                if x_overlap > 0 and y_overlap > 0:
                    count += 1
        
        return count
    
    def _snap_to_grid(self, positions: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        result = positions.copy()
        
        for i in range(len(positions)):
            x = positions[i, 0]
            x_snapped = self.die.x_min + round((x - self.die.x_min) / self.die.site_width) * self.die.site_width
            
            y = positions[i, 1]
            y_snapped = self.die.y_min + round((y - self.die.y_min) / self.die.row_height) * self.die.row_height
            
            result[i, 0] = np.clip(x_snapped, self.die.x_min, self.die.x_max - sizes[i, 0])
            result[i, 1] = np.clip(y_snapped, self.die.y_min, self.die.y_max - sizes[i, 1])
        
        return result
