#!/usr/bin/env python3
"""
Enhanced ARC-AGI Solver with expanded operator library and improved LLM integration.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import copy

try:
    from openai import OpenAI
except ImportError:
    print("Install: pip install openai numpy --break-system-packages")
    raise


# ============================================================================
# WORKSPACE PRIMITIVES
# ============================================================================

@dataclass
class Object:
    """Workspace object with structural properties"""
    positions: Set[Tuple[int, int]]
    color: int

    @property
    def size(self) -> int:
        return len(self.positions)

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        if not self.positions:
            return (0, 0, 0, 0)
        xs = [p[1] for p in self.positions]
        ys = [p[0] for p in self.positions]
        return (min(ys), min(xs), max(ys), max(xs))

    def centroid(self) -> Tuple[float, float]:
        if not self.positions:
            return (0.0, 0.0)
        ys = [p[0] for p in self.positions]
        xs = [p[1] for p in self.positions]
        return (sum(ys) / len(ys), sum(xs) / len(xs))

    def to_grid(self) -> np.ndarray:
        """Extract object as minimal grid"""
        y1, x1, y2, x2 = self.bounding_box()
        h, w = y2 - y1 + 1, x2 - x1 + 1
        grid = np.zeros((h, w), dtype=int)
        for y, x in self.positions:
            grid[y - y1, x - x1] = self.color
        return grid


class OperatorState(Enum):
    SHADOW = "shadow"
    PROBATIONARY = "probationary"
    LOCKED = "locked"


@dataclass
class Operator:
    name: str
    description: str
    state: OperatorState = OperatorState.SHADOW
    mdl_credit: float = 0.0
    uses: int = 0
    probe_successes: int = 0
    probe_attempts: int = 0


# ============================================================================
# EXPANDED TRANSFORMATION OPERATORS (30+ operations)
# ============================================================================

class TransformOps:
    """Comprehensive transformation library"""

    # ===== TILING & REPETITION =====

    @staticmethod
    def tile_pattern_horizontal(grid: np.ndarray, pattern_cols: List[int]) -> np.ndarray:
        """Tile pattern horizontally"""
        result = grid.copy()
        h, w = grid.shape
        if not pattern_cols:
            return result

        pattern = grid[:, pattern_cols]
        for col_offset in range(0, w, len(pattern_cols)):
            for idx, src_col in enumerate(pattern_cols):
                tgt_col = col_offset + idx
                if tgt_col < w:
                    result[:, tgt_col] = pattern[:, idx]
        return result

    @staticmethod
    def tile_pattern_vertical(grid: np.ndarray, pattern_rows: List[int]) -> np.ndarray:
        """Tile pattern vertically"""
        result = grid.copy()
        h, w = grid.shape
        if not pattern_rows:
            return result

        pattern = grid[pattern_rows, :]
        for row_offset in range(0, h, len(pattern_rows)):
            for idx, src_row in enumerate(pattern_rows):
                tgt_row = row_offset + idx
                if tgt_row < h:
                    result[tgt_row, :] = pattern[idx, :]
        return result

    @staticmethod
    def repeat_object(grid: np.ndarray, times_h: int = 1, times_v: int = 1) -> np.ndarray:
        """Repeat entire grid pattern"""
        return np.tile(grid, (times_v, times_h))

    # ===== GEOMETRIC TRANSFORMS =====

    @staticmethod
    def rotate_90(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid)

    @staticmethod
    def rotate_180(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, 2)

    @staticmethod
    def rotate_270(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, 3)

    @staticmethod
    def flip_horizontal(grid: np.ndarray) -> np.ndarray:
        return np.fliplr(grid)

    @staticmethod
    def flip_vertical(grid: np.ndarray) -> np.ndarray:
        return np.flipud(grid)

    @staticmethod
    def transpose(grid: np.ndarray) -> np.ndarray:
        return grid.T

    # ===== SCALING =====

    @staticmethod
    def scale_up(grid: np.ndarray, factor: int = 2) -> np.ndarray:
        """Scale up by repeating each cell"""
        return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)

    @staticmethod
    def scale_down(grid: np.ndarray, factor: int = 2) -> np.ndarray:
        """Scale down by sampling"""
        return grid[::factor, ::factor]

    # ===== EXTRACTION & CROPPING =====

    @staticmethod
    def extract_rectangle(grid: np.ndarray, y1: int, x1: int, y2: int, x2: int) -> np.ndarray:
        """Extract rectangular region"""
        return grid[y1:y2+1, x1:x2+1].copy()

    @staticmethod
    def crop_to_content(grid: np.ndarray) -> np.ndarray:
        """Crop to bounding box of non-zero content"""
        rows = np.any(grid != 0, axis=1)
        cols = np.any(grid != 0, axis=0)
        if not rows.any() or not cols.any():
            return grid
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return grid[y1:y2+1, x1:x2+1].copy()

    # ===== FLOOD FILL & PAINTING =====

    @staticmethod
    def flood_fill(grid: np.ndarray, start_y: int, start_x: int, new_color: int) -> np.ndarray:
        """Flood fill from starting position"""
        result = grid.copy()
        h, w = grid.shape
        if not (0 <= start_y < h and 0 <= start_x < w):
            return result

        target_color = grid[start_y, start_x]
        if target_color == new_color:
            return result

        stack = [(start_y, start_x)]
        while stack:
            y, x = stack.pop()
            if not (0 <= y < h and 0 <= x < w) or result[y, x] != target_color:
                continue

            result[y, x] = new_color
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((y + dy, x + dx))

        return result

    @staticmethod
    def fill_enclosed_regions(grid: np.ndarray, fill_color: int) -> np.ndarray:
        """Fill all enclosed regions"""
        result = grid.copy()
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)

        # Find all regions of color 0
        for i in range(h):
            for j in range(w):
                if grid[i, j] == 0 and not visited[i, j]:
                    # Check if region is enclosed
                    region = []
                    stack = [(i, j)]
                    is_enclosed = True
                    temp_visited = set()

                    while stack:
                        y, x = stack.pop()
                        if (y, x) in temp_visited:
                            continue
                        if not (0 <= y < h and 0 <= x < w):
                            is_enclosed = False
                            continue
                        if grid[y, x] != 0:
                            continue

                        temp_visited.add((y, x))
                        region.append((y, x))

                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((y + dy, x + dx))

                    for y, x in region:
                        visited[y, x] = True

                    if is_enclosed and region:
                        for y, x in region:
                            result[y, x] = fill_color

        return result

    # ===== PATTERN MARKERS =====

    @staticmethod
    def add_cross_pattern(grid: np.ndarray, center_y: int, center_x: int, cross_color: int) -> np.ndarray:
        """Add cross pattern"""
        result = grid.copy()
        h, w = grid.shape

        for dy in [-1, 1]:
            ny = center_y + dy
            if 0 <= ny < h and 0 <= center_x < w:
                result[ny, center_x] = cross_color

        for dx in [-1, 1]:
            nx = center_x + dx
            if 0 <= center_y < h and 0 <= nx < w:
                result[center_y, nx] = cross_color

        return result

    @staticmethod
    def add_border(grid: np.ndarray, border_color: int, thickness: int = 1) -> np.ndarray:
        """Add border around grid"""
        h, w = grid.shape
        result = np.full((h + 2*thickness, w + 2*thickness), border_color, dtype=int)
        result[thickness:thickness+h, thickness:thickness+w] = grid
        return result

    @staticmethod
    def draw_line_horizontal(grid: np.ndarray, y: int, color: int) -> np.ndarray:
        """Draw horizontal line"""
        result = grid.copy()
        if 0 <= y < grid.shape[0]:
            result[y, :] = color
        return result

    @staticmethod
    def draw_line_vertical(grid: np.ndarray, x: int, color: int) -> np.ndarray:
        """Draw vertical line"""
        result = grid.copy()
        if 0 <= x < grid.shape[1]:
            result[:, x] = color
        return result

    # ===== COLOR OPERATIONS =====

    @staticmethod
    def color_mapping(grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
        """Apply color substitution"""
        result = grid.copy()
        for old_color, new_color in mapping.items():
            result[grid == old_color] = new_color
        return result

    @staticmethod
    def swap_colors(grid: np.ndarray, color1: int, color2: int) -> np.ndarray:
        """Swap two colors"""
        result = grid.copy()
        mask1 = grid == color1
        mask2 = grid == color2
        result[mask1] = color2
        result[mask2] = color1
        return result

    @staticmethod
    def replace_color(grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
        """Replace all instances of a color"""
        result = grid.copy()
        result[grid == old_color] = new_color
        return result

    @staticmethod
    def invert_colors(grid: np.ndarray, max_color: int = 9) -> np.ndarray:
        """Invert colors"""
        return max_color - grid

    # ===== OBJECT MANIPULATION =====

    @staticmethod
    def move_objects(grid: np.ndarray, dy: int, dx: int, color: Optional[int] = None) -> np.ndarray:
        """Move all objects (or specific color) by offset"""
        result = np.zeros_like(grid)
        h, w = grid.shape

        for y in range(h):
            for x in range(w):
                if grid[y, x] != 0:
                    if color is None or grid[y, x] == color:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            result[ny, nx] = grid[y, x]
                    else:
                        result[y, x] = grid[y, x]

        return result

    @staticmethod
    def gravity_down(grid: np.ndarray) -> np.ndarray:
        """Apply gravity - move objects down"""
        result = np.zeros_like(grid)
        h, w = grid.shape

        for x in range(w):
            column = []
            for y in range(h):
                if grid[y, x] != 0:
                    column.append(grid[y, x])

            # Place at bottom
            for idx, color in enumerate(column):
                result[h - len(column) + idx, x] = color

        return result

    @staticmethod
    def delete_color(grid: np.ndarray, color: int) -> np.ndarray:
        """Remove all cells of specific color"""
        result = grid.copy()
        result[grid == color] = 0
        return result

    # ===== MASKS & OVERLAYS =====

    @staticmethod
    def apply_mask(grid: np.ndarray, mask_grid: np.ndarray) -> np.ndarray:
        """Apply binary mask"""
        return np.where(mask_grid != 0, grid, 0)

    @staticmethod
    def overlay(bottom: np.ndarray, top: np.ndarray, offset_y: int = 0, offset_x: int = 0) -> np.ndarray:
        """Overlay top grid onto bottom"""
        result = bottom.copy()
        h_top, w_top = top.shape
        h_bot, w_bot = bottom.shape

        for y in range(h_top):
            for x in range(w_top):
                ny, nx = y + offset_y, x + offset_x
                if 0 <= ny < h_bot and 0 <= nx < w_bot and top[y, x] != 0:
                    result[ny, nx] = top[y, x]

        return result

    # ===== GRID OPERATIONS =====

    @staticmethod
    def stack_horizontal(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
        """Stack grids horizontally"""
        return np.hstack([grid1, grid2])

    @staticmethod
    def stack_vertical(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
        """Stack grids vertically"""
        return np.vstack([grid1, grid2])

    @staticmethod
    def mirror_horizontal(grid: np.ndarray) -> np.ndarray:
        """Mirror and concatenate horizontally"""
        return np.hstack([grid, np.fliplr(grid)])

    @staticmethod
    def mirror_vertical(grid: np.ndarray) -> np.ndarray:
        """Mirror and concatenate vertically"""
        return np.vstack([grid, np.flipud(grid)])


# ============================================================================
# WORKSPACE
# ============================================================================

class Workspace:
    """Shared workspace with analysis capabilities"""

    def __init__(self, grid: np.ndarray):
        self.grid = grid.copy()
        self.height, self.width = grid.shape
        self.objects: List[Object] = []

    def objectify_by_color(self) -> List[Object]:
        """Segment into connected components"""
        objects = []
        visited = np.zeros_like(self.grid, dtype=bool)

        for i in range(self.height):
            for j in range(self.width):
                if not visited[i, j] and self.grid[i, j] != 0:
                    color = self.grid[i, j]
                    positions = set()
                    stack = [(i, j)]

                    while stack:
                        y, x = stack.pop()
                        if (y < 0 or y >= self.height or x < 0 or x >= self.width or
                            visited[y, x] or self.grid[y, x] != color):
                            continue

                        visited[y, x] = True
                        positions.add((y, x))

                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((y + dy, x + dx))

                    if positions:
                        objects.append(Object(positions=positions, color=int(color)))

        self.objects = objects
        return objects

    def find_markers(self) -> List[Tuple[int, int, int]]:
        """Find isolated pixels"""
        markers = []
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] != 0:
                    neighbor_count = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.height and 0 <= nj < self.width:
                                if self.grid[ni, nj] == self.grid[i, j]:
                                    neighbor_count += 1

                    if neighbor_count == 0:
                        markers.append((i, j, int(self.grid[i, j])))

        return markers

    def compute_mdl(self) -> float:
        """Compute description length"""
        unique_colors = len(np.unique(self.grid)) - 1
        non_zero = np.count_nonzero(self.grid)

        if unique_colors == 0:
            return 0.0

        base_cost = non_zero * np.log2(unique_colors + 1)
        objects = self.objectify_by_color() if not self.objects else self.objects
        pattern_cost = len(objects) * 2

        return float(base_cost + pattern_cost)

    def to_text_representation(self) -> str:
        """Convert grid to compact text for LLM"""
        lines = []
        for row in self.grid:
            lines.append(" ".join(str(int(cell)) for cell in row))
        return "\n".join(lines)


# ============================================================================
# PATTERN ANALYZER (Enhanced)
# ============================================================================

class PatternAnalyzer:
    """Advanced pattern detection"""

    @staticmethod
    def detect_symmetry(grid: np.ndarray) -> Dict[str, bool]:
        """Detect various symmetries"""
        return {
            'horizontal': np.array_equal(grid, np.fliplr(grid)),
            'vertical': np.array_equal(grid, np.flipud(grid)),
            'rotational_90': np.array_equal(grid, np.rot90(grid)),
            'rotational_180': np.array_equal(grid, np.rot90(grid, 2))
        }

    @staticmethod
    def detect_repeating_pattern(grid: np.ndarray) -> Dict[str, Any]:
        """Detect tiling patterns"""
        height, width = grid.shape

        # Horizontal
        for period in range(1, width // 2 + 1):
            if width % period == 0:
                is_repeating = True
                base = grid[:, :period]

                for offset in range(period, width, period):
                    segment = grid[:, offset:offset+period]
                    if segment.shape[1] == period and not np.array_equal(base, segment):
                        is_repeating = False
                        break

                if is_repeating:
                    return {'type': 'horizontal_tile', 'period': period, 'pattern_cols': list(range(period))}

        # Vertical
        for period in range(1, height // 2 + 1):
            if height % period == 0:
                is_repeating = True
                base = grid[:period, :]

                for offset in range(period, height, period):
                    segment = grid[offset:offset+period, :]
                    if segment.shape[0] == period and not np.array_equal(base, segment):
                        is_repeating = False
                        break

                if is_repeating:
                    return {'type': 'vertical_tile', 'period': period, 'pattern_rows': list(range(period))}

        return {'type': 'none'}

    @staticmethod
    def analyze_transformation(input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """Analyze transformation between input and output"""
        analysis = {
            'shape_changed': input_grid.shape != output_grid.shape,
            'input_shape': input_grid.shape,
            'output_shape': output_grid.shape,
            'colors_changed': False,
            'rotation': None,
            'scaling': None
        }

        # Check for simple transformations
        if np.array_equal(output_grid, np.rot90(input_grid)):
            analysis['rotation'] = 90
        elif np.array_equal(output_grid, np.rot90(input_grid, 2)):
            analysis['rotation'] = 180
        elif np.array_equal(output_grid, np.rot90(input_grid, 3)):
            analysis['rotation'] = 270
        elif np.array_equal(output_grid, np.fliplr(input_grid)):
            analysis['flip'] = 'horizontal'
        elif np.array_equal(output_grid, np.flipud(input_grid)):
            analysis['flip'] = 'vertical'

        # Check colors
        in_colors = set(input_grid.flatten())
        out_colors = set(output_grid.flatten())
        analysis['colors_changed'] = in_colors != out_colors
        analysis['input_colors'] = sorted(list(in_colors))
        analysis['output_colors'] = sorted(list(out_colors))

        return analysis


# ============================================================================
# ENHANCED LLM HYPOTHESIS GENERATOR
# ============================================================================

class LLMHypothesisGenerator:
    """Improved LLM integration with visual examples"""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def generate_hypothesis(self, train_examples: List[Dict], max_attempts: int = 5) -> List[Dict[str, Any]]:
        """Generate top 5 hypotheses"""

        print("  🤖 Calling LLM for hypothesis generation...")

        # Build detailed prompt with actual grid examples
        examples_text = self._format_examples_for_llm(train_examples[:2])

        print(f"  📊 Showing {min(2, len(train_examples))} training examples to LLM")

        prompt = f"""Analyze these ARC-AGI transformation examples and propose the top 5 most likely strategies.

{examples_text}

CRITICAL: Use EXACT strategy names below. Do not use category names.

Available strategies (use these exact names in "strategy" field):

horizontal_tile - repeat column pattern across grid
vertical_tile - repeat row pattern down grid
repeat_object - tile entire grid NxM times
rotate - rotate 90/180/270 degrees (params: {{"degrees": 90}})
flip - horizontal or vertical flip (params: {{"direction": "horizontal"}})
transpose - swap rows and columns
mirror - create symmetric copy (params: {{"direction": "horizontal"}})
extract_rectangle - crop specific area
crop_to_content - remove empty borders
scale_up - enlarge by repeating cells (params: {{"factor": 2}})
scale_down - shrink by sampling (params: {{"factor": 2}})
color_map - substitute colors (params: {{"mapping": {{2:5, 3:7}}}})
swap_colors - exchange two colors (params: {{"color1": 1, "color2": 2}})
add_cross - add + pattern around markers
add_border - frame the grid (params: {{"color": 1}})
move_objects - translate by offset (params: {{"dy": 1, "dx": 0}})
gravity - apply downward force
overlay - combine layers

You can chain multiple transformations! Use "chain" as strategy:

Respond with ONLY valid JSON array of 5 hypotheses:
[
  {{
    "strategy": "horizontal_tile",
    "confidence": 0.9,
    "reasoning": "brief explanation",
    "parameters": {{}}
  }},
  {{
    "strategy": "chain",
    "confidence": 0.8,
    "reasoning": "extract then scale",
    "parameters": {{
      "steps": [
        {{"strategy": "extract_rectangle", "parameters": {{}}}},
        {{"strategy": "scale_up", "parameters": {{"factor": 2}}}}
      ]
    }}
  }},
  {{
    "strategy": "rotate",
    "confidence": 0.6,
    "reasoning": "brief explanation",
    "parameters": {{"degrees": 90}}
  }},
  {{
    "strategy": "chain",
    "confidence": 0.5,
    "reasoning": "flip then add border",
    "parameters": {{
      "steps": [
        {{"strategy": "flip", "parameters": {{"direction": "horizontal"}}}},
        {{"strategy": "add_border", "parameters": {{"color": 1}}}}
      ]
    }}
  }},
  {{
    "strategy": "color_map",
    "confidence": 0.3,
    "reasoning": "brief explanation",
    "parameters": {{}}
  }}
]

NO other text, NO markdown, ONLY JSON array:"""

        try:
            print("  ⏳ Waiting for LLM response...")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You output only valid JSON arrays, no other text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )

            content = response.choices[0].message.content.strip()

            print(f"  ✅ LLM response received ({len(content)} chars)")
            print(f"  📝 Raw response preview: {content[:150]}...")

            # Strip markdown
            if content.startswith("```"):
                print("  🔧 Stripping markdown wrapper")
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if len(lines) > 2 else lines[1:])
                content = content.replace("```json", "").replace("```", "").strip()

            hypotheses = json.loads(content)
            print(f"  ✅ Parsed {len(hypotheses) if isinstance(hypotheses, list) else 1} hypothesis/hypotheses")

            # Ensure list format
            if isinstance(hypotheses, dict):
                hypotheses = [hypotheses]

            # Validate strategy names
            valid_strategies = {
                'horizontal_tile', 'vertical_tile', 'repeat_object',
                'rotate', 'flip', 'transpose', 'mirror',
                'extract_rectangle', 'crop_to_content',
                'scale_up', 'scale_down',
                'color_map', 'swap_colors', 'replace_color',
                'add_cross', 'add_border', 'flood_fill',
                'move_objects', 'gravity', 'overlay',
                'chain'  # Allow chaining
            }

            validated = []
            for hyp in hypotheses[:3]:
                strategy = hyp.get('strategy', '').lower()

                # Fix common LLM mistakes
                if strategy in ['tiling', 'tile']:
                    print(f"  🔧 Fixing strategy: {strategy} → horizontal_tile")
                    hyp['strategy'] = 'horizontal_tile'
                elif strategy in ['geometric', 'rotation']:
                    print(f"  🔧 Fixing strategy: {strategy} → rotate")
                    hyp['strategy'] = 'rotate'
                elif strategy in ['scaling', 'scale']:
                    print(f"  🔧 Fixing strategy: {strategy} → scale_up")
                    hyp['strategy'] = 'scale_up'
                elif strategy in ['extraction', 'extract', 'crop']:
                    print(f"  🔧 Fixing strategy: {strategy} → extract_rectangle")
                    hyp['strategy'] = 'extract_rectangle'
                elif strategy not in valid_strategies:
                    print(f"  ⚠️  Invalid strategy '{strategy}', skipping")
                    continue

                validated.append(hyp)

            if not validated:
                print("  ❌ No valid strategies found, using fallback")
                return self._fallback_hypotheses(train_examples)

            # Show what strategies were proposed
            for i, hyp in enumerate(validated, 1):
                print(f"  💡 Hypothesis {i}: {hyp.get('strategy', 'unknown')} (confidence: {hyp.get('confidence', 0):.2f})")

            return validated

        except Exception as e:
            print(f"  ❌ LLM failed: {e}")
            print("  🔄 Using fallback pattern detection")
            return self._fallback_hypotheses(train_examples)

    def _format_examples_for_llm(self, examples: List[Dict]) -> str:
        """Format examples as visual grids"""
        formatted = []

        for idx, ex in enumerate(examples, 1):
            inp = np.array(ex['input'])
            out = np.array(ex['output'])

            ws_in = Workspace(inp)
            ws_out = Workspace(out)

            analysis = PatternAnalyzer.analyze_transformation(inp, out)

            formatted.append(f"""Example {idx}:
INPUT ({inp.shape[0]}x{inp.shape[1]}):
{ws_in.to_text_representation()}

OUTPUT ({out.shape[0]}x{out.shape[1]}):
{ws_out.to_text_representation()}

Analysis:
- Shape changed: {analysis['shape_changed']}
- Input colors: {analysis['input_colors']}
- Output colors: {analysis['output_colors']}
- Rotation: {analysis.get('rotation', 'none')}
""")

        return "\n".join(formatted)

    def _fallback_hypotheses(self, train_examples: List[Dict]) -> List[Dict[str, Any]]:
        """Fallback pattern-based hypotheses"""
        if not train_examples:
            return [{"strategy": "color_map", "confidence": 0.3, "parameters": {}}]

        hypotheses = []

        example = train_examples[0]
        out_grid = np.array(example['output'])
        in_grid = np.array(example['input'])

        # Check size relationship
        in_h, in_w = in_grid.shape
        out_h, out_w = out_grid.shape

        # Try rotation first
        for deg in [90, 180, 270]:
            rotated = np.rot90(in_grid, k=deg//90)
            if np.array_equal(rotated, out_grid):
                hypotheses.append({
                    "strategy": "rotate",
                    "confidence": 0.95,
                    "reasoning": f"{deg}° rotation detected",
                    "parameters": {"degrees": deg}
                })
                break

        # Try flips
        if np.array_equal(np.fliplr(in_grid), out_grid):
            hypotheses.append({
                "strategy": "flip",
                "confidence": 0.95,
                "reasoning": "Horizontal flip detected",
                "parameters": {"direction": "horizontal"}
            })
        elif np.array_equal(np.flipud(in_grid), out_grid):
            hypotheses.append({
                "strategy": "flip",
                "confidence": 0.95,
                "reasoning": "Vertical flip detected",
                "parameters": {"direction": "vertical"}
            })

        # Try tiling detection
        pattern = PatternAnalyzer.detect_repeating_pattern(out_grid)
        if pattern['type'] == 'horizontal_tile':
            hypotheses.append({
                "strategy": "horizontal_tile",
                "confidence": 0.8,
                "reasoning": "Horizontal tiling detected",
                "parameters": pattern
            })
        elif pattern['type'] == 'vertical_tile':
            hypotheses.append({
                "strategy": "vertical_tile",
                "confidence": 0.8,
                "reasoning": "Vertical tiling detected",
                "parameters": pattern
            })

        # Try extraction
        if out_h < in_h or out_w < in_w:
            hypotheses.append({
                "strategy": "extract_rectangle",
                "confidence": 0.7,
                "reasoning": "Output smaller - likely extraction",
                "parameters": {}
            })

            # Also try extract + rotate
            hypotheses.append({
                "strategy": "chain",
                "confidence": 0.65,
                "reasoning": "Extract then rotate",
                "parameters": {
                    "steps": [
                        {"strategy": "extract_rectangle", "parameters": {}},
                        {"strategy": "rotate", "parameters": {"degrees": 90}}
                    ]
                }
            })

        # Try scaling
        if out_h > in_h and out_w > in_w:
            if out_h % in_h == 0 and out_w % in_w == 0:
                factor_h = out_h // in_h
                factor_w = out_w // in_w
                if factor_h == factor_w:
                    hypotheses.append({
                        "strategy": "scale_up",
                        "confidence": 0.8,
                        "reasoning": f"Scaled up {factor_h}x",
                        "parameters": {"factor": factor_h}
                    })

        # Try common chains
        if len(hypotheses) < 3:
            # Rotate then mirror
            hypotheses.append({
                "strategy": "chain",
                "confidence": 0.6,
                "reasoning": "Rotate then mirror",
                "parameters": {
                    "steps": [
                        {"strategy": "rotate", "parameters": {"degrees": 90}},
                        {"strategy": "mirror", "parameters": {"direction": "horizontal"}}
                    ]
                }
            })

        if len(hypotheses) < 4:
            # Color map then add border
            hypotheses.append({
                "strategy": "chain",
                "confidence": 0.5,
                "reasoning": "Recolor then frame",
                "parameters": {
                    "steps": [
                        {"strategy": "color_map", "parameters": {}},
                        {"strategy": "add_border", "parameters": {"color": 1}}
                    ]
                }
            })

        # Ensure we return 5
        while len(hypotheses) < 5:
            # Add more diverse fallbacks
            if len(hypotheses) == 3:
                hypotheses.append({
                    "strategy": "add_cross",
                    "confidence": 0.25,
                    "reasoning": "Try pattern markers",
                    "parameters": {}
                })
            elif len(hypotheses) == 4:
                hypotheses.append({
                    "strategy": "mirror",
                    "confidence": 0.2,
                    "reasoning": "Try mirroring",
                    "parameters": {"direction": "horizontal"}
                })
            else:
                hypotheses.append({
                    "strategy": "color_map",
                    "confidence": 0.2,
                    "reasoning": "Default color mapping",
                    "parameters": {}
                })

        return hypotheses[:5]


# ============================================================================
# VERIFIER
# ============================================================================

class Verifier:
    """Active verification through probing"""

    def __init__(self):
        self.probe_history: List[Dict] = []

    def verify_transformation(self, input_grid: np.ndarray,
                             predicted: np.ndarray, expected: np.ndarray) -> float:
        """Run full verification suite, return score 0-1"""
        scores = []

        # Exact match
        if np.array_equal(predicted, expected):
            return 1.0

        # Shape match
        if predicted.shape == expected.shape:
            scores.append(0.3)

            # Color distribution similarity
            pred_colors = Counter(predicted.flatten())
            exp_colors = Counter(expected.flatten())

            color_sim = sum(min(pred_colors[c], exp_colors[c]) for c in set(pred_colors) | set(exp_colors))
            total = sum(exp_colors.values())
            scores.append(color_sim / total if total > 0 else 0)

        return np.mean(scores) if scores else 0.0


# ============================================================================
# MAIN SOLVER WITH MULTI-HYPOTHESIS TESTING
# ============================================================================

class ARCSolver:
    """Enhanced solver with multi-hypothesis testing"""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.hypothesis_gen = LLMHypothesisGenerator(model_name, api_key)
        self.verifier = Verifier()
        self.stats = {'attempted': 0, 'correct': 0, 'failed': 0}

    def solve_task(self, task_data: Dict) -> Tuple[Optional[List[List[int]]], Dict]:
        """Solve task by trying multiple hypotheses"""
        train_examples = task_data.get('train', [])
        test_examples = task_data.get('test', [])

        if not train_examples or not test_examples:
            return None, {'error': 'Missing data'}

        print(f"  📚 Training examples: {len(train_examples)}")

        # Generate multiple hypotheses
        hypotheses = self.hypothesis_gen.generate_hypothesis(train_examples)

        best_prediction = None
        best_score = -1
        best_hypothesis = None

        print(f"\n  🔬 Testing {len(hypotheses)} hypotheses on training data...")

        # Try each hypothesis
        for idx, hyp in enumerate(hypotheses, 1):
            try:
                strategy = hyp.get('strategy', 'unknown')
                print(f"\n  🧪 Testing hypothesis {idx}: {strategy}")

                # Test on training examples
                train_scores = []
                for ex_idx, ex in enumerate(train_examples):
                    inp = np.array(ex['input'])
                    exp_out = np.array(ex['output'])

                    pred = self._apply_hypothesis(inp, hyp, train_examples)
                    if pred is not None:
                        score = self.verifier.verify_transformation(inp, pred, exp_out)
                        train_scores.append(score)
                        print(f"    Example {ex_idx+1}: score={score:.2f}")
                    else:
                        print(f"    Example {ex_idx+1}: failed to apply")

                avg_train_score = np.mean(train_scores) if train_scores else 0
                print(f"  📊 Average training score: {avg_train_score:.2f}")

                # If good on training, try test
                if avg_train_score > 0.5:  # Lowered from 0.8 to try more
                    print(f"  ✅ Score above threshold! Applying to test case...")
                    test_input = np.array(test_examples[0]['input'])
                    prediction = self._apply_hypothesis(test_input, hyp, train_examples)

                    if prediction is not None and avg_train_score > best_score:
                        best_score = avg_train_score
                        best_prediction = prediction
                        best_hypothesis = hyp
                        print(f"  ⭐ New best! (score: {best_score:.2f})")

                        # Don't break - try all hypotheses to find the absolute best
                else:
                    print(f"  ❌ Score too low, trying next hypothesis...")

            except Exception as e:
                print(f"  ⚠️  Hypothesis {idx} error: {e}")
                continue

        if best_prediction is not None:
            print(f"\n  🎯 Best strategy: {best_hypothesis.get('strategy')} (score: {best_score:.2f})")
        else:
            print(f"\n  😞 No hypothesis worked well enough")

        metadata = {
            'hypotheses': hypotheses,
            'best_hypothesis': best_hypothesis,
            'best_score': best_score
        }

        if best_prediction is not None:
            return best_prediction.tolist(), metadata
        else:
            return None, metadata

        metadata = {
            'hypotheses': hypotheses,
            'best_hypothesis': best_hypothesis,
            'best_score': best_score
        }

        if best_prediction is not None:
            return best_prediction.tolist(), metadata
        else:
            return None, metadata

    def _apply_hypothesis(self, input_grid: np.ndarray, hypothesis: Dict,
                          train_examples: List[Dict]) -> Optional[np.ndarray]:
        """Apply hypothesis transformation"""
        strategy = hypothesis.get('strategy', 'none')
        params = hypothesis.get('parameters', {})

        print(f"      🔧 Applying strategy: {strategy} with params: {params}")

        # Handle chained transformations
        if strategy == 'chain':
            steps = params.get('steps', [])
            if not steps:
                print(f"      ⚠️  Empty chain")
                return None

            print(f"      🔗 Chaining {len(steps)} transformations...")
            result = input_grid.copy()

            for i, step in enumerate(steps, 1):
                step_strategy = step.get('strategy')
                step_params = step.get('parameters', {})
                print(f"        Step {i}: {step_strategy}")

                # Create temporary hypothesis for this step
                temp_hyp = {'strategy': step_strategy, 'parameters': step_params}
                result = self._apply_hypothesis(result, temp_hyp, train_examples)

                if result is None:
                    print(f"        ❌ Step {i} failed, chain broken")
                    return None

            print(f"      ✅ Chain completed successfully")
            return result

        try:
            # TILING
            if strategy == 'horizontal_tile':
                ws = Workspace(input_grid)
                markers = ws.find_markers()
                if len(markers) >= 2:
                    cols = sorted(set(m[1] for m in markers))
                    return TransformOps.tile_pattern_horizontal(input_grid, cols)

            elif strategy == 'vertical_tile':
                ws = Workspace(input_grid)
                markers = ws.find_markers()
                if len(markers) >= 2:
                    rows = sorted(set(m[0] for m in markers))
                    return TransformOps.tile_pattern_vertical(input_grid, rows)

            elif strategy == 'repeat_object':
                times_h = params.get('times_h', 2)
                times_v = params.get('times_v', 2)
                return TransformOps.repeat_object(input_grid, times_h, times_v)

            # GEOMETRIC
            elif strategy == 'rotate':
                degrees = params.get('degrees', 90)
                if degrees == 90:
                    return TransformOps.rotate_90(input_grid)
                elif degrees == 180:
                    return TransformOps.rotate_180(input_grid)
                elif degrees == 270:
                    return TransformOps.rotate_270(input_grid)

            elif strategy == 'flip':
                direction = params.get('direction', 'horizontal')
                if direction == 'horizontal':
                    return TransformOps.flip_horizontal(input_grid)
                else:
                    return TransformOps.flip_vertical(input_grid)

            elif strategy == 'transpose':
                return TransformOps.transpose(input_grid)

            elif strategy == 'mirror':
                direction = params.get('direction', 'horizontal')
                if direction == 'horizontal':
                    return TransformOps.mirror_horizontal(input_grid)
                else:
                    return TransformOps.mirror_vertical(input_grid)

            # EXTRACTION
            elif strategy == 'extract_rectangle':
                if train_examples:
                    ex = train_examples[0]
                    in_ex = np.array(ex['input'])
                    out_ex = np.array(ex['output'])

                    # Find where output appears in input
                    h_out, w_out = out_ex.shape
                    h_in, w_in = in_ex.shape

                    for y in range(h_in - h_out + 1):
                        for x in range(w_in - w_out + 1):
                            if np.array_equal(in_ex[y:y+h_out, x:x+w_out], out_ex):
                                return input_grid[y:y+h_out, x:x+w_out].copy()

            elif strategy == 'crop_to_content':
                return TransformOps.crop_to_content(input_grid)

            # SCALING
            elif strategy == 'scale_up':
                factor = params.get('factor', 2)
                return TransformOps.scale_up(input_grid, factor)

            elif strategy == 'scale_down':
                factor = params.get('factor', 2)
                return TransformOps.scale_down(input_grid, factor)

            # COLOR OPERATIONS
            elif strategy == 'color_map':
                mapping = self._infer_color_mapping(train_examples)
                if mapping:
                    return TransformOps.color_mapping(input_grid, mapping)

            elif strategy == 'swap_colors':
                color1 = params.get('color1', 1)
                color2 = params.get('color2', 2)
                return TransformOps.swap_colors(input_grid, color1, color2)

            # PATTERN ADDITION
            elif strategy == 'add_cross':
                result = input_grid.copy()
                ws = Workspace(input_grid)
                markers = ws.find_markers()
                for y, x, color in markers:
                    cross_color = params.get('cross_color', 7 if color == 1 else 4)
                    result = TransformOps.add_cross_pattern(result, y, x, cross_color)
                return result

            elif strategy == 'add_border':
                color = params.get('color', 1)
                return TransformOps.add_border(input_grid, color)

            # OBJECT MANIPULATION
            elif strategy == 'move_objects':
                dy = params.get('dy', 0)
                dx = params.get('dx', 0)
                return TransformOps.move_objects(input_grid, dy, dx)

            elif strategy == 'gravity':
                return TransformOps.gravity_down(input_grid)

        except Exception as e:
            pass

        return None

    def _infer_color_mapping(self, train_examples: List[Dict]) -> Dict[int, int]:
        """Infer color mapping from examples"""
        mapping = {}
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])

            if inp.shape != out.shape:
                continue

            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    if inp[i, j] != out[i, j]:
                        old = int(inp[i, j])
                        new = int(out[i, j])
                        if old in mapping and mapping[old] != new:
                            return {}
                        mapping[old] = new

        return mapping

    def solve_directory(self, task_dir: str) -> Dict[str, Any]:
        """Solve all tasks in directory"""
        task_dir = Path(task_dir)
        json_files = sorted(task_dir.glob("*.json"))

        print(f"\n{'='*70}")
        print(f"Enhanced ARC-AGI Solver - Model: {self.model_name}")
        print(f"Features: 30+ operators, multi-hypothesis testing, visual prompts")
        print(f"Tasks directory: {task_dir}")
        print(f"Total tasks: {len(json_files)}")
        print(f"{'='*70}\n")

        results = {}

        for idx, json_file in enumerate(json_files, 1):
            task_id = json_file.stem

            try:
                with open(json_file, 'r') as f:
                    task_data = json.load(f)

                print(f"\n{'─'*70}")
                print(f"[{idx}/{len(json_files)}] Solving: {task_id}")
                print(f"{'─'*70}")

                prediction, metadata = self.solve_task(task_data)

                print(f"\n{'─'*70}")

                if prediction is not None and 'test' in task_data:
                    expected = task_data['test'][0].get('output')
                    if expected is not None:
                        is_correct = np.array_equal(np.array(prediction), np.array(expected))

                        self.stats['attempted'] += 1
                        if is_correct:
                            self.stats['correct'] += 1
                            print(f"  ✓ CORRECT")
                        else:
                            self.stats['failed'] += 1
                            print(f"  ✗ INCORRECT")

                        results[task_id] = {
                            'correct': is_correct,
                            'prediction': prediction,
                            'metadata': metadata
                        }
                    else:
                        print(f"  - No test output")
                        results[task_id] = {
                            'correct': None,
                            'prediction': prediction,
                            'metadata': metadata
                        }
                else:
                    self.stats['attempted'] += 1
                    self.stats['failed'] += 1
                    print(f"  ✗ FAILED")
                    results[task_id] = {
                        'correct': False,
                        'prediction': None,
                        'metadata': metadata
                    }

                if self.stats['attempted'] > 0:
                    rate = (self.stats['correct'] / self.stats['attempted']) * 100
                    print(f"  Running: {rate:.1f}% ({self.stats['correct']}/{self.stats['attempted']})\n")

            except Exception as e:
                print(f"  ✗ ERROR: {e}\n")
                results[task_id] = {'correct': False, 'error': str(e)}

        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Attempted: {self.stats['attempted']}")
        print(f"Correct: {self.stats['correct']}")
        print(f"Failed: {self.stats['failed']}")

        if self.stats['attempted'] > 0:
            rate = (self.stats['correct'] / self.stats['attempted']) * 100
            print(f"Success rate: {rate:.1f}%")

        print(f"{'='*70}\n")

        return {'results': results, 'stats': self.stats}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced ARC-AGI Solver')
    parser.add_argument('model', type=str, help='OpenAI model name')
    parser.add_argument('task_dir', type=str, help='Task directory')
    parser.add_argument('--api-key', type=str, default=None)
    args = parser.parse_args()

    solver = ARCSolver(args.model, args.api_key)
    results = solver.solve_directory(args.task_dir)

    output_file = Path(args.task_dir) / 'enhanced_solver_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()