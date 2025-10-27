#!/usr/bin/env python3
"""
ARC-AGI Solver with EXECUTABLE operators - Living Map principles + working execution.
Key improvement: Operators are actually implemented, not just described to LLM.

60/400 (15%) on eval, 31/400 (7.75%) on train using gpt-4o
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
import argparse
from openai import OpenAI


# ============================================================================
# EXECUTABLE OPERATORS - Actually implement transformations
# ============================================================================

class ExecutableOp(ABC):
    """Operator that can actually execute on grids."""

    @abstractmethod
    def execute(self, grid: np.ndarray) -> np.ndarray:
        """Execute transformation on grid."""
        pass

    @abstractmethod
    def mdl_cost(self) -> float:
        """Description length in bits."""
        pass

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description."""
        pass


class ColorMapOp(ExecutableOp):
    """Map colors: from_color -> to_color."""

    def __init__(self, mappings: Dict[int, int]):
        self.mappings = mappings

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        for from_c, to_c in self.mappings.items():
            result[grid == from_c] = to_c
        return result

    def mdl_cost(self) -> float:
        return 1.0 + len(self.mappings) * 0.5

    def describe(self) -> str:
        maps = ", ".join(f"{k}→{v}" for k, v in self.mappings.items())
        return f"color_map({maps})"


class RotateOp(ExecutableOp):
    """Rotate grid by k*90 degrees."""

    def __init__(self, k: int = 1):
        self.k = k % 4

    def execute(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, self.k)

    def mdl_cost(self) -> float:
        return 1.0

    def describe(self) -> str:
        return f"rotate_{self.k * 90}"


class FlipOp(ExecutableOp):
    """Flip grid vertically or horizontally."""

    def __init__(self, axis: str = 'vertical'):
        self.axis = axis

    def execute(self, grid: np.ndarray) -> np.ndarray:
        if self.axis == 'vertical':
            return np.flipud(grid)
        else:
            return np.fliplr(grid)

    def mdl_cost(self) -> float:
        return 1.0

    def describe(self) -> str:
        return f"flip_{self.axis}"


class TileOp(ExecutableOp):
    """Tile pattern n_rows x n_cols times."""

    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def execute(self, grid: np.ndarray) -> np.ndarray:
        return np.tile(grid, (self.n_rows, self.n_cols))

    def mdl_cost(self) -> float:
        return 2.5 + 0.5 * (self.n_rows > 1) + 0.5 * (self.n_cols > 1)

    def describe(self) -> str:
        return f"tile({self.n_rows}×{self.n_cols})"


class TileWithFlipOp(ExecutableOp):
    """Tile pattern with alternating flips (checkerboard)."""

    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def execute(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        result = np.zeros((h * self.n_rows, w * self.n_cols), dtype=grid.dtype)

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                tile = grid.copy()
                # Flip horizontally on odd columns
                if j % 2 == 1:
                    tile = np.fliplr(tile)
                # Flip vertically on odd rows
                if i % 2 == 1:
                    tile = np.flipud(tile)

                result[i*h:(i+1)*h, j*w:(j+1)*w] = tile

        return result

    def mdl_cost(self) -> float:
        return 3.5 + 0.5 * (self.n_rows > 1) + 0.5 * (self.n_cols > 1)

    def describe(self) -> str:
        return f"tile_with_flip({self.n_rows}×{self.n_cols})"


class GravityOp(ExecutableOp):
    """Apply gravity - move non-zero pixels in direction."""

    def __init__(self, direction: str = 'down'):
        self.direction = direction

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()

        if self.direction == 'down':
            for col in range(grid.shape[1]):
                non_zero = result[:, col][result[:, col] != 0]
                result[:, col] = 0
                if len(non_zero) > 0:
                    result[-len(non_zero):, col] = non_zero

        elif self.direction == 'up':
            for col in range(grid.shape[1]):
                non_zero = result[:, col][result[:, col] != 0]
                result[:, col] = 0
                if len(non_zero) > 0:
                    result[:len(non_zero), col] = non_zero

        elif self.direction == 'left':
            for row in range(grid.shape[0]):
                non_zero = result[row, :][result[row, :] != 0]
                result[row, :] = 0
                if len(non_zero) > 0:
                    result[row, :len(non_zero)] = non_zero

        elif self.direction == 'right':
            for row in range(grid.shape[0]):
                non_zero = result[row, :][result[row, :] != 0]
                result[row, :] = 0
                if len(non_zero) > 0:
                    result[row, -len(non_zero):] = non_zero

        return result

    def mdl_cost(self) -> float:
        return 1.5

    def describe(self) -> str:
        return f"gravity_{self.direction}"


class IdentityOp(ExecutableOp):
    """No transformation."""

    def execute(self, grid: np.ndarray) -> np.ndarray:
        return grid.copy()

    def mdl_cost(self) -> float:
        return 0.5

    def describe(self) -> str:
        return "identity"


class CompositeOp(ExecutableOp):
    """Composition of operators."""

    def __init__(self, operators: List[ExecutableOp], glue: str = 'sequential'):
        self.operators = operators
        self.glue = glue

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid
        for op in self.operators:
            result = op.execute(result)
        return result

    def mdl_cost(self) -> float:
        base = sum(op.mdl_cost() for op in self.operators)
        glue_cost = 0.5 if self.glue == 'sequential' else 1.0
        discount = 0.2 if self.glue == 'sequential' else 0.1
        return (base + glue_cost) * (1 - discount)

    def describe(self) -> str:
        return " → ".join(op.describe() for op in self.operators)


@dataclass
class Hypothesis:
    """Transformation hypothesis with executable operator."""
    program: ExecutableOp
    support: float = 0.0

    def mdl(self) -> float:
        return self.program.mdl_cost()

    def score(self) -> float:
        if self.support == 0:
            return float('inf')
        return self.mdl() / self.support

    def execute(self, grid: np.ndarray) -> np.ndarray:
        return self.program.execute(grid)


# ============================================================================
# PATTERN DETECTION - Analyze examples to propose operators
# ============================================================================

class PatternDetector:
    """Detect transformation patterns from examples."""

    @staticmethod
    def detect_color_mapping(examples: List[Dict]) -> Optional[Dict[int, int]]:
        """Detect if colors are consistently mapped."""
        mappings = None

        for ex in examples:
            in_grid = np.array(ex['input'])
            out_grid = np.array(ex['output'])

            if in_grid.shape != out_grid.shape:
                return None

            # Build mapping for this example
            ex_map = {}
            for in_color in np.unique(in_grid):
                if in_color == 0:
                    continue
                # Get all output values at positions where input is this color
                out_values = out_grid[in_grid == in_color]
                # Take most common non-zero output color
                non_zero = out_values[out_values != 0]
                if len(non_zero) > 0:
                    unique, counts = np.unique(non_zero, return_counts=True)
                    ex_map[int(in_color)] = int(unique[counts.argmax()])

            if mappings is None:
                mappings = ex_map
            elif mappings != ex_map:
                return None

        return mappings if mappings else None

    @staticmethod
    def detect_tiling(examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Detect if output is tiled version of input."""
        for ex in examples:
            in_grid = np.array(ex['input'])
            out_grid = np.array(ex['output'])

            in_h, in_w = in_grid.shape
            out_h, out_w = out_grid.shape

            if out_h % in_h != 0 or out_w % in_w != 0:
                continue

            n_rows = out_h // in_h
            n_cols = out_w // in_w

            # Check if all tiles match (simple tiling)
            matches_simple = True
            for i in range(n_rows):
                for j in range(n_cols):
                    tile = out_grid[i*in_h:(i+1)*in_h, j*in_w:(j+1)*in_w]
                    if not np.array_equal(tile, in_grid):
                        matches_simple = False
                        break

            if matches_simple:
                return (n_rows, n_cols)

            # Check if tiles are flipped (checkerboard tiling)
            matches_flip = True
            for i in range(n_rows):
                for j in range(n_cols):
                    tile = out_grid[i*in_h:(i+1)*in_h, j*in_w:(j+1)*in_w]
                    expected = in_grid.copy()
                    if j % 2 == 1:
                        expected = np.fliplr(expected)
                    if i % 2 == 1:
                        expected = np.flipud(expected)
                    if not np.array_equal(tile, expected):
                        matches_flip = False
                        break

            if matches_flip:
                return ('flip', n_rows, n_cols)

        return None

    @staticmethod
    def detect_rotation(examples: List[Dict]) -> Optional[int]:
        """Detect if output is rotated input."""
        for k in [1, 2, 3]:
            matches_all = True
            for ex in examples:
                in_grid = np.array(ex['input'])
                out_grid = np.array(ex['output'])
                rotated = np.rot90(in_grid, k)
                if not np.array_equal(rotated, out_grid):
                    matches_all = False
                    break
            if matches_all:
                return k
        return None

    @staticmethod
    def detect_flip(examples: List[Dict]) -> Optional[str]:
        """Detect if output is flipped input."""
        for axis in ['vertical', 'horizontal']:
            matches_all = True
            for ex in examples:
                in_grid = np.array(ex['input'])
                out_grid = np.array(ex['output'])
                flipped = np.flipud(in_grid) if axis == 'vertical' else np.fliplr(in_grid)
                if not np.array_equal(flipped, out_grid):
                    matches_all = False
                    break
            if matches_all:
                return axis
        return None


# ============================================================================
# SOLVER - Generate and test hypotheses
# ============================================================================

class LivingMapSolver:
    """Solver using pattern detection + executable operators."""

    def __init__(self, model: str = "gpt-4", bit_budget: int = 100):
        self.client = OpenAI()
        self.model = model
        self.bit_budget = bit_budget
        self.bits_spent = 0

    def solve_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Solve task using pattern detection + verification."""

        examples = task['train']
        test_input = np.array(task['test'][0]['input'])

        # Generate hypotheses from patterns
        hypotheses = self._generate_hypotheses(examples)

        if not hypotheses:
            # Fallback to LLM if no patterns detected
            return self._llm_fallback(task)

        # Test hypotheses on training examples
        for hyp in hypotheses:
            correct = 0
            for ex in examples:
                pred = hyp.execute(np.array(ex['input']))
                if np.array_equal(pred, np.array(ex['output'])):
                    correct += 1
            hyp.support = correct

        # Select best hypothesis
        valid_hyps = [h for h in hypotheses if h.support == len(examples)]
        if valid_hyps:
            best = min(valid_hyps, key=lambda h: h.mdl())
        else:
            best = max(hypotheses, key=lambda h: h.support) if hypotheses else None

        # Execute on test
        if best:
            prediction = best.execute(test_input).tolist()
        else:
            prediction = test_input.tolist()

        return {
            'prediction': prediction,
            'hypothesis': best.program.describe() if best else None,
            'mdl': best.mdl() if best else None,
            'bits_spent': self.bits_spent,
            'num_hypotheses': len(hypotheses),
            'support': f"{int(best.support) if best else 0}/{len(examples)}"
        }

    def _generate_hypotheses(self, examples: List[Dict]) -> List[Hypothesis]:
        """Generate hypotheses using pattern detection."""
        hypotheses = []

        # Try color mapping
        color_map = PatternDetector.detect_color_mapping(examples)
        if color_map:
            hypotheses.append(Hypothesis(ColorMapOp(color_map)))

        # Try tiling
        tiling = PatternDetector.detect_tiling(examples)
        if tiling:
            if isinstance(tiling, tuple) and len(tiling) == 2:
                hypotheses.append(Hypothesis(TileOp(*tiling)))
            elif isinstance(tiling, tuple) and tiling[0] == 'flip':
                hypotheses.append(Hypothesis(TileWithFlipOp(tiling[1], tiling[2])))

        # Try rotation
        rotation = PatternDetector.detect_rotation(examples)
        if rotation:
            hypotheses.append(Hypothesis(RotateOp(rotation)))

        # Try flip
        flip = PatternDetector.detect_flip(examples)
        if flip:
            hypotheses.append(Hypothesis(FlipOp(flip)))

        # Try color map + tiling
        if color_map and tiling and isinstance(tiling, tuple) and len(tiling) == 2:
            comp = CompositeOp([ColorMapOp(color_map), TileOp(*tiling)])
            hypotheses.append(Hypothesis(comp))

        # Try tiling + flip
        if tiling and isinstance(tiling, tuple) and len(tiling) == 2:
            for axis in ['vertical', 'horizontal']:
                comp = CompositeOp([TileOp(*tiling), FlipOp(axis)])
                hypotheses.append(Hypothesis(comp))

        return hypotheses

    def _llm_fallback(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to LLM when no patterns detected."""
        examples_str = ""
        for i, ex in enumerate(task['train'][:2]):
            examples_str += f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}\n\n"

        test_str = f"Test input: {task['test'][0]['input']}"

        prompt = f"""Analyze these ARC examples and output the test result.

{examples_str}
{test_str}

Output ONLY a valid JSON array (the transformed grid)."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            self.bits_spent += 1

            pred = self._parse_grid(response.choices[0].message.content)
            return {
                'prediction': pred if pred else task['test'][0]['input'],
                'hypothesis': 'llm_fallback',
                'mdl': 10.0,
                'bits_spent': self.bits_spent,
                'num_hypotheses': 0,
                'support': '?'
            }
        except:
            return {
                'prediction': task['test'][0]['input'],
                'hypothesis': None,
                'mdl': None,
                'bits_spent': self.bits_spent,
                'num_hypotheses': 0,
                'support': '0'
            }

    def _parse_grid(self, response: str) -> Optional[List[List[int]]]:
        """Parse grid from LLM response."""
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        response = response.strip()

        try:
            grid = json.loads(response)
            if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                return grid
        except:
            pass
        return None

    def solve_directory(self, json_dir: str) -> Dict[str, Any]:
        """Solve all tasks."""
        results = {}
        json_path = Path(json_dir)

        json_files = sorted(json_path.glob("*.json"))
        print(f"Found {len(json_files)} tasks\n")

        for json_file in json_files:
            task_id = json_file.stem
            print(f"Solving {task_id}...")

            try:
                with open(json_file) as f:
                    task = json.load(f)

                self.bits_spent = 0
                result = self.solve_task(task)

                expected = task['test'][0].get('output')
                result['expected'] = expected
                result['correct'] = result['prediction'] == expected if expected else None

                results[task_id] = result

                status = "✓" if result['correct'] else "✗" if expected else "?"
                print(f"  {status} {result.get('hypothesis', 'none')} | "
                      f"MDL: {result.get('mdl', 'N/A')}, "
                      f"Support: {result.get('support', '?')}\n")

            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                results[task_id] = {'error': str(e)}

        return results

    def print_summary(self, results: Dict[str, Any]):
        """Print summary."""
        total = len(results)
        errors = sum(1 for r in results.values() if 'error' in r)
        correct = sum(1 for r in results.values() if r.get('correct'))
        evaluated = sum(1 for r in results.values() if r.get('correct') is not None)

        total_bits = sum(r.get('bits_spent', 0) for r in results.values() if 'error' not in r)
        avg_bits = total_bits / (total - errors) if total > errors else 0

        mdls = [r.get('mdl') for r in results.values() if r.get('mdl') is not None]
        avg_mdl = np.mean(mdls) if mdls else 0

        print("\n" + "="*60)
        print("LIVING MAP METRICS")
        print("="*60)
        print(f"Total: {total} | Correct: {correct}/{evaluated} ({100*correct/evaluated if evaluated else 0:.1f}%)")
        print(f"Avg bits: {avg_bits:.1f} | Avg MDL: {avg_mdl:.2f}")
        print(f"Velocity: {correct/total_bits if total_bits > 0 else 0:.4f} correct/bit")


def main():
    parser = argparse.ArgumentParser(description="Living Map ARC Solver with executable operators")
    parser.add_argument("json_dir", help="Directory with JSON files")
    parser.add_argument("model", default="gpt-4", nargs="?", help="GPT model")
    parser.add_argument("--bit-budget", type=int, default=100)
    parser.add_argument("--output", default="results.json")

    args = parser.parse_args()

    print(f"Living Map ARC Solver v3 (Executable Operators)")
    print(f"Model: {args.model} | Directory: {args.json_dir}\n")

    solver = LivingMapSolver(model=args.model, bit_budget=args.bit_budget)
    results = solver.solve_directory(args.json_dir)
    solver.print_summary(results)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.output}")


if __name__ == "__main__":
    main()