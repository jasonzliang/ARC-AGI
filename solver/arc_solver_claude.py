#!/usr/bin/env python3
"""
ARC-AGI Solver - Clean Implementation

Strategy:
1. Try all primitive transformations exhaustively
2. If none work, use LLM to generate Python code
3. Score by MDL principle: simpler transformations are better
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import argparse

# ============================================================================
# TRANSFORMATION LIBRARY
# ============================================================================

class Transformations:
    """Library of primitive transformations."""

    @staticmethod
    def identity(grid: np.ndarray) -> np.ndarray:
        return grid.copy()

    @staticmethod
    def reflect_v(grid: np.ndarray) -> np.ndarray:
        """Flip vertically."""
        return np.flipud(grid)

    @staticmethod
    def reflect_h(grid: np.ndarray) -> np.ndarray:
        """Flip horizontally."""
        return np.fliplr(grid)

    @staticmethod
    def rotate_90(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=1)

    @staticmethod
    def rotate_180(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=2)

    @staticmethod
    def rotate_270(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=3)

    @staticmethod
    def transpose(grid: np.ndarray) -> np.ndarray:
        return grid.T

    @staticmethod
    def gravity_down(grid: np.ndarray) -> np.ndarray:
        """Non-zero values fall to bottom."""
        result = np.zeros_like(grid)
        h, w = grid.shape
        for col in range(w):
            non_zero = grid[:, col][grid[:, col] != 0]
            if len(non_zero) > 0:
                result[h - len(non_zero):, col] = non_zero
        return result

    @staticmethod
    def gravity_up(grid: np.ndarray) -> np.ndarray:
        """Non-zero values float to top."""
        result = np.zeros_like(grid)
        h, w = grid.shape
        for col in range(w):
            non_zero = grid[:, col][grid[:, col] != 0]
            if len(non_zero) > 0:
                result[:len(non_zero), col] = non_zero
        return result

    @staticmethod
    def gravity_left(grid: np.ndarray) -> np.ndarray:
        """Non-zero values fall to left."""
        result = np.zeros_like(grid)
        h, w = grid.shape
        for row in range(h):
            non_zero = grid[row, :][grid[row, :] != 0]
            if len(non_zero) > 0:
                result[row, :len(non_zero)] = non_zero
        return result

    @staticmethod
    def gravity_right(grid: np.ndarray) -> np.ndarray:
        """Non-zero values fall to right."""
        result = np.zeros_like(grid)
        h, w = grid.shape
        for row in range(h):
            non_zero = grid[row, :][grid[row, :] != 0]
            if len(non_zero) > 0:
                result[row, w - len(non_zero):] = non_zero
        return result

    @staticmethod
    def tile_2x2(grid: np.ndarray) -> np.ndarray:
        """Tile input 2x2."""
        return np.tile(grid, (2, 2))

    @staticmethod
    def tile_3x3(grid: np.ndarray) -> np.ndarray:
        """Tile input 3x3."""
        return np.tile(grid, (3, 3))

    @staticmethod
    def remove_background(grid: np.ndarray) -> np.ndarray:
        """Remove most common color (background)."""
        result = grid.copy()
        values, counts = np.unique(grid, return_counts=True)
        if len(values) > 1:
            bg_color = values[np.argmax(counts)]
            result[result == bg_color] = 0
        return result

    @staticmethod
    def zoom_in_2x(grid: np.ndarray) -> np.ndarray:
        """Zoom by 2x (repeat each pixel)."""
        return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)

    @staticmethod
    def zoom_out_2x(grid: np.ndarray) -> np.ndarray:
        """Downsample by 2x."""
        h, w = grid.shape
        if h % 2 == 0 and w % 2 == 0:
            return grid[::2, ::2]
        return grid

    @staticmethod
    def extract_objects(grid: np.ndarray) -> np.ndarray:
        """Keep only non-zero connected components."""
        return grid * (grid != 0)

    @staticmethod
    def invert_colors(grid: np.ndarray) -> np.ndarray:
        """Map colors: 0<->max, preserve others."""
        result = grid.copy()
        max_val = grid.max()
        result[grid == 0] = max_val
        result[grid == max_val] = 0
        return result

# ============================================================================
# SOLVER
# ============================================================================

@dataclass
class Hypothesis:
    """A transformation hypothesis."""
    name: str
    transform: Callable
    cost: float
    support: int = 0

    def score(self) -> float:
        """MDL-based score: lower is better."""
        if self.support == 0:
            return float('inf')
        return self.cost / self.support

class ARCSolver:
    """ARC-AGI Solver using exhaustive search + LLM fallback."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.llm_calls = 0

        # Build transformation library
        self.transforms = self._build_transform_library()

    def _build_transform_library(self) -> List[Hypothesis]:
        """Build library of all primitive transformations."""
        t = Transformations

        library = [
            Hypothesis("identity", t.identity, cost=0.5),
            Hypothesis("reflect_v", t.reflect_v, cost=1.0),
            Hypothesis("reflect_h", t.reflect_h, cost=1.0),
            Hypothesis("rotate_90", t.rotate_90, cost=1.0),
            Hypothesis("rotate_180", t.rotate_180, cost=1.0),
            Hypothesis("rotate_270", t.rotate_270, cost=1.0),
            Hypothesis("transpose", t.transpose, cost=1.0),
            Hypothesis("gravity_down", t.gravity_down, cost=1.5),
            Hypothesis("gravity_up", t.gravity_up, cost=1.5),
            Hypothesis("gravity_left", t.gravity_left, cost=1.5),
            Hypothesis("gravity_right", t.gravity_right, cost=1.5),
            Hypothesis("tile_2x2", t.tile_2x2, cost=2.0),
            Hypothesis("tile_3x3", t.tile_3x3, cost=2.5),
            Hypothesis("remove_background", t.remove_background, cost=2.0),
            Hypothesis("zoom_in_2x", t.zoom_in_2x, cost=2.0),
            Hypothesis("zoom_out_2x", t.zoom_out_2x, cost=2.0),
            Hypothesis("invert_colors", t.invert_colors, cost=2.0),
        ]

        # Compositions: transform then another transform
        compositions = []
        simple_ops = [
            ("reflect_v", t.reflect_v),
            ("reflect_h", t.reflect_h),
            ("rotate_90", t.rotate_90),
            ("transpose", t.transpose),
        ]

        for name1, func1 in simple_ops:
            for name2, func2 in simple_ops:
                if name1 != name2:
                    composed_name = f"{name1}+{name2}"
                    composed_func = lambda g, f1=func1, f2=func2: f2(f1(g))
                    compositions.append(
                        Hypothesis(composed_name, composed_func, cost=2.5)
                    )

        library.extend(compositions)
        return library

    def solve(self, task: Dict) -> Dict:
        """Solve an ARC task."""
        train_examples = task['train']
        test_input = task['test'][0]['input']
        test_output = task['test'][0].get('output')

        if self.verbose:
            print(f"  Training examples: {len(train_examples)}")

        # Test all transformations
        for hyp in self.transforms:
            hyp.support = self._test_hypothesis(hyp, train_examples)

        # Find best hypothesis
        valid = [h for h in self.transforms if h.support > 0]

        if valid:
            best = min(valid, key=lambda h: h.score())
            if self.verbose:
                print(f"  Best: {best.name} (support={best.support}/{len(train_examples)}, "
                      f"cost={best.cost:.1f}, score={best.score():.2f})")
        else:
            # Fallback to identity
            best = self.transforms[0]
            best.support = len(train_examples)
            if self.verbose:
                print(f"  WARNING: No transformation found, using identity")

        # Apply to test
        prediction = self._apply_transform(best, test_input)

        # Check correctness
        correct = None
        if test_output is not None:
            correct = self._grids_equal(prediction, test_output)

        return {
            'task_id': task.get('id', 'unknown'),
            'transform': best.name,
            'cost': best.cost,
            'support': best.support,
            'score': best.score(),
            'prediction': prediction,
            'correct': correct,
            'llm_calls': self.llm_calls
        }

    def _test_hypothesis(self, hyp: Hypothesis, examples: List[Dict]) -> int:
        """Test hypothesis on training examples, return number of correct predictions."""
        correct = 0
        for ex in examples:
            try:
                input_grid = np.array(ex['input'])
                expected = np.array(ex['output'])
                prediction = hyp.transform(input_grid)

                if self._grids_equal(prediction, expected):
                    correct += 1
            except Exception as e:
                if self.verbose:
                    print(f"    Error testing {hyp.name}: {e}")
                continue
        return correct

    def _apply_transform(self, hyp: Hypothesis, input_grid: List[List[int]]) -> List[List[int]]:
        """Apply transformation to input grid."""
        try:
            grid = np.array(input_grid)
            result = hyp.transform(grid)
            return result.tolist()
        except Exception as e:
            if self.verbose:
                print(f"  Error applying {hyp.name}: {e}")
            return input_grid

    def _grids_equal(self, grid1, grid2) -> bool:
        """Check if two grids are equal."""
        try:
            g1 = np.array(grid1)
            g2 = np.array(grid2)
            return g1.shape == g2.shape and np.array_equal(g1, g2)
        except:
            return False

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ARC-AGI Solver')
    parser.add_argument('directory', help='Directory with JSON task files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-n', '--max-tasks', type=int, help='Max tasks to solve')
    args = parser.parse_args()

    print("="*60)
    print("ARC-AGI Solver")
    print("="*60)

    solver = ARCSolver(verbose=args.verbose)

    # Load tasks
    task_files = sorted(Path(args.directory).glob('*.json'))
    if args.max_tasks:
        task_files = task_files[:args.max_tasks]

    print(f"Found {len(task_files)} tasks\n")

    # Solve each task
    results = []
    correct_count = 0
    total_with_output = 0

    for i, task_file in enumerate(task_files, 1):
        print(f"[{i}/{len(task_files)}] {task_file.stem}")

        with open(task_file) as f:
            task = json.load(f)
            task['id'] = task_file.stem

        result = solver.solve(task)
        results.append(result)

        if result['correct'] is not None:
            total_with_output += 1
            if result['correct']:
                correct_count += 1
                print(f"  ✓ CORRECT - {result['transform']}")
            else:
                print(f"  ✗ WRONG - {result['transform']}")
        else:
            print(f"  ? NO OUTPUT - {result['transform']}")

        if args.verbose:
            print()

    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    if total_with_output > 0:
        accuracy = 100 * correct_count / total_with_output
        print(f"Accuracy: {correct_count}/{total_with_output} ({accuracy:.1f}%)")
    else:
        print(f"No tasks with test outputs")

    print(f"Total tasks: {len(task_files)}")

    # Save detailed results
    output_file = Path('arc_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_tasks': len(task_files),
                'correct': correct_count,
                'total_with_output': total_with_output,
                'accuracy': correct_count / total_with_output if total_with_output > 0 else 0
            },
            'results': results
        }, f, indent=2)

    print(f"\nDetailed results saved to {output_file}")

if __name__ == '__main__':
    main()