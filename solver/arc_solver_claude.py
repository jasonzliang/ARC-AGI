#!/usr/bin/env python3
"""
Living Map ARC-AGI Solver - Final Compact Version

Core principle: Find the SHORTEST program that explains all training examples.
Score = MDL / support (minimize: prefer simple rules that work well)
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from openai import OpenAI

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Object:
    """A connected component in a grid."""
    color: int
    pixels: List[Tuple[int, int]]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        rows, cols = zip(*self.pixels)
        return (min(rows), min(cols), max(rows), max(cols))

@dataclass
class ObjectGraph:
    """Structured representation of a grid."""
    objects: List[Object]

    def __init__(self, grid: np.ndarray):
        self.objects = self._extract_objects(grid)

    def _extract_objects(self, grid: np.ndarray) -> List[Object]:
        """Extract connected components."""
        objects = []
        seen = set()
        h, w = grid.shape

        for r in range(h):
            for c in range(w):
                if (r, c) in seen or grid[r, c] == 0:
                    continue

                # BFS to find connected component
                color = grid[r, c]
                pixels = []
                queue = [(r, c)]
                seen.add((r, c))

                while queue:
                    cr, cc = queue.pop(0)
                    pixels.append((cr, cc))

                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w and
                            (nr, nc) not in seen and grid[nr, nc] == color):
                            queue.append((nr, nc))
                            seen.add((nr, nc))

                objects.append(Object(color, pixels))

        return objects

@dataclass
class PrimitiveOp:
    """A primitive transformation operator."""
    name: str
    params: Dict = field(default_factory=dict)

    BASE_COSTS = {
        'identity': 0.5,
        'color_map': 1.5,
        'reflect_vertical': 1.0,
        'reflect_horizontal': 1.0,
        'rotate_90': 1.0,
        'rotate_180': 1.0,
        'rotate_270': 1.0,
        'gravity_down': 1.5,
        'gravity_up': 1.5,
        'gravity_left': 1.5,
        'gravity_right': 1.5,
        'tile_pattern': 2.5,
    }

    def cost(self) -> float:
        return self.BASE_COSTS.get(self.name, 3.0) + 0.5 * len(self.params)

    def describe(self) -> str:
        if self.params:
            params_str = ', '.join(f'{k}={v}' for k, v in self.params.items())
            return f"{self.name}({params_str})"
        return self.name

@dataclass
class Hypothesis:
    """A transformation hypothesis with support validation."""
    program: PrimitiveOp
    support: float = 0.0

    def mdl(self) -> float:
        """Minimum Description Length in bits."""
        return self.program.cost()

    def score(self) -> float:
        """Score = MDL / support (minimize: lower is better).

        Interpretation: bits per explained example
        - Low MDL + High support → Low score (BEST)
        - High MDL or Low support → High score (WORSE)
        """
        return float('inf') if self.support == 0 else self.mdl() / self.support

# ============================================================================
# SYMBOLIC EXECUTION (0 bits)
# ============================================================================

class SymbolicExecutor:
    """Deterministic execution for common operators."""

    @staticmethod
    def execute(op_name: str, grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Execute operator symbolically. Returns None if not available."""
        arr = np.array(grid)

        if op_name == 'identity':
            return grid

        elif op_name == 'tile_pattern':
            return SymbolicExecutor._tile_3x3(arr)

        elif op_name == 'reflect_vertical':
            return grid[::-1]

        elif op_name == 'reflect_horizontal':
            return [row[::-1] for row in grid]

        elif op_name == 'rotate_90':
            return np.rot90(arr, k=-1).tolist()

        elif op_name == 'rotate_180':
            return np.rot90(arr, k=2).tolist()

        elif op_name == 'rotate_270':
            return np.rot90(arr, k=1).tolist()

        elif op_name == 'gravity_down':
            return SymbolicExecutor._gravity(arr, 'down')

        elif op_name == 'gravity_up':
            return SymbolicExecutor._gravity(arr, 'up')

        elif op_name == 'gravity_left':
            return SymbolicExecutor._gravity(arr, 'left')

        elif op_name == 'gravity_right':
            return SymbolicExecutor._gravity(arr, 'right')

        return None

    @staticmethod
    def _tile_3x3(arr: np.ndarray) -> List[List[int]]:
        """Tile 3x3 with alternating horizontal flips."""
        h, w = arr.shape
        output = []

        for block_idx in range(3):
            for row in arr:
                if block_idx % 2 == 0:
                    output.append(np.tile(row, 3).tolist())
                else:
                    output.append(np.tile(row[::-1], 3).tolist())

        return output

    @staticmethod
    def _gravity(arr: np.ndarray, direction: str) -> List[List[int]]:
        """Apply gravity in specified direction."""
        h, w = arr.shape
        output = np.zeros_like(arr)

        if direction == 'down':
            for col in range(w):
                non_zero = [arr[row, col] for row in range(h) if arr[row, col] != 0]
                for i, val in enumerate(non_zero):
                    output[h - len(non_zero) + i, col] = val

        elif direction == 'up':
            for col in range(w):
                non_zero = [arr[row, col] for row in range(h) if arr[row, col] != 0]
                for i, val in enumerate(non_zero):
                    output[i, col] = val

        elif direction == 'left':
            for row in range(h):
                non_zero = [arr[row, col] for col in range(w) if arr[row, col] != 0]
                for i, val in enumerate(non_zero):
                    output[row, i] = val

        elif direction == 'right':
            for row in range(h):
                non_zero = [arr[row, col] for col in range(w) if arr[row, col] != 0]
                for i, val in enumerate(non_zero):
                    output[row, w - len(non_zero) + i] = val

        return output.tolist()

# ============================================================================
# LIVING MAP SOLVER
# ============================================================================

class LivingMapSolver:
    """Compression-guided search for transformation rules."""

    def __init__(self, model: str = "gpt-4", verbose: bool = False):
        self.client = OpenAI()
        self.model = model
        self.verbose = verbose
        self.bits_spent = 0

    def solve_task(self, task: Dict) -> Dict:
        """Main solving pipeline."""
        if self.verbose:
            print(f"\n{'='*60}\nSOLVING TASK\n{'='*60}")

        # Phase 1: Typed Perception
        train_examples = task['train']
        test_input = task['test'][0]['input']

        if self.verbose:
            print(f"Phase 1: Typed Perception")
            print(f"  Training examples: {len(train_examples)}")

        # Phase 2: Structural Analysis
        analysis = self._analyze_structure(train_examples)

        if self.verbose:
            print(f"\nPhase 2: Structural Analysis")
            print(f"  Shape changes: {analysis['shape_changes']}")

        # Phase 3: Generate Hypotheses
        hypotheses = self._generate_hypotheses(train_examples, analysis)

        if self.verbose:
            print(f"\nPhase 3: Generated {len(hypotheses)} hypotheses")
            for i, h in enumerate(hypotheses, 1):
                print(f"  {i}. {h.program.describe()}: MDL={h.mdl():.1f}, "
                      f"Support={h.support:.1f}, Score={h.score():.2f}")

        # Phase 4: Select Best
        best = min(hypotheses, key=lambda h: h.score()) if hypotheses else None

        if self.verbose and best:
            print(f"\nPhase 4: Best hypothesis: {best.program.describe()}")
            print(f"  Score: {best.score():.2f}")

        # Phase 5: Execute
        prediction = self._execute(best, test_input) if best else None

        if self.verbose:
            print(f"\nTotal bits spent: {self.bits_spent}\n{'='*60}")

        return {
            'prediction': prediction,
            'hypothesis': best.program.describe() if best else None,
            'mdl': best.mdl() if best else None,
            'support': best.support if best else 0,
            'score': best.score() if best else float('inf'),
            'bits_spent': self.bits_spent,
            'num_hypotheses': len(hypotheses),
        }

    def _analyze_structure(self, examples: List[Dict]) -> Dict:
        """Extract structural patterns."""
        analysis = {'shape_changes': [], 'colors': set()}

        for ex in examples:
            in_shape = np.array(ex['input']).shape
            out_shape = np.array(ex['output']).shape
            analysis['shape_changes'].append({
                'in': in_shape,
                'out': out_shape,
                'scaled': out_shape[0] % in_shape[0] == 0 if in_shape[0] > 0 else False
            })
            analysis['colors'].update(np.unique(ex['input']))
            analysis['colors'].update(np.unique(ex['output']))

        return analysis

    def _generate_hypotheses(self, examples: List[Dict], analysis: Dict) -> List[Hypothesis]:
        """Generate and validate hypotheses."""
        if self.verbose:
            print(f"  Calling LLM for hypotheses...")

        # Build prompt
        prompt = self._build_prompt(examples, analysis)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )

            self.bits_spent += 1

            # Parse LLM response
            hypotheses = self._parse_hypotheses(response.choices[0].message.content)

            # Validate support
            for h in hypotheses:
                h.support = self._compute_support(h, examples)

            if self.verbose:
                print(f"  Validated {len(hypotheses)} hypotheses")

            return hypotheses

        except Exception as e:
            print(f"Error generating hypotheses: {e}")
            return [Hypothesis(PrimitiveOp('identity'), len(examples))]

    def _system_prompt(self) -> str:
        """System prompt for hypothesis generation."""
        return """You are an ARC-AGI solver. Propose transformation rules using these operators:

OPERATORS (bits):
- identity (0.5) - no change
- tile_pattern (2.5) - tile 3x3 with flips
- color_map (1.5) - remap colors
- reflect_vertical/horizontal (1.0) - flip
- rotate_90/180/270 (1.0) - rotate
- gravity_down/up/left/right (1.5) - objects fall

OUTPUT FORMAT (exactly):
HYPOTHESIS 1: [operator]
MDL: [cost]
Support: [count]/[total]

Propose 10-15 hypotheses from simplest to most complex."""

    def _build_prompt(self, examples: List[Dict], analysis: Dict) -> str:
        """Build user prompt with examples."""
        prompt = f"Analyze these {len(examples)} examples and propose transformation rules.\n\n"

        for i, ex in enumerate(examples[:2], 1):
            in_arr = np.array(ex['input'])
            out_arr = np.array(ex['output'])
            prompt += f"Example {i}:\n"
            prompt += f"  Input: {in_arr.shape}, colors: {set(in_arr.flatten()) - {0}}\n"
            prompt += f"  Output: {out_arr.shape}, colors: {set(out_arr.flatten()) - {0}}\n\n"

        if analysis['shape_changes'][0]['scaled']:
            prompt += "Note: Output is scaled/tiled from input.\n"

        return prompt

    def _parse_hypotheses(self, response: str) -> List[Hypothesis]:
        """Parse LLM response into hypotheses."""
        hypotheses = []
        lines = response.split('\n')

        current_desc = None

        for line in lines:
            line = line.strip()

            if line.upper().startswith('HYPOTHESIS'):
                if ':' in line:
                    current_desc = line.split(':', 1)[1].strip().lower()

                    # Map to known operators
                    if 'tile' in current_desc:
                        op = PrimitiveOp('tile_pattern', {'n_rows': 3, 'n_cols': 3})
                    elif 'reflect' in current_desc and 'vertical' in current_desc:
                        op = PrimitiveOp('reflect_vertical')
                    elif 'reflect' in current_desc and 'horizontal' in current_desc:
                        op = PrimitiveOp('reflect_horizontal')
                    elif 'rotate_90' in current_desc:
                        op = PrimitiveOp('rotate_90')
                    elif 'rotate_180' in current_desc:
                        op = PrimitiveOp('rotate_180')
                    elif 'rotate_270' in current_desc:
                        op = PrimitiveOp('rotate_270')
                    elif 'gravity' in current_desc and 'down' in current_desc:
                        op = PrimitiveOp('gravity_down')
                    elif 'gravity' in current_desc and 'up' in current_desc:
                        op = PrimitiveOp('gravity_up')
                    elif 'gravity' in current_desc and 'left' in current_desc:
                        op = PrimitiveOp('gravity_left')
                    elif 'gravity' in current_desc and 'right' in current_desc:
                        op = PrimitiveOp('gravity_right')
                    elif 'color' in current_desc or 'map' in current_desc:
                        op = PrimitiveOp('color_map')
                    else:
                        op = PrimitiveOp('identity')

                    hypotheses.append(Hypothesis(op))

        return hypotheses if hypotheses else [Hypothesis(PrimitiveOp('identity'))]

    def _compute_support(self, hypothesis: Hypothesis, examples: List[Dict]) -> float:
        """Compute actual support by validation."""
        correct = 0

        for ex in examples:
            try:
                prediction = self._execute(hypothesis, ex['input'])
                if prediction == ex['output']:
                    correct += 1
            except:
                continue

        return float(correct)

    def _execute(self, hypothesis: Hypothesis, input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Execute hypothesis using symbolic execution."""
        op_name = hypothesis.program.name

        # Try symbolic execution (0 bits)
        result = SymbolicExecutor.execute(op_name, input_grid)

        if result is not None:
            if self.verbose:
                print(f"  ✓ Symbolic execution ({op_name})")
            return result

        # No symbolic executor available
        if self.verbose:
            print(f"  ✗ No symbolic executor for {op_name}")

        return input_grid  # Fallback to identity

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Living Map ARC-AGI Solver')
    parser.add_argument('directory', help='Directory with JSON task files')
    parser.add_argument('model', default='gpt-4o', nargs='?', help='OpenAI model')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print(f"Living Map ARC-AGI Solver")
    print(f"Model: {args.model}")
    print(f"Directory: {args.directory}\n")

    solver = LivingMapSolver(model=args.model, verbose=args.verbose)

    # Solve all tasks
    task_files = sorted(Path(args.directory).glob('*.json'))
    results = {}
    correct = 0
    total = 0

    for task_file in task_files:
        task_id = task_file.stem
        print(f"Solving {task_id}...")

        with open(task_file) as f:
            task = json.load(f)

        solver.bits_spent = 0
        result = solver.solve_task(task)

        # Check correctness
        expected = task['test'][0].get('output')
        is_correct = result['prediction'] == expected if expected else None
        result['correct'] = is_correct

        if is_correct:
            correct += 1
        if expected:
            total += 1

        results[task_id] = result

        status = "✓" if is_correct else "✗" if expected else "?"
        print(f"  {status} Score: {result['score']:.2f}, Bits: {result['bits_spent']}, "
              f"Rule: {result['hypothesis']}\n")

    # Summary
    print(f"{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Correct: {correct}/{total} ({100*correct/total:.1f}%)")

    avg_bits = np.mean([r['bits_spent'] for r in results.values()])
    print(f"Average bits: {avg_bits:.1f}")

    # Save results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results.json")

if __name__ == '__main__':
    main()