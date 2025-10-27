#!/usr/bin/env python3
"""
Living Map ARC-AGI Solver - Improved Version

Fixes:
1. Better operator parsing from LLM responses
2. Fallback to exhaustive search of primitives
3. LLM code generation when primitives fail
4. Proper comparison logic for grid matching
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from openai import OpenAI

# ============================================================================
# DATA STRUCTURES
# ============================================================================

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
        'transpose': 1.0,
        'llm_code': 5.0,  # High cost for LLM-generated code
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
    code: Optional[str] = None  # For LLM-generated code

    def mdl(self) -> float:
        """Minimum Description Length in bits."""
        return self.program.cost()

    def score(self) -> float:
        """Score = MDL / support (minimize: lower is better)."""
        return float('inf') if self.support == 0 else self.mdl() / self.support

# ============================================================================
# SYMBOLIC EXECUTION
# ============================================================================

class SymbolicExecutor:
    """Deterministic execution for common operators."""

    @staticmethod
    def execute(op_name: str, grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Execute operator symbolically."""
        arr = np.array(grid)

        ops = {
            'identity': lambda: grid,
            'tile_pattern': lambda: SymbolicExecutor._tile_3x3(arr),
            'reflect_vertical': lambda: arr[::-1].tolist(),
            'reflect_horizontal': lambda: arr[:, ::-1].tolist(),
            'rotate_90': lambda: np.rot90(arr, k=-1).tolist(),
            'rotate_180': lambda: np.rot90(arr, k=2).tolist(),
            'rotate_270': lambda: np.rot90(arr, k=1).tolist(),
            'transpose': lambda: arr.T.tolist(),
            'gravity_down': lambda: SymbolicExecutor._gravity(arr, 'down'),
            'gravity_up': lambda: SymbolicExecutor._gravity(arr, 'up'),
            'gravity_left': lambda: SymbolicExecutor._gravity(arr, 'left'),
            'gravity_right': lambda: SymbolicExecutor._gravity(arr, 'right'),
        }

        return ops.get(op_name, lambda: None)()

    @staticmethod
    def _tile_3x3(arr: np.ndarray) -> List[List[int]]:
        """Tile 3x3 with alternating horizontal flips."""
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
        """Solve a single ARC task."""
        if self.verbose:
            print(f"  Training examples: {len(task['train'])}")

        # First, try all primitive operators exhaustively
        hypotheses = self._try_all_primitives(task['train'])

        # If no primitive works, use LLM to generate hypotheses
        if all(h.support == 0 for h in hypotheses):
            if self.verbose:
                print(f"  No primitive worked, consulting LLM...")
            analysis = self._analyze_examples(task['train'])
            llm_hypotheses = self._generate_llm_hypotheses(task['train'], analysis)
            hypotheses.extend(llm_hypotheses)

        # Select best by score
        valid_hypotheses = [h for h in hypotheses if h.support > 0]
        if not valid_hypotheses:
            if self.verbose:
                print(f"  WARNING: No valid hypotheses found!")
            best_hyp = hypotheses[0]  # Default to identity
        else:
            best_hyp = min(valid_hypotheses, key=lambda h: h.score())

        if self.verbose:
            print(f"  Best: {best_hyp.program.describe()} "
                  f"(support={best_hyp.support}, score={best_hyp.score():.2f})")

        # Apply to test
        test_input = task['test'][0]['input']
        prediction = self._execute(best_hyp, test_input)

        return {
            'hypothesis': best_hyp.program.describe(),
            'score': best_hyp.score(),
            'support': best_hyp.support,
            'mdl': best_hyp.mdl(),
            'prediction': prediction,
            'bits_spent': self.bits_spent
        }

    def _try_all_primitives(self, examples: List[Dict]) -> List[Hypothesis]:
        """Try all primitive operators systematically."""
        all_ops = [
            'identity', 'tile_pattern', 'reflect_vertical', 'reflect_horizontal',
            'rotate_90', 'rotate_180', 'rotate_270', 'transpose',
            'gravity_down', 'gravity_up', 'gravity_left', 'gravity_right'
        ]

        hypotheses = []
        for op_name in all_ops:
            h = Hypothesis(PrimitiveOp(op_name))
            h.support = self._compute_support(h, examples)
            hypotheses.append(h)

            if self.verbose and h.support > 0:
                print(f"  ✓ {op_name}: support={h.support}/{len(examples)}, "
                      f"score={h.score():.2f}")

        return hypotheses

    def _analyze_examples(self, examples: List[Dict]) -> Dict:
        """Quick analysis of training examples."""
        analysis = {'examples_summary': []}

        for i, ex in enumerate(examples):
            in_arr = np.array(ex['input'])
            out_arr = np.array(ex['output'])
            analysis['examples_summary'].append({
                'index': i,
                'in_shape': in_arr.shape,
                'out_shape': out_arr.shape,
                'in_colors': set(in_arr.flatten().tolist()) - {0},
                'out_colors': set(out_arr.flatten().tolist()) - {0}
            })

        return analysis

    def _generate_llm_hypotheses(self, examples: List[Dict], analysis: Dict) -> List[Hypothesis]:
        """Use LLM to generate code-based hypotheses."""
        if self.verbose:
            print(f"  Requesting LLM code generation...")

        prompt = self._build_code_gen_prompt(examples, analysis)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._code_gen_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            self.bits_spent += 5  # High cost for LLM code generation

            code = self._extract_code(response.choices[0].message.content)
            if code:
                h = Hypothesis(PrimitiveOp('llm_code'), code=code)
                h.support = self._compute_support(h, examples)
                if self.verbose:
                    print(f"  LLM code: support={h.support}/{len(examples)}")
                return [h]

        except Exception as e:
            print(f"  Error in LLM code generation: {e}")

        return []

    def _code_gen_system_prompt(self) -> str:
        """System prompt for code generation."""
        return """You are an ARC-AGI solver. Generate a Python function that transforms the input grid to the output grid.

Requirements:
- Function signature: def transform(grid: List[List[int]]) -> List[List[int]]:
- Input/output are 2D lists of integers
- Use numpy if needed (import as np)
- Be concise and efficient
- Only return the function, no explanation"""

    def _build_code_gen_prompt(self, examples: List[Dict], analysis: Dict) -> str:
        """Build prompt for code generation."""
        prompt = "Generate a transform function for these examples:\n\n"

        for i, ex in enumerate(examples[:2], 1):
            in_arr = np.array(ex['input'])
            out_arr = np.array(ex['output'])
            prompt += f"Example {i}:\n"
            prompt += f"Input ({in_arr.shape}):\n{in_arr.tolist()}\n"
            prompt += f"Output ({out_arr.shape}):\n{out_arr.tolist()}\n\n"

        return prompt

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        # Look for code blocks
        if '```python' in response:
            start = response.find('```python') + 9
            end = response.find('```', start)
            return response[start:end].strip()
        elif 'def transform' in response:
            # Try to extract function directly
            lines = []
            in_function = False
            for line in response.split('\n'):
                if 'def transform' in line:
                    in_function = True
                if in_function:
                    lines.append(line)
                    if line and not line[0].isspace() and 'def transform' not in line:
                        break
            return '\n'.join(lines).strip()
        return None

    def _compute_support(self, hypothesis: Hypothesis, examples: List[Dict]) -> float:
        """Compute actual support by validation."""
        correct = 0

        for ex in examples:
            try:
                prediction = self._execute(hypothesis, ex['input'])
                expected = ex['output']

                # Proper comparison handling nested lists and numpy arrays
                if self._grids_equal(prediction, expected):
                    correct += 1
            except Exception as e:
                if self.verbose:
                    print(f"    Error in {hypothesis.program.name}: {e}")
                continue

        return float(correct)

    def _grids_equal(self, grid1, grid2) -> bool:
        """Compare two grids for equality."""
        try:
            arr1 = np.array(grid1)
            arr2 = np.array(grid2)
            return arr1.shape == arr2.shape and np.array_equal(arr1, arr2)
        except:
            return grid1 == grid2

    def _execute(self, hypothesis: Hypothesis, input_grid: List[List[int]]) -> List[List[int]]:
        """Execute hypothesis."""
        if hypothesis.code:
            # Execute LLM-generated code
            try:
                namespace = {'np': np, 'List': List}
                exec(hypothesis.code, namespace)
                transform = namespace.get('transform')
                if transform:
                    result = transform(input_grid)
                    return result if isinstance(result, list) else result.tolist()
            except Exception as e:
                if self.verbose:
                    print(f"    Error executing LLM code: {e}")
                return input_grid
        else:
            # Try symbolic execution
            result = SymbolicExecutor.execute(hypothesis.program.name, input_grid)
            return result if result is not None else input_grid

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Living Map ARC-AGI Solver')
    parser.add_argument('directory', help='Directory with JSON task files')
    parser.add_argument('model', default='gpt-4o', nargs='?', help='OpenAI model')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print(f"Living Map ARC-AGI Solver - Improved")
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
        is_correct = solver._grids_equal(result['prediction'], expected) if expected else None
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
    if total > 0:
        print(f"Correct: {correct}/{total} ({100*correct/total:.1f}%)")
    else:
        print(f"Correct: {correct}/{total} (no test outputs available)")

    avg_bits = np.mean([r['bits_spent'] for r in results.values()])
    print(f"Average bits: {avg_bits:.1f}")

    # Save results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results.json")

if __name__ == '__main__':
    main()