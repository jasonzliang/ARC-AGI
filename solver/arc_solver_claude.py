#!/usr/bin/env python3
"""
Compositional ARC Solver - Multi-Step Transformations

Implements Living Map principle of "subadditive composition":
- Single steps first (cheap)
- 2-step compositions (with glue discount)
- 3-step compositions (for complex tasks)

MDL principle: Compositions should compress (whole < sum of parts)

Accuracy: 2/400 (0.5%) on eval using gpt-4o
"""


import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import argparse
from itertools import combinations, permutations

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

# ============================================================================
# TRANSFORMATIONS
# ============================================================================

class Transforms:
    """Primitive transformation library."""

    @staticmethod
    def identity(g: np.ndarray) -> np.ndarray:
        return g.copy()

    @staticmethod
    def reflect_v(g: np.ndarray) -> np.ndarray:
        return np.flipud(g)

    @staticmethod
    def reflect_h(g: np.ndarray) -> np.ndarray:
        return np.fliplr(g)

    @staticmethod
    def rotate_90(g: np.ndarray) -> np.ndarray:
        return np.rot90(g, k=1)

    @staticmethod
    def rotate_180(g: np.ndarray) -> np.ndarray:
        return np.rot90(g, k=2)

    @staticmethod
    def rotate_270(g: np.ndarray) -> np.ndarray:
        return np.rot90(g, k=3)

    @staticmethod
    def transpose(g: np.ndarray) -> np.ndarray:
        return g.T

    @staticmethod
    def gravity_down(g: np.ndarray) -> np.ndarray:
        result = np.zeros_like(g)
        h, w = g.shape
        for col in range(w):
            non_zero = g[:, col][g[:, col] != 0]
            if len(non_zero) > 0:
                result[h - len(non_zero):, col] = non_zero
        return result

    @staticmethod
    def gravity_up(g: np.ndarray) -> np.ndarray:
        result = np.zeros_like(g)
        h, w = g.shape
        for col in range(w):
            non_zero = g[:, col][g[:, col] != 0]
            if len(non_zero) > 0:
                result[:len(non_zero), col] = non_zero
        return result

    @staticmethod
    def gravity_left(g: np.ndarray) -> np.ndarray:
        result = np.zeros_like(g)
        h, w = g.shape
        for row in range(h):
            non_zero = g[row, :][g[row, :] != 0]
            if len(non_zero) > 0:
                result[row, :len(non_zero)] = non_zero
        return result

    @staticmethod
    def gravity_right(g: np.ndarray) -> np.ndarray:
        result = np.zeros_like(g)
        h, w = g.shape
        for row in range(h):
            non_zero = g[row, :][g[row, :] != 0]
            if len(non_zero) > 0:
                result[row, w - len(non_zero):] = non_zero
        return result

    @staticmethod
    def tile_2x2(g: np.ndarray) -> np.ndarray:
        return np.tile(g, (2, 2))

    @staticmethod
    def tile_3x3(g: np.ndarray) -> np.ndarray:
        return np.tile(g, (3, 3))

    @staticmethod
    def zoom_2x(g: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)

    @staticmethod
    def compress_2x(g: np.ndarray) -> np.ndarray:
        h, w = g.shape
        if h % 2 == 0 and w % 2 == 0:
            return g[::2, ::2]
        return g

    @staticmethod
    def extract_nonzero(g: np.ndarray) -> np.ndarray:
        """Extract bounding box of non-zero elements."""
        if np.all(g == 0):
            return g
        rows = np.any(g != 0, axis=1)
        cols = np.any(g != 0, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return g[rmin:rmax+1, cmin:cmax+1]

# ============================================================================
# PROGRAM & HYPOTHESIS
# ============================================================================

@dataclass
class Program:
    """Sequence of transformations."""
    steps: List[str]

    def execute(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """Execute all steps in sequence."""
        result = grid.copy()
        for step in self.steps:
            if hasattr(Transforms, step):
                try:
                    result = getattr(Transforms, step)(result)
                except:
                    return None
            else:
                return None
        return result

    def mdl(self) -> float:
        """Minimum Description Length with subadditive composition."""
        if not self.steps:
            return 0.5

        # Base costs
        costs = {
            'identity': 0.5,
            'reflect_v': 1.0, 'reflect_h': 1.0,
            'rotate_90': 1.0, 'rotate_180': 1.0, 'rotate_270': 1.0,
            'transpose': 1.0,
            'gravity_down': 1.5, 'gravity_up': 1.5,
            'gravity_left': 1.5, 'gravity_right': 1.5,
            'tile_2x2': 2.0, 'tile_3x3': 2.5,
            'zoom_2x': 2.0, 'compress_2x': 2.0,
            'extract_nonzero': 2.0,
        }

        total_cost = sum(costs.get(step, 3.0) for step in self.steps)

        # SUBADDITIVE COMPOSITION DISCOUNT
        # Living Map principle: well-composed programs compress
        if len(self.steps) > 1:
            # Each additional step gets a discount
            discount = 0.5 * (len(self.steps) - 1)
            total_cost = max(1.0, total_cost - discount)

        return total_cost

    def __str__(self) -> str:
        if not self.steps:
            return "identity"
        return " → ".join(self.steps)

@dataclass
class Hypothesis:
    """Program with evidence."""
    program: Program
    support: int = 0
    total: int = 0

    def score(self) -> float:
        """MDL / support."""
        if self.support == 0:
            return float('inf')
        return self.program.mdl() / self.support

    def velocity(self) -> float:
        """Generalization velocity."""
        if self.total == 0:
            return 0.0
        return self.support / (self.program.mdl() * self.total)

# ============================================================================
# COMPOSITIONAL SOLVER
# ============================================================================

class CompositionalSolver:
    """Solver with multi-step composition capability."""

    def __init__(self, model: str = None, verbose: bool = False, max_steps: int = 3):
        self.model = model
        self.verbose = verbose
        self.max_steps = max_steps
        self.use_llm = HAS_OPENAI and model is not None

        if self.use_llm:
            self.client = OpenAI()

        # Build primitive list
        self.primitives = [
            'reflect_v', 'reflect_h',
            'rotate_90', 'rotate_180', 'rotate_270',
            'transpose',
            'gravity_down', 'gravity_up', 'gravity_left', 'gravity_right',
            'tile_2x2', 'tile_3x3',
            'zoom_2x', 'compress_2x',
            'extract_nonzero',
        ]

    def solve(self, task: Dict) -> Dict:
        """Solve with compositional search."""
        train = task['train']
        test_input = np.array(task['test'][0]['input'])
        test_output = task['test'][0].get('output')

        if self.verbose:
            print(f"\n  Training: {len(train)} examples")
            print(f"  Max composition depth: {self.max_steps}")

        # Try increasing composition depths
        all_hypotheses = []

        # 1-step (single primitives)
        if self.verbose:
            print(f"\n  Testing 1-step transformations...")
        hypotheses_1 = self._test_programs(
            self._generate_1_step_programs(),
            train
        )
        all_hypotheses.extend(hypotheses_1)

        # Check if we found a perfect match
        perfect = [h for h in hypotheses_1 if h.support == len(train)]
        if perfect:
            best = min(perfect, key=lambda h: h.score())
            if self.verbose:
                print(f"  ✓ Perfect 1-step match: {best.program}")
        else:
            # 2-step compositions
            if self.max_steps >= 2:
                if self.verbose:
                    print(f"  Testing 2-step compositions...")
                hypotheses_2 = self._test_programs(
                    self._generate_2_step_programs(hypotheses_1, train),
                    train
                )
                all_hypotheses.extend(hypotheses_2)

                # Check for perfect match
                perfect = [h for h in hypotheses_2 if h.support == len(train)]
                if perfect:
                    best = min(perfect, key=lambda h: h.score())
                    if self.verbose:
                        print(f"  ✓ Perfect 2-step match: {best.program}")
                else:
                    # 3-step compositions (only if enabled)
                    if self.max_steps >= 3:
                        if self.verbose:
                            print(f"  Testing 3-step compositions...")
                        hypotheses_3 = self._test_programs(
                            self._generate_3_step_programs(hypotheses_2, train),
                            train
                        )
                        all_hypotheses.extend(hypotheses_3)

        # Select best hypothesis
        valid = [h for h in all_hypotheses if h.support > 0]

        if valid:
            best = min(valid, key=lambda h: h.score())
            bits_spent = 0

            if self.verbose:
                print(f"\n  Best: {best.program}")
                print(f"    Support: {best.support}/{len(train)}")
                print(f"    MDL: {best.program.mdl():.2f}")
                print(f"    Score: {best.score():.2f}")
                print(f"    Velocity: {best.velocity():.3f}")
        else:
            # Try LLM if available
            if self.use_llm:
                if self.verbose:
                    print(f"\n  No composition worked, trying LLM...")
                best, bits_spent = self._llm_suggest(train)
            else:
                # Ultimate fallback
                best = Hypothesis(Program(['identity']), len(train), len(train))
                bits_spent = 0
                if self.verbose:
                    print(f"\n  No solution found, using identity")

        # Apply to test
        try:
            prediction = best.program.execute(test_input)
            if prediction is None:
                prediction = test_input
        except:
            prediction = test_input

        correct = None
        if test_output is not None:
            correct = np.array_equal(prediction, np.array(test_output))

        return {
            'program': str(best.program),
            'steps': len(best.program.steps),
            'mdl': best.program.mdl(),
            'score': best.score(),
            'support': best.support,
            'velocity': best.velocity(),
            'bits_spent': bits_spent,
            'prediction': prediction.tolist(),
            'correct': correct
        }

    def _generate_1_step_programs(self) -> List[Program]:
        """Generate all 1-step programs."""
        return [Program([p]) for p in self.primitives]

    def _generate_2_step_programs(self, step1_results: List[Hypothesis],
                                   train: List[Dict]) -> List[Program]:
        """Generate 2-step programs guided by partial matches."""
        programs = []

        # Strategy 1: Combine primitives that showed some promise
        promising = [h for h in step1_results if h.support > 0]

        if promising:
            # Try composing promising operations
            for h1 in promising[:5]:  # Top 5 by support
                for p2 in self.primitives[:10]:  # Most common operations
                    programs.append(Program(h1.program.steps + [p2]))

        # Strategy 2: Common 2-step patterns
        common_pairs = [
            ('rotate_90', 'reflect_v'),
            ('reflect_v', 'rotate_90'),
            ('rotate_180', 'reflect_h'),
            ('transpose', 'rotate_90'),
            ('extract_nonzero', 'tile_2x2'),
            ('compress_2x', 'tile_2x2'),
            ('gravity_down', 'reflect_v'),
            ('tile_2x2', 'rotate_90'),
        ]

        for p1, p2 in common_pairs:
            programs.append(Program([p1, p2]))

        return programs[:50]  # Limit search space

    def _generate_3_step_programs(self, step2_results: List[Hypothesis],
                                   train: List[Dict]) -> List[Program]:
        """Generate 3-step programs from promising 2-step results."""
        programs = []

        # Only extend the most promising 2-step programs
        promising = sorted([h for h in step2_results if h.support > 0],
                         key=lambda h: h.support, reverse=True)[:3]

        for h2 in promising:
            for p3 in self.primitives[:8]:  # Most common
                programs.append(Program(h2.program.steps + [p3]))

        return programs[:30]  # Strict limit on 3-step search

    def _test_programs(self, programs: List[Program],
                      train: List[Dict]) -> List[Hypothesis]:
        """Test programs on training examples."""
        hypotheses = []

        for prog in programs:
            support = 0
            for ex in train:
                try:
                    in_grid = np.array(ex['input'])
                    out_grid = np.array(ex['output'])
                    prediction = prog.execute(in_grid)

                    if prediction is not None and np.array_equal(prediction, out_grid):
                        support += 1
                except:
                    continue

            hyp = Hypothesis(prog, support, len(train))
            hypotheses.append(hyp)

            # Early exit if perfect match
            if support == len(train):
                return hypotheses

        return hypotheses

    def _llm_suggest(self, train: List[Dict]) -> Tuple[Hypothesis, int]:
        """Use LLM to suggest composition."""
        prompt = self._build_llm_prompt(train)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content":
                     "You are an ARC-AGI solver. Suggest a sequence of 1-3 "
                     "transformations from: " + ", ".join(self.primitives) + ". "
                     "Format: step1 → step2 → step3"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )

            # Parse response
            text = response.choices[0].message.content.strip()
            steps = [s.strip() for s in text.split('→')]
            steps = [s for s in steps if s in self.primitives]

            if steps:
                prog = Program(steps)
                # Test it
                support = 0
                for ex in train:
                    try:
                        prediction = prog.execute(np.array(ex['input']))
                        if prediction is not None and np.array_equal(
                            prediction, np.array(ex['output'])):
                            support += 1
                    except:
                        pass

                if self.verbose:
                    print(f"  LLM suggested: {prog} (support={support}/{len(train)})")

                return Hypothesis(prog, support, len(train)), 5

        except Exception as e:
            if self.verbose:
                print(f"  LLM error: {e}")

        # Fallback
        return Hypothesis(Program(['identity']), len(train), len(train)), 5

    def _build_llm_prompt(self, train: List[Dict]) -> str:
        """Build prompt for LLM."""
        prompt = f"Analyze {len(train)} examples and suggest transformation sequence:\n\n"

        for i, ex in enumerate(train[:2], 1):
            in_arr = np.array(ex['input'])
            out_arr = np.array(ex['output'])
            prompt += f"Example {i}:\n"
            prompt += f"  Input: {in_arr.shape}\n"
            prompt += f"  Output: {out_arr.shape}\n"

            # Key observations
            if in_arr.shape != out_arr.shape:
                prompt += f"  Shape change: {in_arr.shape} → {out_arr.shape}\n"

            prompt += "\n"

        prompt += "Suggest sequence of transformations (1-3 steps)."
        return prompt

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compositional ARC Solver')
    parser.add_argument('directory', help='Directory with JSON tasks')
    parser.add_argument('model', nargs='?', default=None,
                       help='OpenAI model (optional)')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-n', '--max-tasks', type=int)
    parser.add_argument('--max-steps', type=int, default=3,
                       help='Max composition depth (1-3, default: 3)')
    args = parser.parse_args()

    print("="*60)
    print("COMPOSITIONAL ARC SOLVER")
    print("="*60)
    print(f"Max steps: {args.max_steps}")
    print(f"Model: {args.model or 'None (compositions only)'}")
    print(f"Directory: {args.directory}\n")

    solver = CompositionalSolver(
        model=args.model,
        verbose=args.verbose,
        max_steps=args.max_steps
    )

    # Load tasks
    task_files = sorted(Path(args.directory).glob('*.json'))
    if args.max_tasks:
        task_files = task_files[:args.max_tasks]

    print(f"Tasks: {len(task_files)}\n")

    # Solve
    results = []
    correct_count = 0
    total_with_output = 0
    total_bits = 0
    composition_counts = {1: 0, 2: 0, 3: 0}

    for i, task_file in enumerate(task_files, 1):
        task_id = task_file.stem

        if not args.verbose:
            print(f"[{i}/{len(task_files)}] {task_id}", end='')
        else:
            print(f"\n[{i}/{len(task_files)}] {task_id}")

        with open(task_file) as f:
            task = json.load(f)

        result = solver.solve(task)
        result['task_id'] = task_id
        results.append(result)

        total_bits += result['bits_spent']
        composition_counts[min(result['steps'], 3)] += 1

        if result['correct'] is not None:
            total_with_output += 1
            if result['correct']:
                correct_count += 1
                status = '✓'
            else:
                status = '✗'
        else:
            status = '?'

        if not args.verbose:
            print(f"  {status} {result['program'][:40]}... "
                  f"(steps: {result['steps']}, MDL: {result['mdl']:.1f})")

    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    if total_with_output > 0:
        accuracy = 100 * correct_count / total_with_output
        print(f"Accuracy: {correct_count}/{total_with_output} ({accuracy:.1f}%)")

    print(f"\nComposition breakdown:")
    for steps in [1, 2, 3]:
        count = composition_counts.get(steps, 0)
        pct = 100 * count / len(task_files) if task_files else 0
        print(f"  {steps}-step: {count} ({pct:.1f}%)")

    avg_bits = total_bits / len(task_files) if task_files else 0
    print(f"\nAverage bits: {avg_bits:.1f}")
    print(f"Total tasks: {len(task_files)}")

    # Save
    output_file = Path('compositional_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'model': args.model,
                'max_steps': args.max_steps,
                'total_tasks': len(task_files),
                'correct': correct_count,
                'total_with_output': total_with_output,
                'accuracy': correct_count / total_with_output if total_with_output > 0 else 0,
                'avg_bits': avg_bits,
                'composition_counts': composition_counts
            },
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main()