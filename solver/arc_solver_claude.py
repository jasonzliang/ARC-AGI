#!/usr/bin/env python3
"""
Living Map ARC-AGI Solver

Principles:
1. DSL as bias shaper - Language makes true rules short, spurious long
2. Interaction as bits - Every LLM call costs information budget
3. Perception as typed objects - Raw grids → structured object graphs
4. Glue over content - Role binding, attention shift, propose-test
5. Generalization velocity - Speed to rule-lock under strict budget

Architecture:
- Primitives library (geometry + control)
- LLM as compression oracle (proposes short programs)
- MDL scoring (cost / support with shortcut inflation penalty)
- Interaction budget tracking
- Propose-test refinement loop
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
import argparse

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

# ============================================================================
# TYPED PERCEPTION - Object Graphs
# ============================================================================

@dataclass
class ObjectGraph:
    """Perception as typed objects, not raw pixels."""
    objects: List[Dict]  # color, pixels, bbox, shape
    relations: Dict[str, List]  # adjacency, enclosure, alignment, symmetry

    @staticmethod
    def from_grid(grid: np.ndarray) -> 'ObjectGraph':
        """Extract typed object graph from raw grid."""
        objects = []
        seen = set()
        h, w = grid.shape

        # Extract connected components (objects)
        for r in range(h):
            for c in range(w):
                if (r, c) in seen or grid[r, c] == 0:
                    continue

                # BFS for connected component
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

                if pixels:
                    rows, cols = zip(*pixels)
                    objects.append({
                        'color': int(color),
                        'pixels': pixels,
                        'bbox': (min(rows), min(cols), max(rows), max(cols)),
                        'size': len(pixels),
                        'shape': ObjectGraph._classify_shape(pixels)
                    })

        # Extract relations
        relations = {
            'symmetric': ObjectGraph._find_symmetries(grid),
            'aligned': ObjectGraph._find_alignments(objects),
            'enclosed': ObjectGraph._find_enclosures(objects),
        }

        return ObjectGraph(objects, relations)

    @staticmethod
    def _classify_shape(pixels: List[Tuple[int, int]]) -> str:
        """Classify object shape."""
        if len(pixels) == 1:
            return 'point'
        rows, cols = zip(*pixels)
        h, w = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
        density = len(pixels) / (h * w)

        if density > 0.9:
            if h == w:
                return 'square'
            return 'rectangle'
        elif density < 0.3:
            return 'sparse'
        return 'irregular'

    @staticmethod
    def _find_symmetries(grid: np.ndarray) -> List[str]:
        """Detect symmetries."""
        symmetries = []
        if np.array_equal(grid, np.flipud(grid)):
            symmetries.append('vertical')
        if np.array_equal(grid, np.fliplr(grid)):
            symmetries.append('horizontal')
        if np.array_equal(grid, grid.T):
            symmetries.append('diagonal')
        return symmetries

    @staticmethod
    def _find_alignments(objects: List[Dict]) -> List[Tuple[int, int]]:
        """Find aligned object pairs."""
        aligned = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                r1, c1, _, _ = obj1['bbox']
                r2, c2, _, _ = obj2['bbox']
                if r1 == r2 or c1 == c2:  # Same row or column
                    aligned.append((i, j))
        return aligned

    @staticmethod
    def _find_enclosures(objects: List[Dict]) -> List[Tuple[int, int]]:
        """Find enclosed object pairs."""
        enclosed = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i == j:
                    continue
                r1, c1, r2, c2 = obj1['bbox']
                r3, c3, r4, c4 = obj2['bbox']
                if r3 >= r1 and c3 >= c1 and r4 <= r2 and c4 <= c2:
                    enclosed.append((i, j))
        return enclosed

# ============================================================================
# DSL - Language as Bias Shaper
# ============================================================================

class DSL:
    """Domain-specific language with costs calibrated by MDL principle."""

    OPERATORS = {
        # Core geometric operators (low cost - fundamental)
        'identity': {'cost': 0.5, 'category': 'geometric'},
        'reflect_v': {'cost': 1.0, 'category': 'geometric'},
        'reflect_h': {'cost': 1.0, 'category': 'geometric'},
        'rotate_90': {'cost': 1.0, 'category': 'geometric'},
        'rotate_180': {'cost': 1.0, 'category': 'geometric'},
        'rotate_270': {'cost': 1.0, 'category': 'geometric'},
        'transpose': {'cost': 1.0, 'category': 'geometric'},

        # Grouping operators
        'tile_nxn': {'cost': 2.0, 'category': 'grouping'},
        'zoom_2x': {'cost': 1.5, 'category': 'grouping'},
        'compress_2x': {'cost': 1.5, 'category': 'grouping'},

        # Symmetry operators
        'mirror_extend': {'cost': 2.0, 'category': 'symmetry'},
        'complete_symmetry': {'cost': 2.5, 'category': 'symmetry'},

        # Counting/parity operators
        'gravity_down': {'cost': 1.5, 'category': 'counting'},
        'gravity_up': {'cost': 1.5, 'category': 'counting'},
        'gravity_left': {'cost': 1.5, 'category': 'counting'},
        'gravity_right': {'cost': 1.5, 'category': 'counting'},
        'count_by_color': {'cost': 2.0, 'category': 'counting'},

        # Alignment operators
        'align_horizontal': {'cost': 2.0, 'category': 'alignment'},
        'align_vertical': {'cost': 2.0, 'category': 'alignment'},
        'distribute_evenly': {'cost': 2.5, 'category': 'alignment'},

        # GLUE operators (control flow - critical for composition)
        'role_bind': {'cost': 1.0, 'category': 'glue'},  # Bind identity across transforms
        'attention_shift': {'cost': 1.0, 'category': 'glue'},  # Move focus
        'guarded_compose': {'cost': 1.5, 'category': 'glue'},  # Conditional composition
        'propose_test': {'cost': 1.0, 'category': 'glue'},  # Hypothesis refinement

        # Expensive spurious operators (high cost - discouraged)
        'color_histogram_match': {'cost': 5.0, 'category': 'spurious'},
        'template_match': {'cost': 5.0, 'category': 'spurious'},
        'pixel_by_pixel': {'cost': 10.0, 'category': 'spurious'},
    }

    @staticmethod
    def get_cost(operator: str, params: Dict = None) -> float:
        """Get MDL cost of operator + params."""
        base_cost = DSL.OPERATORS.get(operator, {'cost': 3.0})['cost']
        param_cost = 0.5 * len(params) if params else 0
        return base_cost + param_cost

    @staticmethod
    def is_glue(operator: str) -> bool:
        """Check if operator is glue/control."""
        return DSL.OPERATORS.get(operator, {}).get('category') == 'glue'

    @staticmethod
    def is_spurious(operator: str) -> bool:
        """Check if operator is spurious (should be expensive)."""
        return DSL.OPERATORS.get(operator, {}).get('category') == 'spurious'

# ============================================================================
# PROGRAM - Composable Transformation
# ============================================================================

@dataclass
class Program:
    """A composable transformation program."""
    operators: List[str]
    params: List[Dict]
    glue: List[str]  # Binding/control between operators

    def mdl(self) -> float:
        """Minimum Description Length."""
        op_cost = sum(DSL.get_cost(op, p) for op, p in zip(self.operators, self.params))
        glue_cost = sum(DSL.get_cost(g) for g in self.glue)

        # Subadditivity bonus: compositions should compress
        if len(self.operators) > 1:
            # Discount for well-composed programs
            composition_discount = 0.5 * (len(self.operators) - 1)
            return max(0.5, op_cost + glue_cost - composition_discount)

        return op_cost + glue_cost

    def __str__(self) -> str:
        """Human-readable program representation."""
        if not self.operators:
            return "identity"

        parts = []
        for i, (op, param) in enumerate(zip(self.operators, self.params)):
            if param:
                param_str = ','.join(f'{k}={v}' for k, v in param.items())
                parts.append(f"{op}({param_str})")
            else:
                parts.append(op)

            # Add glue between operators
            if i < len(self.glue):
                parts.append(f"→[{self.glue[i]}]→")

        return ' '.join(parts)

# ============================================================================
# HYPOTHESIS - Program + Evidence
# ============================================================================

@dataclass
class Hypothesis:
    """Program with support evidence and shortcut inflation measure."""
    program: Program
    support: int = 0  # Number of examples explained
    total_examples: int = 0
    shortcut_inflation: float = 0.0  # Penalty for being shortcut-like

    def score(self) -> float:
        """MDL-based score: lower is better."""
        if self.support == 0:
            return float('inf')

        # Base: MDL per explained example
        base_score = self.program.mdl() / self.support

        # Penalty for shortcut characteristics
        penalty = self.shortcut_inflation * 2.0

        return base_score + penalty

    def generalization_velocity(self) -> float:
        """How quickly this hypothesis locks onto pattern."""
        if self.total_examples == 0:
            return 0.0
        return self.support / (self.program.mdl() * self.total_examples)

# ============================================================================
# LLM ORACLE - Compression via Language Model
# ============================================================================

class LLMOracle:
    """LLM as compression oracle - proposes short programs given examples."""

    def __init__(self, client: OpenAI, model: str, verbose: bool = False):
        self.client = client
        self.model = model
        self.verbose = verbose
        self.bits_spent = 0
        self.interaction_count = 0

    def propose_programs(self,
                        object_graphs: List[Tuple[ObjectGraph, ObjectGraph]],
                        budget: int = 3) -> List[Program]:
        """Use LLM to propose programs given input/output object graphs.

        Key: We don't ask LLM to solve, we ask it to compress using our DSL.
        """
        if self.interaction_count >= budget:
            if self.verbose:
                print(f"  Budget exhausted ({budget} bits)")
            return []

        # Build structured prompt with typed objects
        prompt = self._build_compression_prompt(object_graphs)

        if self.verbose:
            print(f"  LLM interaction {self.interaction_count + 1}/{budget}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )

            self.bits_spent += 3  # Cost of LLM interaction
            self.interaction_count += 1

            # Parse programs from response
            programs = self._parse_programs(response.choices[0].message.content)

            if self.verbose:
                print(f"  Proposed {len(programs)} programs")
                for p in programs[:3]:
                    print(f"    - {p} (MDL: {p.mdl():.1f})")

            return programs

        except Exception as e:
            if self.verbose:
                print(f"  LLM error: {e}")
            return []

    def _system_prompt(self) -> str:
        """System prompt emphasizing compression and composition."""
        operators = [k for k, v in DSL.OPERATORS.items()
                    if v['category'] not in ['spurious']]

        return f"""You are a compression oracle for ARC-AGI puzzles.

YOUR TASK: Propose SHORT programs using our DSL that transform input → output.

DSL OPERATORS (with MDL costs):
{self._format_operators()}

PRINCIPLES:
1. SHORTER IS BETTER - Minimize MDL cost
2. COMPOSE - Use glue operators (role_bind, attention_shift, guarded_compose)
3. AVOID SHORTCUTS - No color_histogram_match or template_match
4. THINK OBJECTS - Work with typed objects (shapes, positions, relations)
5. SEEK SYMMETRY - Prefer geometric and symmetry operators

OUTPUT FORMAT:
PROGRAM 1: operator1 → [glue] → operator2
MDL: X.X
PROGRAM 2: operator2(param=value)
MDL: Y.Y

Propose 3-5 programs from simplest to slightly more complex."""

    def _format_operators(self) -> str:
        """Format operator list with costs."""
        by_category = {}
        for name, info in DSL.OPERATORS.items():
            cat = info['category']
            if cat == 'spurious':
                continue
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(f"  {name} (cost: {info['cost']})")

        lines = []
        for cat, ops in sorted(by_category.items()):
            lines.append(f"\n{cat.upper()}:")
            lines.extend(ops)
        return '\n'.join(lines)

    def _build_compression_prompt(self,
                                  graphs: List[Tuple[ObjectGraph, ObjectGraph]]) -> str:
        """Build prompt from object graphs (not raw pixels)."""
        prompt = "Transform these structured representations:\n\n"

        for i, (in_graph, out_graph) in enumerate(graphs[:2], 1):
            prompt += f"Example {i}:\n"
            prompt += f"  Input: {len(in_graph.objects)} objects, "
            prompt += f"symmetries: {in_graph.relations['symmetric']}\n"
            prompt += f"  Output: {len(out_graph.objects)} objects, "
            prompt += f"symmetries: {out_graph.relations['symmetric']}\n"

            # Key structural changes
            if len(in_graph.objects) != len(out_graph.objects):
                ratio = len(out_graph.objects) / max(1, len(in_graph.objects))
                prompt += f"  Change: {ratio}x objects\n"

            prompt += "\n"

        prompt += "Propose short programs (low MDL) using our DSL operators."
        return prompt

    def _parse_programs(self, response: str) -> List[Program]:
        """Parse LLM response into Program objects."""
        programs = []
        lines = response.split('\n')

        current_ops = []
        current_params = []
        current_glue = []

        for line in lines:
            line = line.strip()

            if line.upper().startswith('PROGRAM'):
                # Save previous program
                if current_ops:
                    programs.append(Program(current_ops, current_params, current_glue))
                    current_ops, current_params, current_glue = [], [], []

                # Parse new program
                if ':' in line:
                    prog_str = line.split(':', 1)[1].strip()
                    ops, params, glue = self._parse_program_string(prog_str)
                    current_ops, current_params, current_glue = ops, params, glue

        # Add last program
        if current_ops:
            programs.append(Program(current_ops, current_params, current_glue))

        return programs

    def _parse_program_string(self, prog_str: str) -> Tuple[List[str], List[Dict], List[str]]:
        """Parse program string into operators, params, glue."""
        operators = []
        params = []
        glue = []

        # Split by arrows to find glue
        parts = prog_str.split('→')

        for i, part in enumerate(parts):
            part = part.strip()

            # Check if this is glue (in brackets)
            if part.startswith('[') and part.endswith(']'):
                glue.append(part[1:-1].strip())
                continue

            # Parse operator
            if '(' in part:
                op_name = part[:part.index('(')].strip()
                param_str = part[part.index('(')+1:part.rindex(')')].strip()
                param_dict = {}
                for p in param_str.split(','):
                    if '=' in p:
                        k, v = p.split('=', 1)
                        param_dict[k.strip()] = v.strip()
                operators.append(op_name)
                params.append(param_dict)
            else:
                # Simple operator
                if part and part in DSL.OPERATORS:
                    operators.append(part)
                    params.append({})

        return operators, params, glue

# ============================================================================
# EXECUTOR - Execute Programs on Grids
# ============================================================================

class Executor:
    """Execute programs on numpy grids."""

    @staticmethod
    def execute(program: Program, grid: np.ndarray) -> Optional[np.ndarray]:
        """Execute program on grid."""
        result = grid.copy()

        for op, param in zip(program.operators, program.params):
            result = Executor._execute_operator(op, result, param)
            if result is None:
                return None

        return result

    @staticmethod
    def _execute_operator(op: str, grid: np.ndarray, params: Dict) -> Optional[np.ndarray]:
        """Execute single operator."""
        try:
            if op == 'identity':
                return grid
            elif op == 'reflect_v':
                return np.flipud(grid)
            elif op == 'reflect_h':
                return np.fliplr(grid)
            elif op == 'rotate_90':
                return np.rot90(grid, k=1)
            elif op == 'rotate_180':
                return np.rot90(grid, k=2)
            elif op == 'rotate_270':
                return np.rot90(grid, k=3)
            elif op == 'transpose':
                return grid.T
            elif op == 'tile_nxn':
                n = int(params.get('n', 2))
                return np.tile(grid, (n, n))
            elif op == 'zoom_2x':
                return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)
            elif op == 'compress_2x':
                h, w = grid.shape
                if h % 2 == 0 and w % 2 == 0:
                    return grid[::2, ::2]
                return grid
            elif op.startswith('gravity_'):
                return Executor._gravity(grid, op.split('_')[1])
            else:
                return grid  # Unknown operator, return unchanged
        except Exception as e:
            return None

    @staticmethod
    def _gravity(grid: np.ndarray, direction: str) -> np.ndarray:
        """Apply gravity."""
        result = np.zeros_like(grid)
        h, w = grid.shape

        if direction == 'down':
            for col in range(w):
                non_zero = grid[:, col][grid[:, col] != 0]
                if len(non_zero) > 0:
                    result[h-len(non_zero):, col] = non_zero
        elif direction == 'up':
            for col in range(w):
                non_zero = grid[:, col][grid[:, col] != 0]
                if len(non_zero) > 0:
                    result[:len(non_zero), col] = non_zero
        elif direction == 'left':
            for row in range(h):
                non_zero = grid[row, :][grid[row, :] != 0]
                if len(non_zero) > 0:
                    result[row, :len(non_zero)] = non_zero
        elif direction == 'right':
            for row in range(h):
                non_zero = grid[row, :][grid[row, :] != 0]
                if len(non_zero) > 0:
                    result[row, w-len(non_zero):] = non_zero

        return result

# ============================================================================
# LIVING MAP SOLVER
# ============================================================================

class LivingMapSolver:
    """ARC solver following Living Map principles."""

    def __init__(self, model: str, verbose: bool = False):
        if not HAS_OPENAI:
            raise ImportError("OpenAI package required. Install with: pip install openai")
        self.client = OpenAI()
        self.oracle = LLMOracle(self.client, model, verbose)
        self.verbose = verbose

    def solve(self, task: Dict) -> Dict:
        """Solve task with interaction budget."""
        train = task['train']
        test_input = np.array(task['test'][0]['input'])
        test_output = task['test'][0].get('output')

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Task: {len(train)} training examples")
            print(f"{'='*60}")

        # Phase 1: Perception - Extract typed object graphs
        train_graphs = []
        for ex in train:
            in_graph = ObjectGraph.from_grid(np.array(ex['input']))
            out_graph = ObjectGraph.from_grid(np.array(ex['output']))
            train_graphs.append((in_graph, out_graph))

        if self.verbose:
            print(f"\n📊 PERCEPTION")
            print(f"  Extracted object graphs from {len(train)} examples")

        # Phase 2: Propose - LLM proposes programs (interaction budget: 3 bits)
        self.oracle.bits_spent = 0
        self.oracle.interaction_count = 0

        if self.verbose:
            print(f"\n💡 PROPOSE (Budget: 3 LLM calls)")

        proposed = self.oracle.propose_programs(train_graphs, budget=3)

        # Phase 3: Test - Validate on training examples
        if self.verbose:
            print(f"\n🧪 TEST")

        hypotheses = []
        for prog in proposed:
            support = 0
            for ex in train:
                in_grid = np.array(ex['input'])
                out_grid = np.array(ex['output'])

                prediction = Executor.execute(prog, in_grid)
                if prediction is not None and np.array_equal(prediction, out_grid):
                    support += 1

            # Compute shortcut inflation (penalize if uses spurious operators)
            inflation = sum(2.0 for op in prog.operators if DSL.is_spurious(op))

            hyp = Hypothesis(prog, support, len(train), inflation)
            hypotheses.append(hyp)

            if self.verbose and support > 0:
                print(f"  ✓ {prog} - support: {support}/{len(train)}, "
                      f"score: {hyp.score():.2f}, velocity: {hyp.generalization_velocity():.3f}")

        # Phase 4: Select - Choose by MDL score
        if self.verbose:
            print(f"\n🎯 SELECT")

        valid = [h for h in hypotheses if h.support > 0]

        if valid:
            # Sort by score (lower is better)
            valid.sort(key=lambda h: h.score())
            best = valid[0]

            if self.verbose:
                print(f"  Best: {best.program}")
                print(f"  MDL: {best.program.mdl():.2f}, Score: {best.score():.2f}")
                print(f"  Generalization velocity: {best.generalization_velocity():.3f}")
        else:
            # Fallback to identity
            best = Hypothesis(Program(['identity'], [{}], []), len(train), len(train))
            if self.verbose:
                print(f"  ⚠ No valid program found, using identity")

        # Phase 5: Apply to test
        if self.verbose:
            print(f"\n🚀 APPLY")

        prediction = Executor.execute(best.program, test_input)
        if prediction is None:
            prediction = test_input  # Fallback

        correct = None
        if test_output is not None:
            correct = np.array_equal(prediction, np.array(test_output))
            if self.verbose:
                print(f"  Result: {'✓ CORRECT' if correct else '✗ WRONG'}")

        return {
            'program': str(best.program),
            'mdl': best.program.mdl(),
            'score': best.score(),
            'support': best.support,
            'total_examples': best.total_examples,
            'generalization_velocity': best.generalization_velocity(),
            'bits_spent': self.oracle.bits_spent,
            'interactions': self.oracle.interaction_count,
            'prediction': prediction.tolist(),
            'correct': correct
        }

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Living Map ARC-AGI Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Principles:
  1. DSL as bias shaper - True rules are short, spurious are expensive
  2. Interaction as bits - LLM calls have information cost
  3. Perception as objects - Work with typed object graphs
  4. Glue over content - Composition via role binding and control
  5. Generalization velocity - Speed to rule-lock under budget
        """
    )
    parser.add_argument('directory', help='Directory with JSON task files')
    parser.add_argument('model', nargs='?', default='gpt-4o',
                       help='OpenAI model (default: gpt-4o)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('-n', '--max-tasks', type=int,
                       help='Maximum tasks to solve')
    args = parser.parse_args()

    print("="*60)
    print("LIVING MAP ARC-AGI SOLVER")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Directory: {args.directory}")
    print()

    solver = LivingMapSolver(model=args.model, verbose=args.verbose)

    # Load tasks
    task_files = sorted(Path(args.directory).glob('*.json'))
    if args.max_tasks:
        task_files = task_files[:args.max_tasks]

    print(f"Tasks to solve: {len(task_files)}\n")

    # Solve tasks
    results = []
    correct_count = 0
    total_with_output = 0
    total_bits = 0
    total_velocity = 0

    for i, task_file in enumerate(task_files, 1):
        task_id = task_file.stem

        if not args.verbose:
            print(f"[{i}/{len(task_files)}] {task_id}", end='')

        with open(task_file) as f:
            task = json.load(f)
            task['id'] = task_id

        result = solver.solve(task)
        result['task_id'] = task_id
        results.append(result)

        total_bits += result['bits_spent']
        total_velocity += result['generalization_velocity']

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
            print(f"  {status} {result['program'][:50]}... "
                  f"(MDL: {result['mdl']:.1f}, bits: {result['bits_spent']})")

    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    if total_with_output > 0:
        accuracy = 100 * correct_count / total_with_output
        print(f"Accuracy: {correct_count}/{total_with_output} ({accuracy:.1f}%)")

    avg_bits = total_bits / len(task_files) if task_files else 0
    avg_velocity = total_velocity / len(task_files) if task_files else 0

    print(f"Average bits spent: {avg_bits:.1f}")
    print(f"Average generalization velocity: {avg_velocity:.3f}")
    print(f"Total tasks: {len(task_files)}")

    # Save results
    output_file = Path('living_map_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'model': args.model,
                'total_tasks': len(task_files),
                'correct': correct_count,
                'total_with_output': total_with_output,
                'accuracy': correct_count / total_with_output if total_with_output > 0 else 0,
                'avg_bits': avg_bits,
                'avg_generalization_velocity': avg_velocity
            },
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main()