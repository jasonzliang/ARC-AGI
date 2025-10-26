#!/usr/bin/env python3
"""
ARC-AGI Solver implementing Living Map principles:
- DSL with explicit operators and MDL costs
- Typed perception (object extraction with roles)
- Propose-test loop with hypothesis ranking by compression
- Glue operators for composition
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
import argparse
from openai import OpenAI


# ============================================================================
# TYPED PERCEPTION: Extract structured objects from grids
# ============================================================================

@dataclass
class VisualObject:
    """Typed object with role and properties."""
    pixels: Set[Tuple[int, int]]
    color: int
    bbox: Tuple[int, int, int, int]  # min_r, min_c, max_r, max_c
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        return len(self.pixels)
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.bbox[2] - self.bbox[0] + 1, 
                self.bbox[3] - self.bbox[1] + 1)


class ObjectGraph:
    """Structured representation with objects and relations."""
    
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.objects = self._extract_objects()
        self.relations = self._compute_relations()
    
    def _extract_objects(self) -> List[VisualObject]:
        """Extract connected components by color."""
        objects = []
        visited = np.zeros_like(self.grid, dtype=bool)
        
        for r in range(self.grid.shape[0]):
            for c in range(self.grid.shape[1]):
                if not visited[r, c] and self.grid[r, c] != 0:
                    pixels = self._flood_fill(r, c, visited)
                    if pixels:
                        color = self.grid[r, c]
                        bbox = self._compute_bbox(pixels)
                        props = self._compute_properties(pixels, color)
                        objects.append(VisualObject(pixels, color, bbox, props))
        
        return objects
    
    def _flood_fill(self, r: int, c: int, visited: np.ndarray) -> Set[Tuple[int, int]]:
        """BFS flood fill for connected component."""
        color = self.grid[r, c]
        pixels = set()
        queue = [(r, c)]
        
        while queue:
            cr, cc = queue.pop(0)
            if (cr < 0 or cr >= self.grid.shape[0] or 
                cc < 0 or cc >= self.grid.shape[1] or
                visited[cr, cc] or self.grid[cr, cc] != color):
                continue
            
            visited[cr, cc] = True
            pixels.add((cr, cc))
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                queue.append((cr + dr, cc + dc))
        
        return pixels
    
    def _compute_bbox(self, pixels: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """Compute bounding box."""
        rows, cols = zip(*pixels)
        return (min(rows), min(cols), max(rows), max(cols))
    
    def _compute_properties(self, pixels: Set[Tuple[int, int]], color: int) -> Dict:
        """Compute shape properties."""
        rows, cols = zip(*pixels)
        bbox = self._compute_bbox(pixels)
        height = bbox[2] - bbox[0] + 1
        width = bbox[3] - bbox[1] + 1
        
        return {
            'density': len(pixels) / (height * width) if height * width > 0 else 0,
            'is_rectangular': len(pixels) == height * width,
            'has_vertical_symmetry': self._check_vertical_symmetry(pixels, bbox),
            'has_horizontal_symmetry': self._check_horizontal_symmetry(pixels, bbox),
        }
    
    def _check_vertical_symmetry(self, pixels: Set[Tuple[int, int]], bbox: Tuple) -> bool:
        """Check if object has vertical symmetry."""
        mid_c = (bbox[1] + bbox[3]) / 2
        for r, c in pixels:
            mirror_c = int(2 * mid_c - c)
            if (r, mirror_c) not in pixels:
                return False
        return True
    
    def _check_horizontal_symmetry(self, pixels: Set[Tuple[int, int]], bbox: Tuple) -> bool:
        """Check if object has horizontal symmetry."""
        mid_r = (bbox[0] + bbox[2]) / 2
        for r, c in pixels:
            mirror_r = int(2 * mid_r - r)
            if (mirror_r, c) not in pixels:
                return False
        return True
    
    def _compute_relations(self) -> Dict[str, List[Tuple[int, int]]]:
        """Compute spatial relations between objects."""
        relations = defaultdict(list)
        
        for i, obj1 in enumerate(self.objects):
            for j, obj2 in enumerate(self.objects):
                if i >= j:
                    continue
                
                if self._is_adjacent(obj1, obj2):
                    relations['adjacent'].append((i, j))
                if self._is_aligned_vertical(obj1, obj2):
                    relations['aligned_vertical'].append((i, j))
                if self._is_aligned_horizontal(obj1, obj2):
                    relations['aligned_horizontal'].append((i, j))
        
        return dict(relations)
    
    def _is_adjacent(self, obj1: VisualObject, obj2: VisualObject) -> bool:
        """Check if objects are adjacent."""
        for r1, c1 in obj1.pixels:
            for r2, c2 in obj2.pixels:
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    return True
        return False
    
    def _is_aligned_vertical(self, obj1: VisualObject, obj2: VisualObject) -> bool:
        """Check if objects share column range."""
        return not (obj1.bbox[3] < obj2.bbox[1] or obj2.bbox[3] < obj1.bbox[1])
    
    def _is_aligned_horizontal(self, obj1: VisualObject, obj2: VisualObject) -> bool:
        """Check if objects share row range."""
        return not (obj1.bbox[2] < obj2.bbox[0] or obj2.bbox[2] < obj1.bbox[0])


# ============================================================================
# DSL: Operators with MDL costs
# ============================================================================

class Operator(ABC):
    """Base operator with description length cost."""
    
    @abstractmethod
    def mdl_cost(self) -> float:
        """Description length in bits."""
        pass
    
    @abstractmethod
    def describe(self) -> str:
        """Human-readable description for LLM."""
        pass


@dataclass
class PrimitiveOp(Operator):
    """Primitive operators: the vocabulary of transformations."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    # MDL costs: primitives are cheap, parameters add cost
    BASE_COSTS = {
        'identity': 0.5,
        'color_map': 1.0,
        'reflect_vertical': 1.0,
        'reflect_horizontal': 1.0,
        'rotate_90': 1.0,
        'rotate_180': 1.0,
        'rotate_270': 1.0,
        'translate': 1.5,
        'scale': 2.0,
        'gravity_down': 1.5,
        'gravity_up': 1.5,
        'gravity_left': 1.5,
        'gravity_right': 1.5,
        'align_objects': 2.0,
        'tile_pattern': 2.5,
        'extract_pattern': 2.0,
        'clear_background': 1.0,
    }
    
    def mdl_cost(self) -> float:
        base = self.BASE_COSTS.get(self.name, 3.0)
        param_cost = len(self.params) * 0.5
        return base + param_cost
    
    def describe(self) -> str:
        if self.params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.name}({param_str})"
        return self.name


@dataclass
class CompositeOp(Operator):
    """Composition of operators with glue."""
    operators: List[Operator]
    glue_type: str = "sequential"  # sequential, conditional, parallel
    
    def mdl_cost(self) -> float:
        """Should be subadditive when glue is canonical."""
        base_cost = sum(op.mdl_cost() for op in self.operators)
        
        # Glue costs
        glue_costs = {
            'sequential': 0.5,
            'conditional': 1.5,
            'parallel': 1.0,
        }
        glue_cost = glue_costs.get(self.glue_type, 2.0)
        
        # Composition discount: good glue makes composition cheaper
        discount = 0.2 if self.glue_type == 'sequential' else 0.1
        
        return (base_cost + glue_cost) * (1 - discount)
    
    def describe(self) -> str:
        op_descs = [op.describe() for op in self.operators]
        if self.glue_type == 'sequential':
            return " → ".join(op_descs)
        elif self.glue_type == 'conditional':
            return f"IF {op_descs[0]} THEN {op_descs[1]}"
        else:
            return f"({' || '.join(op_descs)})"


@dataclass
class Hypothesis:
    """A transformation hypothesis with MDL."""
    program: Operator
    support: float = 0.0  # How many examples it explains
    
    def mdl(self) -> float:
        """Total description length."""
        return self.program.mdl_cost()
    
    def score(self) -> float:
        """MDL-based score: prefer high support, low MDL."""
        if self.support == 0:
            return float('inf')
        return self.mdl() / self.support


# ============================================================================
# PROPOSE-TEST: LLM-guided hypothesis generation with MDL ranking
# ============================================================================

class LivingMapSolver:
    """Solver using typed perception, DSL, and MDL-based ranking."""
    
    def __init__(self, model: str = "gpt-4", bit_budget: int = 100, verbose: bool = False):
        try:
            self.client = OpenAI()
        except:
            self.client = None  # Allow testing without API key
        self.model = model
        self.bit_budget = bit_budget
        self.bits_spent = 0
        self.verbose = verbose
    
    def solve_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Main solving loop with propose-test."""
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SOLVING TASK")
            print(f"{'='*60}")
        
        # Phase 1: Typed perception
        if self.verbose:
            print(f"Phase 1: Typed Perception")
        train_graphs = [ObjectGraph(np.array(ex['input'])) for ex in task['train']]
        test_graph = ObjectGraph(np.array(task['test'][0]['input']))
        
        if self.verbose:
            print(f"  Training examples: {len(train_graphs)}")
            print(f"  Objects detected: {[len(g.objects) for g in train_graphs]}")
        
        # Phase 2: Analyze structure and propose operators
        if self.verbose:
            print(f"\nPhase 2: Structural Analysis")
        analysis = self._analyze_structure(train_graphs, task['train'])
        
        if self.verbose:
            print(f"  Shape changes: {analysis['shape_changes']}")
            print(f"  Color mappings: {len(analysis['color_mappings'])}")
        
        # Phase 3: Generate hypotheses using LLM + DSL
        if self.verbose:
            print(f"\nPhase 3: Generate Hypotheses")
        hypotheses = self._generate_hypotheses(analysis, task['train'])
        
        if self.verbose:
            print(f"  Generated {len(hypotheses)} hypotheses:")
            for i, h in enumerate(hypotheses):
                print(f"    {i+1}. {h.program.describe()}")
                print(f"       MDL: {h.mdl():.2f}, Support: {h.support:.1f}, Score: {h.score():.2f}")
        
        # Phase 4: Rank by MDL and select best
        if self.verbose:
            print(f"\nPhase 4: Select Best Hypothesis")
        best_hypothesis = min(hypotheses, key=lambda h: h.score()) if hypotheses else None
        
        if self.verbose and best_hypothesis:
            print(f"  Best: {best_hypothesis.program.describe()}")
            print(f"  Score: {best_hypothesis.score():.2f}")
        
        # Phase 5: Execute and return
        if self.verbose:
            print(f"\nPhase 5: Execute on Test Input")
        
        if best_hypothesis:
            prediction = self._execute_with_llm(
                best_hypothesis, 
                task['test'][0]['input'],
                analysis
            )
        else:
            if self.verbose:
                print(f"  ✗ No hypothesis available")
            prediction = None
        
        if self.verbose:
            print(f"\nTotal bits spent: {self.bits_spent}")
            print(f"{'='*60}\n")
        
        return {
            'prediction': prediction,
            'hypothesis': best_hypothesis.program.describe() if best_hypothesis else None,
            'mdl': best_hypothesis.mdl() if best_hypothesis else None,
            'bits_spent': self.bits_spent,
            'num_hypotheses': len(hypotheses),
        }
    
    def _analyze_structure(self, graphs: List[ObjectGraph], 
                          examples: List[Dict]) -> Dict[str, Any]:
        """Extract structural patterns from object graphs."""
        analysis = {
            'num_objects_in': [len(g.objects) for g in graphs],
            'colors_in': [set(obj.color for obj in g.objects) for g in graphs],
            'colors_out': [set(np.unique(ex['output'])) - {0} for ex in examples],
            'shape_changes': [],
            'color_mappings': [],
            'symmetries': [],
            'relations': [],
        }
        
        # Analyze input vs output
        for i, (graph, example) in enumerate(zip(graphs, examples)):
            in_shape = np.array(example['input']).shape
            out_shape = np.array(example['output']).shape
            
            analysis['shape_changes'].append({
                'in': in_shape,
                'out': out_shape,
                'same': in_shape == out_shape,
                'scaled': (out_shape[0] % in_shape[0] == 0 and 
                          out_shape[1] % in_shape[1] == 0) if in_shape[0] > 0 and in_shape[1] > 0 else False,
            })
            
            # Detect color mappings
            in_colors = set(obj.color for obj in graph.objects)
            out_colors = set(np.unique(example['output'])) - {0}
            
            if in_colors and out_colors and in_colors != out_colors:
                analysis['color_mappings'].append({
                    'in': in_colors,
                    'out': out_colors,
                })
            
            # Detect symmetries
            symmetries = [obj.properties.get('has_vertical_symmetry', False) 
                         for obj in graph.objects]
            if any(symmetries):
                analysis['symmetries'].append('vertical')
        
        return analysis
    
    def _generate_hypotheses(self, analysis: Dict, 
                            examples: List[Dict]) -> List[Hypothesis]:
        """Generate hypotheses using analysis + LLM."""
        
        # Build prompt emphasizing operators and MDL
        prompt = self._build_operator_prompt(analysis, examples)
        
        all_hypotheses = []
        
        # Try multiple temperature settings to get diverse hypotheses
        temperatures = [0.3, 0.7]  # Lower for precise, higher for creative
        
        for temp in temperatures:
            if self.verbose:
                print(f"  Calling LLM (temp={temp})...")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                    max_tokens=2000
                )
                
                self.bits_spent += 1  # Count LLM calls as interaction bits
                
                # Parse response into hypotheses
                hypotheses = self._parse_hypotheses(
                    response.choices[0].message.content,
                    len(examples)
                )
                
                if self.verbose:
                    print(f"    Parsed {len(hypotheses)} hypotheses")
                
                all_hypotheses.extend(hypotheses)
                
            except Exception as e:
                print(f"Hypothesis generation failed at temp={temp}: {e}")
                continue
        
        if self.verbose:
            print(f"  Computing actual support for hypotheses...")
        
        # CRITICAL FIX: Actually compute support by validating on training data
        for i, h in enumerate(all_hypotheses):
            old_support = h.support
            h.support = self._compute_support(h, examples, analysis)
            if self.verbose and old_support != h.support:
                print(f"    Hypothesis {i+1}: LLM said {old_support:.0f}, actual is {h.support:.0f}")
        
        # Deduplicate by operator description and keep best (highest support/MDL ratio)
        seen = {}
        for h in all_hypotheses:
            key = h.program.describe().lower()
            if key not in seen or h.score() < seen[key].score():
                seen[key] = h
        
        final_hypotheses = list(seen.values())
        
        if self.verbose:
            print(f"  After deduplication: {len(final_hypotheses)} unique hypotheses")
        
        # If we got no hypotheses, create a fallback
        if not final_hypotheses:
            if self.verbose:
                print(f"  ⚠ No hypotheses generated, using identity fallback")
            final_hypotheses.append(Hypothesis(PrimitiveOp('identity'), len(examples)))
        
        return final_hypotheses
    
    def _compute_support(self, hypothesis: Hypothesis, 
                        examples: List[Dict],
                        analysis: Dict) -> float:
        """Compute actual support by validating on training examples.
        
        Only validates if we have symbolic execution (free).
        Otherwise, trust LLM's estimate to stay within bit budget.
        """
        op_desc = hypothesis.program.describe().lower()
        
        # Check if we can use free symbolic execution
        has_symbolic = any(kw in op_desc for kw in ['tile', 'color_map', 'gravity_down'])
        
        if not has_symbolic:
            # No symbolic execution available - keep LLM's estimate
            # Computing support would cost too many bits
            return hypothesis.support
        
        # We have symbolic execution - validate for free!
        correct = 0
        for ex in examples:
            try:
                prediction = self._execute_with_llm(hypothesis, ex['input'], analysis)
                if prediction == ex['output']:
                    correct += 1
            except:
                continue
        
        return float(correct)
    
    def _get_system_prompt(self) -> str:
        """System prompt emphasizing Living Map principles."""
        return """You are an expert ARC-AGI solver. Your job: find the SHORTEST transformation rule.

AVAILABLE OPERATORS (lower MDL = simpler):
• color_map(A→B) - 1.5 bits - remap colors
• gravity_down/up/left/right - 1.5 bits - objects fall in direction
• reflect_vertical - 1.0 bits - flip top↔bottom
• reflect_horizontal - 1.0 bits - flip left↔right
• rotate_90/180/270 - 1.0 bits - rotate clockwise
• tile_3x3 - 2.5 bits - tile pattern 3×3 with alternating flips
• extract_and_copy - 2.0 bits - copy one object's structure
• move_objects_to_corner - 2.0 bits - arrange objects at corner
• clear_objects(color) - 1.5 bits - remove specific color
• identity - 0.5 bits - no change (rarely correct!)

COMPOSITION:
• A → B (do A then B): +0.5 bits
• A + B (do both): +0.5 bits

FEW-SHOT EXAMPLES:

Example Task: Small object determines output color
Pattern: There's a small colored shape (1) and large shape (8). The 8 becomes a different color based on 1's shape, then 1 disappears.
HYPOTHESIS 1: shape_based_color_map(1→determine_color) → clear_objects(1)
MDL: 3.5
Support: 5/5

Example Task: Objects move to bottom
Pattern: Colored objects are scattered, they all move to the bottom row and arrange horizontally.
HYPOTHESIS 1: gravity_down → arrange_horizontal
MDL: 3.0
Support: 4/4

Example Task: Simple tiling
Pattern: Small 2×2 grid becomes 6×6 by repeating 3 times with flips.
HYPOTHESIS 1: tile_3x3
MDL: 2.5
Support: 2/2

Example Task: Color remapping only
Pattern: All 8s become 2s, structure unchanged.
HYPOTHESIS 1: color_map(8→2)
MDL: 1.5
Support: 3/3

NOW YOUR TURN. Output format (EXACTLY):

HYPOTHESIS 1: [operator or composition]
MDL: [number]
Support: [X]/[total]

HYPOTHESIS 2: [operator or composition]
MDL: [number]
Support: [X]/[total]

[Continue for 4-6 hypotheses]

RULES:
1. Propose 4-6 diverse hypotheses
2. Start with simplest (lowest MDL)
3. Consider: color changes? objects moving? tiling? gravity? arrangement?
4. Look at ALL examples, not just first one
5. If multiple objects with different colors, consider their relationship"""
    
    def _build_operator_prompt(self, analysis: Dict, examples: List[Dict]) -> str:
        """Build prompt with structural analysis."""
        
        # More detailed structural analysis
        structure_summary = []
        
        # Shape analysis
        if all(sc['same'] for sc in analysis['shape_changes']):
            structure_summary.append("✓ Grid shape PRESERVED (same size in/out)")
        elif any(sc['scaled'] for sc in analysis['shape_changes']):
            in_shape = analysis['shape_changes'][0]['in']
            out_shape = analysis['shape_changes'][0]['out']
            ratio_r = out_shape[0] / in_shape[0] if in_shape[0] > 0 else 1
            ratio_c = out_shape[1] / in_shape[1] if in_shape[1] > 0 else 1
            structure_summary.append(f"✓ Grid SCALED/TILED: {in_shape} → {out_shape} (×{ratio_r:.0f} rows, ×{ratio_c:.0f} cols)")
        else:
            structure_summary.append("✓ Grid shape CHANGES")
        
        # Color analysis
        all_in_colors = set()
        all_out_colors = set()
        for i, ex in enumerate(examples):
            in_colors = set(c for row in ex['input'] for c in row if c != 0)
            out_colors = set(c for row in ex['output'] for c in row if c != 0)
            all_in_colors.update(in_colors)
            all_out_colors.update(out_colors)
        
        if all_in_colors == all_out_colors:
            structure_summary.append(f"✓ Colors PRESERVED: {all_in_colors}")
        elif len(all_in_colors) > len(all_out_colors):
            structure_summary.append(f"✓ Colors REDUCED: {all_in_colors} → {all_out_colors} (some removed)")
        elif all_in_colors != all_out_colors:
            structure_summary.append(f"✓ Colors CHANGED: {all_in_colors} → {all_out_colors}")
        
        # Object count analysis
        num_input_objs = analysis['num_objects_in']
        avg_in = sum(num_input_objs) / len(num_input_objs) if num_input_objs else 0
        structure_summary.append(f"✓ Average {avg_in:.1f} input objects per example")
        
        # Multiple colored objects?
        if len(all_in_colors) > 1:
            structure_summary.append(f"✓ MULTIPLE colored objects ({len(all_in_colors)} colors) - consider relationships!")
        
        # Symmetry
        if 'vertical' in analysis.get('symmetries', []):
            structure_summary.append("✓ Some objects have VERTICAL SYMMETRY")
        
        # Show detailed examples
        example_str = "\nDETAILED EXAMPLES:\n"
        for i, ex in enumerate(examples[:2]):
            in_grid = np.array(ex['input'])
            out_grid = np.array(ex['output'])
            
            example_str += f"\n{'='*60}\nExample {i+1}:\n"
            example_str += f"Input: {in_grid.shape[0]}×{in_grid.shape[1]}\n"
            example_str += f"Colors in: {set(in_grid.flatten()) - {0}}\n"
            
            # Show grid if small enough
            if in_grid.shape[0] <= 6 and in_grid.shape[1] <= 6:
                example_str += "Grid:\n"
                for row in in_grid:
                    example_str += "  " + str(row.tolist()) + "\n"
            else:
                # Show just non-zero regions
                example_str += "Non-zero regions:\n"
                for r in range(min(6, in_grid.shape[0])):
                    if any(in_grid[r] != 0):
                        example_str += f"  Row {r}: {in_grid[r].tolist()}\n"
            
            example_str += f"\nOutput: {out_grid.shape[0]}×{out_grid.shape[1]}\n"
            example_str += f"Colors out: {set(out_grid.flatten()) - {0}}\n"
            
            if out_grid.shape[0] <= 6 and out_grid.shape[1] <= 6:
                example_str += "Grid:\n"
                for row in out_grid:
                    example_str += "  " + str(row.tolist()) + "\n"
            else:
                example_str += "Non-zero regions:\n"
                for r in range(min(6, out_grid.shape[0])):
                    if any(out_grid[r] != 0):
                        example_str += f"  Row {r}: {out_grid[r].tolist()}\n"
        
        prompt = f"""Analyze these ARC-AGI examples and propose DIVERSE transformation hypotheses.

STRUCTURAL OBSERVATIONS:
{chr(10).join(structure_summary)}

{example_str}

ANALYSIS QUESTIONS TO CONSIDER:
1. Do objects move? (gravity, translation, arrangement)
2. Do colors change? (remapping, based on another object?)
3. Does the grid tile/repeat? (3×3, 2×3, etc.)
4. Are there multiple colored objects? (one might control the other)
5. Do objects disappear or get removed?
6. Are objects combined or separated?

YOUR TASK:
Propose 4-6 DIVERSE hypotheses, from simplest to most complex.

CRITICAL FORMAT (use EXACTLY):

HYPOTHESIS 1: [simplest explanation]
MDL: [cost]
Support: [count]/{len(examples)}

HYPOTHESIS 2: [alternative explanation]
MDL: [cost]
Support: [count]/{len(examples)}

HYPOTHESIS 3: [more complex if needed]
MDL: [cost]
Support: [count]/{len(examples)}

[Continue for 4-6 hypotheses]

REQUIREMENTS:
- Be SPECIFIC (e.g., "gravity_down" not "movement")
- Consider RELATIONSHIPS between colored objects
- Look at ALL examples, not just first
- If you see small+large objects, consider one affecting the other
- Start with low MDL (simple) first"""
        
        return prompt
    
    def _parse_hypotheses(self, response: str, num_examples: int) -> List[Hypothesis]:
        """Parse LLM response into Hypothesis objects."""
        hypotheses = []
        lines = response.split('\n')
        
        current_desc = ""
        current_mdl = None
        current_support = num_examples  # Default to full support
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for HYPOTHESIS marker
            if line.upper().startswith('HYPOTHESIS'):
                # Save previous hypothesis if exists
                if current_desc and current_mdl is not None:
                    op = self._parse_operator_description(current_desc)
                    hypotheses.append(Hypothesis(op, current_support))
                
                # Extract description from same line or next lines
                if ':' in line:
                    current_desc = line.split(':', 1)[1].strip()
                else:
                    current_desc = ""
                
                # Look ahead for MDL and Support on next lines
                current_mdl = None
                current_support = num_examples
                
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j].strip()
                    
                    if next_line.upper().startswith('MDL'):
                        try:
                            mdl_str = next_line.split(':', 1)[1].strip()
                            # Extract first number
                            import re
                            numbers = re.findall(r'\d+\.?\d*', mdl_str)
                            if numbers:
                                current_mdl = float(numbers[0])
                        except:
                            current_mdl = 3.0
                    
                    elif next_line.upper().startswith('SUPPORT'):
                        try:
                            support_str = next_line.split(':', 1)[1].strip()
                            if '/' in support_str:
                                num = support_str.split('/')[0].strip()
                                current_support = float(num)
                            else:
                                import re
                                numbers = re.findall(r'\d+', support_str)
                                if numbers:
                                    current_support = float(numbers[0])
                        except:
                            pass
                
                # If we found MDL in lookahead, we have a complete hypothesis
                if current_mdl is None:
                    current_mdl = 3.0  # Default
            
            i += 1
        
        # Save last hypothesis
        if current_desc and current_mdl is not None:
            op = self._parse_operator_description(current_desc)
            hypotheses.append(Hypothesis(op, current_support))
        
        # If no hypotheses parsed, create a fallback
        if not hypotheses:
            hypotheses.append(Hypothesis(PrimitiveOp('identity'), num_examples))
        
        return hypotheses
    
    def _parse_operator_description(self, desc: str) -> Operator:
        """Parse operator description into Operator object."""
        desc = desc.lower().strip()
        
        # Remove common filler words
        desc = desc.replace('operator:', '').replace('transformation:', '').strip()
        
        # Check for composition first (contains →, then, +, and, or "followed by")
        composition_markers = ['→', 'then', ' + ', ' and ', 'followed by', 'combine']
        is_composition = any(marker in desc for marker in composition_markers)
        
        if is_composition:
            # Split on composition markers
            for marker in ['→', 'then']:
                if marker in desc:
                    parts = [p.strip() for p in desc.split(marker)[:2]]
                    if len(parts) == 2:
                        op1 = self._parse_operator_description(parts[0])
                        op2 = self._parse_operator_description(parts[1])
                        return CompositeOp([op1, op2], 'sequential')
        
        # Check for specific patterns (most specific first)
        
        # Gravity/movement
        if 'gravity' in desc or 'fall' in desc or 'drop' in desc or 'sink' in desc:
            if 'down' in desc or 'bottom' in desc:
                return PrimitiveOp('gravity_down')
            elif 'up' in desc or 'top' in desc:
                return PrimitiveOp('gravity_up')
            elif 'left' in desc:
                return PrimitiveOp('gravity_left')
            elif 'right' in desc:
                return PrimitiveOp('gravity_right')
            return PrimitiveOp('gravity_down')  # Default
        
        # Movement/translation
        if 'move' in desc or 'shift' in desc or 'translate' in desc:
            if 'corner' in desc or 'arrange' in desc:
                return PrimitiveOp('move_objects_to_corner', {})
            if 'bottom' in desc or 'down' in desc:
                return PrimitiveOp('gravity_down')
            return PrimitiveOp('translate', {'dr': 0, 'dc': 0})
        
        # Arrangement
        if 'arrange' in desc or 'align' in desc or 'organize' in desc:
            if 'horizontal' in desc:
                return PrimitiveOp('align_objects', {'axis': 'horizontal'})
            elif 'vertical' in desc:
                return PrimitiveOp('align_objects', {'axis': 'vertical'})
            return PrimitiveOp('align_objects', {'axis': 'horizontal'})
        
        # Removal/clearing
        if 'clear' in desc or 'remove' in desc or 'delete' in desc or 'disappear' in desc:
            return PrimitiveOp('clear_objects', {})
        
        # Extraction/copying
        if 'extract' in desc or 'copy' in desc or 'duplicate' in desc:
            return PrimitiveOp('extract_pattern', {})
        
        # Tiling (check before color mapping)
        if 'tile' in desc or 'repeat' in desc:
            if '3' in desc and ('3' in desc[desc.index('3')+1:] if desc.index('3') < len(desc)-1 else False):
                return PrimitiveOp('tile_pattern', {'n_rows': 3, 'n_cols': 3})
            elif '2' in desc and '3' in desc:
                return PrimitiveOp('tile_pattern', {'n_rows': 2, 'n_cols': 3})
            return PrimitiveOp('tile_pattern', {'n_rows': 3, 'n_cols': 3})
        
        # Color mapping (very common)
        if 'color' in desc or 'map' in desc or '→' in desc or 'remap' in desc or 'change' in desc:
            if 'based' in desc or 'determine' in desc or 'depend' in desc:
                # Complex color mapping based on another object
                return PrimitiveOp('shape_based_color_map', {})
            return PrimitiveOp('color_map', {})
        
        # Geometric transforms
        if 'reflect' in desc or 'flip' in desc or 'mirror' in desc:
            if 'vertical' in desc or 'vert' in desc or 'up' in desc or 'down' in desc or 'v' == desc[-1]:
                return PrimitiveOp('reflect_vertical')
            elif 'horizontal' in desc or 'horiz' in desc or 'left' in desc or 'right' in desc or 'h' == desc[-1]:
                return PrimitiveOp('reflect_horizontal')
            return PrimitiveOp('reflect_vertical')  # Default
        
        if 'rotate' in desc or 'turn' in desc:
            if '180' in desc:
                return PrimitiveOp('rotate_180')
            elif '270' in desc:
                return PrimitiveOp('rotate_270')
            else:
                return PrimitiveOp('rotate_90')
        
        # Scale/resize
        if 'scale' in desc or 'resize' in desc or 'grow' in desc or 'shrink' in desc:
            return PrimitiveOp('scale', {})
        
        # Identity (last resort)
        if 'identity' in desc or 'none' in desc or 'unchanged' in desc:
            return PrimitiveOp('identity')
        
        # If we can't parse it, try to guess from keywords
        keywords = desc.split()
        for word in keywords:
            if word in PrimitiveOp.BASE_COSTS:
                return PrimitiveOp(word)
        
        # Ultimate fallback
        return PrimitiveOp('identity')
    
    def _execute_with_llm(self, hypothesis: Hypothesis, 
                         test_input: List[List[int]],
                         analysis: Dict) -> List[List[int]]:
        """Execute hypothesis on test input using symbolic execution when possible."""
        
        op_desc = hypothesis.program.describe().lower()
        
        if self.verbose:
            print(f"    Executing: {op_desc}")
        
        # Try symbolic execution first (0 bits, deterministic)
        result = self._try_symbolic_execution(op_desc, test_input, analysis)
        
        if result is not None:
            if self.verbose:
                print(f"    ✓ Symbolic execution succeeded (0 bits)")
            self.bits_spent += 0  # Symbolic is free!
            return result
        
        # Fall back to LLM execution (costs bits)
        if self.verbose:
            print(f"    ⚠ No symbolic executor, falling back to LLM (1 bit)")
        
        return self._execute_with_llm_fallback(hypothesis, test_input, analysis)
    
    def _try_symbolic_execution(self, op_desc: str, 
                                input_grid: List[List[int]],
                                analysis: Dict) -> Optional[List[List[int]]]:
        """Try to execute using symbolic operators. Returns None if not possible."""
        
        # Tiling
        if 'tile' in op_desc:
            return self._execute_tile_pattern(input_grid)
        
        # Color mapping
        if 'color_map' in op_desc or 'color' in op_desc:
            return self._execute_color_map(input_grid, op_desc, analysis)
        
        # Gravity
        if 'gravity_down' in op_desc or ('gravity' in op_desc and 'down' in op_desc):
            return self._execute_gravity_down(input_grid)
        if 'gravity_up' in op_desc or ('gravity' in op_desc and 'up' in op_desc):
            return self._execute_gravity_up(input_grid)
        if 'gravity_left' in op_desc:
            return self._execute_gravity_left(input_grid)
        if 'gravity_right' in op_desc:
            return self._execute_gravity_right(input_grid)
        
        # Reflections
        if 'reflect_vertical' in op_desc or ('reflect' in op_desc and 'vertical' in op_desc):
            return self._execute_reflect_vertical(input_grid)
        if 'reflect_horizontal' in op_desc or ('reflect' in op_desc and 'horizontal' in op_desc):
            return self._execute_reflect_horizontal(input_grid)
        
        # Rotations
        if 'rotate_90' in op_desc or ('rotate' in op_desc and '90' in op_desc):
            return self._execute_rotate_90(input_grid)
        if 'rotate_180' in op_desc or ('rotate' in op_desc and '180' in op_desc):
            return self._execute_rotate_180(input_grid)
        if 'rotate_270' in op_desc or ('rotate' in op_desc and '270' in op_desc):
            return self._execute_rotate_270(input_grid)
        
        # Identity
        if 'identity' in op_desc:
            return self._execute_identity(input_grid)
        
        # No symbolic executor available
        return None
    
    def _execute_tile_pattern(self, input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Symbolically execute tiling: tile 3x3 with alternating row flips."""
        try:
            input_arr = np.array(input_grid)
            h, w = input_arr.shape
            
            output = []
            
            # Pattern: normal block, flipped block, normal block (vertically)
            # Each block is h rows tall, tiled 3 times horizontally
            for block_idx in range(3):
                if block_idx % 2 == 0:
                    # Normal block - tile each row 3 times horizontally
                    for row in input_arr:
                        output_row = np.tile(row, 3).tolist()
                        output.append(output_row)
                else:
                    # Flipped block - flip horizontally then tile
                    for row in input_arr:
                        flipped_row = row[::-1]
                        output_row = np.tile(flipped_row, 3).tolist()
                        output.append(output_row)
            
            return output
        except Exception as e:
            return None
    
    def _execute_gravity_up(self, input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Objects float to the top."""
        try:
            input_arr = np.array(input_grid)
            h, w = input_arr.shape
            output = np.zeros_like(input_arr)
            
            for col in range(w):
                non_zero = [input_arr[row, col] for row in range(h) if input_arr[row, col] != 0]
                for i, val in enumerate(non_zero):
                    output[i, col] = val
            
            return output.tolist()
        except:
            return None
    
    def _execute_gravity_left(self, input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Objects slide to the left."""
        try:
            input_arr = np.array(input_grid)
            h, w = input_arr.shape
            output = np.zeros_like(input_arr)
            
            for row in range(h):
                non_zero = [input_arr[row, col] for col in range(w) if input_arr[row, col] != 0]
                for i, val in enumerate(non_zero):
                    output[row, i] = val
            
            return output.tolist()
        except:
            return None
    
    def _execute_gravity_right(self, input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Objects slide to the right."""
        try:
            input_arr = np.array(input_grid)
            h, w = input_arr.shape
            output = np.zeros_like(input_arr)
            
            for row in range(h):
                non_zero = [input_arr[row, col] for col in range(w) if input_arr[row, col] != 0]
                for i, val in enumerate(non_zero):
                    output[row, w - len(non_zero) + i] = val
            
            return output.tolist()
        except:
            return None
    
    def _execute_reflect_vertical(self, input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Flip top to bottom."""
        try:
            return input_grid[::-1]
        except:
            return None
    
    def _execute_reflect_horizontal(self, input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Flip left to right."""
        try:
            return [row[::-1] for row in input_grid]
        except:
            return None
    
    def _execute_rotate_90(self, input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Rotate 90 degrees clockwise."""
        try:
            return np.rot90(np.array(input_grid), k=-1).tolist()
        except:
            return None
    
    def _execute_rotate_180(self, input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Rotate 180 degrees."""
        try:
            return np.rot90(np.array(input_grid), k=2).tolist()
        except:
            return None
    
    def _execute_rotate_270(self, input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Rotate 270 degrees clockwise (90 counter-clockwise)."""
        try:
            return np.rot90(np.array(input_grid), k=1).tolist()
        except:
            return None
    
    def _execute_identity(self, input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """No transformation."""
        try:
            return input_grid
        except:
            return None
    
    def _execute_gravity_down(self, input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Symbolically execute gravity_down: objects fall to the bottom."""
        try:
            input_arr = np.array(input_grid)
            h, w = input_arr.shape
            output = np.zeros_like(input_arr)
            
            # For each column, make non-zero elements fall to the bottom
            for col in range(w):
                # Get all non-zero values in this column
                non_zero = [input_arr[row, col] for row in range(h) if input_arr[row, col] != 0]
                
                # Place them at the bottom of the column
                for i, val in enumerate(non_zero):
                    output[h - len(non_zero) + i, col] = val
            
            return output.tolist()
        except Exception as e:
            return None
    
    def _execute_color_map(self, input_grid: List[List[int]], 
                          op_desc: str, analysis: Dict) -> Optional[List[List[int]]]:
        """Symbolically execute color mapping."""
        try:
            # Try to infer the mapping from analysis
            if not analysis.get('color_mappings'):
                return None
            
            mapping = {}
            cm = analysis['color_mappings'][0]
            
            # Simple heuristic: map all input colors to output colors
            in_colors = sorted(list(cm['in']))
            out_colors = sorted(list(cm['out']))
            
            if len(in_colors) == len(out_colors):
                for i, c in enumerate(in_colors):
                    mapping[c] = out_colors[i]
            elif len(out_colors) == 1:
                # Map everything to single color
                for c in in_colors:
                    mapping[c] = list(out_colors)[0]
            else:
                return None
            
            # Apply mapping
            input_arr = np.array(input_grid)
            output = np.copy(input_arr)
            
            for old_color, new_color in mapping.items():
                output[input_arr == old_color] = new_color
            
            return output.tolist()
        except:
            return None
    
    def _execute_with_llm_fallback(self, hypothesis: Hypothesis, 
                         test_input: List[List[int]],
                         analysis: Dict) -> List[List[int]]:
        """Execute hypothesis using LLM (fallback)."""
        
        # Build a very concrete prompt with examples
        op_desc = hypothesis.program.describe()
        
        prompt = f"""Execute this transformation on the test input.

TRANSFORMATION RULE: {op_desc}

TEST INPUT:
{json.dumps(test_input)}

INSTRUCTIONS:
1. Analyze the input grid shape: {len(test_input)}x{len(test_input[0]) if test_input else 0}
2. Apply the transformation: {op_desc}
3. Return ONLY the output grid as a JSON array

EXAMPLES OF TRANSFORMATIONS:

tile_3x3:
- Input: 2x2 → Output: 6x6 (repeat 3 times horizontally and vertically with alternating reflections)

tile_pattern:
- Tile the input pattern, alternating reflections

color_map:
- Change colors: color_map(8→2) means replace all 8s with 2s

reflect_vertical:
- Flip top↔bottom

reflect_horizontal:  
- Flip left↔right

rotate_90:
- Rotate 90° clockwise

OUTPUT FORMAT (JSON array only):
[[1, 2, 3],
 [4, 5, 6]]"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You execute grid transformations. Output ONLY the JSON array, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic
                max_tokens=3000
            )
            
            self.bits_spent += 1
            
            result = self._parse_grid(response.choices[0].message.content)
            
            if result and len(result) > 0 and len(result[0]) > 0:
                return result
            
            # If parsing failed, try one more time with even more explicit instructions
            retry_prompt = f"""The test input is:
{json.dumps(test_input)}

Apply: {op_desc}

Output the transformed grid as a valid JSON 2D array. NO explanations, ONLY the array."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output JSON only."},
                    {"role": "user", "content": retry_prompt}
                ],
                temperature=0.0,
                max_tokens=3000
            )
            
            self.bits_spent += 1
            result = self._parse_grid(response.choices[0].message.content)
            
            return result if result else test_input
            
        except Exception as e:
            print(f"Execution failed: {e}")
            return test_input
    
    def _parse_grid(self, response: str) -> Optional[List[List[int]]]:
        """Extract grid from response."""
        import re
        
        response = response.strip()
        
        # Remove markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            # Find the JSON content between code fences
            start_idx = 1
            end_idx = len(lines)
            for i, line in enumerate(lines):
                if i > 0 and line.strip().startswith("```"):
                    end_idx = i
                    break
            response = "\n".join(lines[start_idx:end_idx])
        
        response = response.strip()
        
        # Try direct JSON parse
        try:
            grid = json.loads(response)
            if isinstance(grid, list) and len(grid) > 0 and all(isinstance(row, list) for row in grid):
                # Validate it's all integers
                for row in grid:
                    if not all(isinstance(x, (int, np.integer)) for x in row):
                        return None
                return grid
        except:
            pass
        
        # Try to find JSON array in text
        # Look for pattern [[...], [...], ...]
        match = re.search(r'\[\s*\[[\d,\s\[\]]+\]\s*\]', response, re.DOTALL)
        if match:
            try:
                grid = json.loads(match.group(0))
                if isinstance(grid, list) and len(grid) > 0 and all(isinstance(row, list) for row in grid):
                    return grid
            except:
                pass
        
        # Try line-by-line parsing
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        grid = []
        for line in lines:
            if line.startswith("[") and line.endswith("]"):
                try:
                    row = json.loads(line)
                    if isinstance(row, list) and all(isinstance(x, (int, np.integer)) for x in row):
                        grid.append(row)
                except:
                    continue
        
        if grid and len(grid) > 0:
            return grid
        
        return None
    
    def solve_directory(self, json_dir: str) -> Dict[str, Any]:
        """Solve all tasks in directory."""
        results = {}
        json_path = Path(json_dir)
        
        if not json_path.exists():
            raise ValueError(f"Directory not found: {json_dir}")
        
        json_files = sorted(json_path.glob("*.json"))
        print(f"Found {len(json_files)} tasks")
        print(f"Bit budget: {self.bit_budget} per task\n")
        
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
                print(f"  {status} MDL: {result.get('mdl', 'N/A')}, "
                      f"Bits: {result['bits_spent']}, "
                      f"Hypotheses: {result['num_hypotheses']}")
                print(f"  Rule: {result.get('hypothesis', 'None')}\n")
                
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                results[task_id] = {'error': str(e)}
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print summary with Living Map metrics."""
        total = len(results)
        errors = sum(1 for r in results.values() if 'error' in r)
        correct = sum(1 for r in results.values() if r.get('correct'))
        evaluated = sum(1 for r in results.values() if r.get('correct') is not None)
        
        total_bits = sum(r.get('bits_spent', 0) for r in results.values() 
                        if 'error' not in r)
        avg_bits = total_bits / (total - errors) if total > errors else 0
        
        avg_mdl = np.mean([r.get('mdl', 0) for r in results.values() 
                          if r.get('mdl') is not None])
        
        print("\n" + "="*60)
        print("LIVING MAP METRICS")
        print("="*60)
        print(f"Total tasks: {total}")
        print(f"Errors: {errors}")
        print(f"Evaluated: {evaluated}")
        print(f"Correct: {correct}/{evaluated} ({100*correct/evaluated if evaluated else 0:.1f}%)")
        print(f"\nGeneralization Velocity:")
        print(f"  Correct per bit: {correct/total_bits if total_bits > 0 else 0:.4f}")
        print(f"  Average bits spent: {avg_bits:.1f}")
        print(f"\nCompression:")
        print(f"  Average MDL: {avg_mdl:.2f} bits")


def main():
    parser = argparse.ArgumentParser(
        description="Living Map ARC-AGI Solver with DSL and MDL",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("json_dir", help="Directory containing ARC task JSON files")
    parser.add_argument("model", help="GPT model name", default="gpt-4", nargs="?")
    parser.add_argument("--bit-budget", type=int, default=100, 
                       help="Interaction bit budget per task")
    parser.add_argument("--output", default="results.json", 
                       help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed execution logging")
    
    args = parser.parse_args()
    
    print(f"Living Map ARC-AGI Solver")
    print(f"Model: {args.model}")
    print(f"Directory: {args.json_dir}")
    print(f"Bit Budget: {args.bit_budget}")
    if args.verbose:
        print(f"Verbose: ON")
    print()
    
    solver = LivingMapSolver(model=args.model, bit_budget=args.bit_budget, verbose=args.verbose)
    results = solver.solve_directory(args.json_dir)
    solver.print_summary(results)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()