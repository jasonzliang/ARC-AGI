#!/usr/bin/env python3

import os
import json
import argparse
import re
from openai import OpenAI

# ========================================
#  FINAL BENCHMARK RESULTS (ON EVAL)
# ========================================
# Model:           gpt-4o
# Total Tasks:     400
# Correct Solved:  24
# Success Rate:    6.00%
# ========================================


def get_rigorous_solver_prompt():
    """
    This is a new, highly-procedural prompt designed to force rigorous,
    step-by-step reasoning and prevent "roleplaying" and "hallucinated falsification".
    """
    return """
You are a hyper-rigorous, self-critical, and logical AI solver for the ARC-AGI benchmark.
Your goal is to find the single, simplest, most general rule (Minimum Description Length) that explains all training examples.
Do NOT use simple pattern matching. Do NOT guess. Be a formal logician.

You MUST follow this exact 5-step reasoning process:

---
**Step 1: Deep Perception & Object Typing**
- Analyze ALL training pairs.
- Describe the input grids not as pixels, but as "typed objects," "roles," and "relations."
- **Crucially: Explicitly search for 'controller' objects.** Is there a single object, color, or shape that *determines the rule* for the other objects? (e.g., "a blue pixel in the corner acts as a 'key' that sets the output color," or "the smallest shape's orientation determines the grid rotation").
- State your observations clearly.

---
**Step 2: Hypothesis Generation (H1)**
- Based on your perception, state your **Hypothesis 1 (H1)**.
- This rule must be a *composition* of core operators (e.g., "Move", "Copy", "Rotate", "Fill", "FindObjectByRole", "If-Then").
- The rule must explain *all* aspects of the transformation.

---
**Step 3: Rigorous Falsification Loop**
This is the most important step. You will now test H1 against EVERY training example, one by one. You must show your work.

* **Testing H1 on Training Example 1:**
    * **Input 1:** [Show compact input grid]
    * **Applying H1:** [Describe *exactly* how H1's operators transform Input 1]
    * **H1 Simulated Output:** [Show the *full grid* that H1 would produce]
    * **Actual Output 1:** [Show compact actual output grid]
    * **Result:** **[PASS]** or **[FAIL]**.
    * **Analysis (if FAIL):** [Explain *exactly why* H1 failed. e.g., "H1 produced a '2' at (0,1), but the actual output has a '3'."]

* **Testing H1 on Training Example 2:**
    * [Repeat the *entire* process above for Example 2]
    * ...

* [Repeat this process for *all* remaining training examples.]

---
**Step 4: Iteration or Selection**
- **IF H1 [FAIL]ed on *any* example:**
    * Your analysis of the failure is now your new starting point.
    * State **Hypothesis 2 (H2)**. H2 *must* be a new rule that specifically *corrects the failure* of H1.
    * You must then run H2 through the *entire* "Step 3: Rigorous Falsification Loop."
    * Continue this process (H3, H4...) until a hypothesis **[PASS]es all training examples.**

- **IF H1 [PASS]ed all examples:**
    * State this clearly.
    * (Optional: If you see another, *simpler* rule (better MDL), you may propose it as H2 and test it as well).

- **IF no hypothesis can be found:**
    * State this. Do not guess.

---
**Step 5: Final Application & Formatting**
- Once you have a single, verified hypothesis (e.g., H2) that passed *all* training examples:
    * State this final, verified rule.
    * Apply this one rule to the **Test Input 1**.
    * Show your work in the scratchpad.
    * Provide the final answer in the required format.

---
**Output Format**
1.  First, provide your full reasoning in a `<scratchpad>` tag. Follow the 5 steps above *explicitly*.
2.  After the scratchpad, provide ONLY the final predicted test output grid in a `<final_answer>` tag.
3.  The final answer **MUST** be a valid, minified JSON 2D array (a list of lists of integers).
    * **CORRECT EXAMPLE (for a 2x3 grid):** `[[1,2,3],[4,5,6]]`
    * **WRONG EXAMPLE:** `[[123],[456]]`
    * **WRONG EXAMPLE:** `1 2 3 \n 4 5 6`
"""

def grid_to_string(grid):
    """Converts a 2D list into a human-readable string."""
    if not grid:
        return "[empty grid]"
    # Use a compact representation for the prompt, but valid JSON-like
    return "[" + ",\n ".join(map(str, grid)) + "]"

def format_task(task_data):
    """Formats the ARC task JSON into a string for the prompt."""
    formatted_str = "TRAINING EXAMPLES:\n"
    for i, pair in enumerate(task_data['train']):
        formatted_str += f"---\nExample {i+1}:\n"
        formatted_str += "Input:\n" + grid_to_string(pair['input']) + "\n"
        formatted_str += "Output:\n" + grid_to_string(pair['output']) + "\n"

    formatted_str += "\nTESTING EXAMPLE:\n"
    if task_data['test']:
        pair = task_data['test'][0]
        formatted_str += f"---\nTest Input 1:\n"
        formatted_str += grid_to_string(pair['input']) + "\n"
        formatted_str += "Output (to be predicted):\n?\n"

    return formatted_str

def parse_final_answer(response_content):
    """Extracts the JSON grid from the <final_answer> tag."""
    match = re.search(r'<final_answer>(.*?)</final_answer>', response_content, re.DOTALL | re.IGNORECASE)
    if not match:
        print("Error: No <final_answer> tag found in the response.")
        return None

    answer_str = match.group(1).strip()
    try:
        # The model should output a valid JSON array string
        answer_json = json.loads(answer_str)
        if isinstance(answer_json, list) and (not answer_json or all(isinstance(row, list) for row in answer_json)):
            return answer_json
        else:
            print(f"Error: Final answer is not a valid 2D array: {answer_json}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from final answer. Error: {e}")
        print(f"Raw answer string: {answer_str}")
        return None

def solve_task(task_path, model_name, client):
    """
    Solves a single ARC task file and returns True if correct, False otherwise.
    """
    try:
        with open(task_path, 'r') as f:
            task_data = json.load(f)
    except Exception as e:
        print(f"Error loading task {task_path}: {e}")
        return False # Return failure

    task_id = os.path.basename(task_path)
    print(f"--- Solving Task: {task_id} using {model_name} ---")

    if not task_data.get('test') or not task_data['test']:
        print(f"Task {task_id} has no test cases. Skipping.")
        return False

    ground_truth_grid = task_data['test'][0]['output']

    # 1. Get the new, rigorous system prompt
    system_prompt = get_rigorous_solver_prompt()

    # 2. Format the task data into a user prompt
    user_prompt = format_task(task_data)

    # 3. Call the OpenAI API
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=4096,
        )

        response_content = response.choices[0].message.content

        # 4. Parse and print the reasoning (the "trajectory")
        print("\n<Model Scratchpad (Reasoning Trajectory):>")
        scratchpad_match = re.search(r'<scratchpad>(.*?)</scratchpad>', response_content, re.DOTALL | re.IGNORECASE)
        if scratchpad_match:
            print(scratchpad_match.group(1).strip())
        else:
            print("...No scratchpad found. Printing full response...")
            print(response_content)
        print("</Model Scratchpad>\n")

        # 5. Parse the final answer
        predicted_grid = parse_final_answer(response_content)

        # 6. Compare prediction to ground truth
        is_correct = False
        if predicted_grid:
            print(f"Predicted Output for {task_id}:")
            # Use grid_to_string for consistent pretty-printing
            print(grid_to_string(predicted_grid))

            if predicted_grid == ground_truth_grid:
                print("\n>>> Result: SUCCESS\n")
                is_correct = True
            else:
                print("\n>>> Result: FAILURE\n")
                print("Expected Output:")
                print(grid_to_string(ground_truth_grid))
        else:
            print(f"Failed to extract a valid prediction for {task_id}.")
            print("\n>>> Result: FAILURE (Parse Error)\n")
            print("Expected Output:")
            print(grid_to_string(ground_truth_grid))

        return is_correct

    except Exception as e:
        print(f"An error occurred while calling the OpenAI API: {e}")
        return False # Return failure

def main():
    parser = argparse.ArgumentParser(
        description='ARC-AGI Solver (v2 - Rigorous) based on "THE LIVING MAP".',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('json_directory', type=str, help='Directory containing ARC task JSON files.')
    parser.add_argument('gpt_model_name', type=str, help='Name of the GPT model to use (e.g., "gpt-4o", "gpt-4-turbo").')

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set this variable before running the solver:")
        print("export OPENAI_API_KEY='your_api_key_here'")
        return

    client = OpenAI(api_key=api_key)

    task_files = sorted([f for f in os.listdir(args.json_directory) if f.endswith('.json')])
    if not task_files:
        print(f"No .json files found in directory: {args.json_directory}")
        return

    total_files = len(task_files)
    correct_tasks = 0

    print(f"Found {total_files} tasks. Starting v2 solver...")

    for i, filename in enumerate(task_files):
        task_path = os.path.join(args.json_directory, filename)

        is_correct = solve_task(task_path, args.gpt_model_name, client)

        if is_correct:
            correct_tasks += 1

        current_total_tasks = i + 1
        current_rate = (correct_tasks / current_total_tasks) * 100

        print(f"\n--- Running Score: {correct_tasks} / {current_total_tasks} ({current_rate:.2f}%) ---")
        print("-" * 60)

    # Print final summary
    print("\n" * 3)
    print("=" * 40)
    print(" FINAL BENCHMARK RESULTS")
    print("=" * 40)
    print(f"Model:           {args.gpt_model_name}")
    print(f"Total Tasks:     {total_files}")
    print(f"Correct Solved:  {correct_tasks}")

    if total_files > 0:
        final_rate = (correct_tasks / total_files) * 100
        print(f"Success Rate:    {final_rate:.2f}%")
    else:
        print("Success Rate:    N/A (No tasks found)")
    print("=" * 40)


if __name__ == "__main__":
    main()