"""
Remove 'current_stage' from all node return dicts.
LangGraph controls navigation via graph edges, not current_stage.
current_stage caused concurrent update errors when multiple nodes returned it in the same step.
"""
import re
import os

os.chdir("/ssd/mkqin/workspace/VLABench/scripts/vlabench_agent/nodes")

for fname in ['analyzer.py', 'normalizer.py', 'asset_manager.py', 'skill_planner.py', 'code_generator.py', 'vlm_data.py']:
    path = fname
    with open(path) as f:
        content = f.read()

    # Remove "current_stage": "...", from dict literals (4-space indented)
    new_content = re.sub(r'    "current_stage": "[^"]*",\n', '', content)

    if new_content != content:
        with open(path, 'w') as f:
            f.write(new_content)
        count = len(re.findall(r'"current_stage": "[^"]*"', content))
        print(f'Fixed {fname}: removed {count} occurrences')
    else:
        print(f'No changes in {fname}')
