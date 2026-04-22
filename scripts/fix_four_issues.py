"""
Fix Normalizer LLM prompt: only take first line of response.
Fix Code Generator: get_target_entity must return spec, not hardcoded string.
Fix Code Generator: load_objects method must be generated for multi-step.
Fix Code Generator: skill param name target_uid -> target_entity_name.
"""
import re

# Fix 1: Normalizer - take only first line of LLM response
normalizer_path = "/ssd/mkqin/workspace/VLABench/scripts/vlabench_agent/nodes/normalizer.py"
with open(normalizer_path) as f:
    content = f.read()

old = '''    try:
        response = llm.invoke(prompt)
        result = response.content.strip()'''
new = '''    try:
        response = llm.invoke(prompt)
        # 只取第一行，避免 LLM 返回多行 thinking 过程
        result = response.content.strip().split("\\n")[0].strip()'''
content = content.replace(old, new)
with open(normalizer_path, 'w') as f:
    f.write(content)
print("Fixed normalizer.py: take only first line of LLM response")

# Fix 2: Code Generator - get_target_entity must return spec (target_entity)
cg_path = "/ssd/mkqin/workspace/VLABench/scripts/vlabench_agent/nodes/code_generator.py"
with open(cg_path) as f:
    content = f.read()

old = '''        target_entity_name=target_entity or "target_entity",'''
new = '''        target_entity_name=target_entity,'''
content = content.replace(old, new)
with open(cg_path, 'w') as f:
    f.write(content)
print("Fixed code_generator.py: get_target_entity returns spec")

# Fix 3: Code Generator - MULTI_STEP_TEMPLATE must include load_objects method
# The template currently has no load_objects. For multi-step, we need to call
# _collect_entity_load_methods and include its output in the template.
# But the issue is that load_methods is empty because asset_status keys are uid
# (beaker_0) but entity templates look up by spec (beaker).
# The real fix: when asset_status only has uid keys, the load_methods won't generate.
# Let's ensure the template calls the parent's load_objects properly.

# Actually the real issue: build_entity_code needs xml_path for uid, but we're passing
# the uid (beaker_0) as entity_name. The template uses entity_name to look up
# name2class_xml. But name2class_xml uses spec keys, not uid.
#
# For multi-step template, we need to override load_objects entirely.
# Let me update MULTI_STEP_TEMPLATE to include a load_objects method.

# Read the current template
with open(cg_path) as f:
    content = f.read()

old_template = '''MULTI_STEP_TEMPLATE = (
    '@register.add_config_manager("{task_name}")\\n'
    'class {class_prefix}ConfigManager(BenchTaskConfigManager):\\n'
    '    def __init__(self, task_name, num_objects=[1, 1], **kwargs):\\n'
    '        super().__init__(task_name, num_objects, **kwargs)\\n'
    '\\n'
    '{load_methods}'
    '\\n'
    '    def get_instruction(self, target_entity, {extra_params}**kwargs):\\n'
    '        instruction = ["{instruction_template}"]\\n'
    '        self.config["task"]["instructions"] = instruction\\n'
    '\\n'
    '    def get_condition_config(self, target_entity, {extra_params}**kwargs):\\n'
    '        # 执行完即成功\\n'
    '        pass\\n'
    '\\n'
    '    def get_target_entity(self):\\n'
    '        return "{target_entity_name}"\\n'
    '\\n'
    '\\n'
    '@register.add_task("{task_name}")\\n'
    'class {class_prefix}Task(PrimitiveTask):\\n'
    '    def __init__(self, task_name, robot, **kwargs):\\n'
    '        super().__init__(task_name, robot=robot, **kwargs)\\n'
    '\\n'
    '    def get_expert_skill_sequence(self, physics):\\n'
    '        skill_sequence = [\\n'
    '{skill_lines}\\n'
    '        ]\\n'
    '        return skill_sequence\\n'
)'''

new_template = '''MULTI_STEP_TEMPLATE = (
    '@register.add_config_manager("{task_name}")\\n'
    'class {class_prefix}ConfigManager(BenchTaskConfigManager):\\n'
    '    def __init__(self, task_name, num_objects=[1, 1], **kwargs):\\n'
    '        super().__init__(task_name, num_objects, **kwargs)\\n'
    '\\n'
    '{load_methods}'
    '\\n'
    '    def get_instruction(self, target_entity, {extra_params}**kwargs):\\n'
    '        instruction = ["{instruction_template}"]\\n'
    '        self.config["task"]["instructions"] = instruction\\n'
    '\\n'
    '    def get_condition_config(self, target_entity, {extra_params}**kwargs):\\n'
    '        # 执行完即成功\\n'
    '        pass\\n'
    '\\n'
    '    def get_target_entity(self):\\n'
    '        return "{target_entity_name}"\\n'
    '\\n'
    '\\n'
    '@register.add_task("{task_name}")\\n'
    'class {class_prefix}Task(PrimitiveTask):\\n'
    '    def __init__(self, task_name, robot, **kwargs):\\n'
    '        super().__init__(task_name, robot=robot, **kwargs)\\n'
    '\\n'
    '    def get_expert_skill_sequence(self, physics):\\n'
    '        skill_sequence = [\\n'
    '{skill_lines}\\n'
    '        ]\\n'
    '        return skill_sequence\\n'
)'''

if old_template in content:
    print("Template found, no change needed")
else:
    print("WARNING: Template not found exactly, trying regex fix")

# Fix 4: SkillLib.pick param name is target_entity_name, not target_uid
# _format_skill_line_v2 uses uid_to_spec mapping to convert uid -> spec
# but the param name in the skill dict from Skill Planner is "target_uid"
# We need to rename it to "target_entity_name"
# The params come from Skill Planner atomic_sequence which uses "target_uid"
# We need to map "target_uid" -> "target_entity_name"

with open(cg_path) as f:
    content = f.read()

# In _format_skill_line_v2, rename the param key from "target_uid" to "target_entity_name"
# This is because SkillLib.pick expects target_entity_name
old_func = '''def _format_skill_line_v2(entry: Dict, uid_to_spec: Dict) -> str:
    """新格式：将单个 atomic_sequence 条目转为 partial(...) 调用字符串

    新格式中 params 的 uid 直接是 entity name（spec），不需要占位符替换。
    """
    skill = entry.get("skill", "")
    params = entry.get("params", {})

    # uid → spec 映射
    resolved_params = {}
    for k, v in params.items():
        if isinstance(v, str):
            resolved_params[k] = uid_to_spec.get(v, v)
        elif isinstance(v, list):
            resolved_params[k] = [
                uid_to_spec.get(item, item) if isinstance(item, str) else item
                for item in v
            ]
        else:
            resolved_params[k] = v

    param_parts = [_format_skill_param_v2(k, v) for k, v in resolved_params.items()]
    params_str = ", ".join(param_parts)
    if params_str:
        return f"partial(SkillLib.{skill}, {params_str}),"
    else:
        return f"partial(SkillLib.{skill}),"'''

new_func = '''def _format_skill_line_v2(entry: Dict, uid_to_spec: Dict) -> str:
    """新格式：将单个 atomic_sequence 条目转为 partial(...) 调用字符串

    新格式中 params 的 uid 直接是 entity name（spec），不需要占位符替换。
    SkillLib 参数名映射：target_uid -> target_entity_name 等。
    """
    skill = entry.get("skill", "")
    params = entry.get("params", {})

    # uid → spec 映射，同时重命名参数名
    resolved_params = {}
    param_rename = {
        "target_uid": "target_entity_name",
        "target_container_name": "target_container",
    }
    for k, v in params.items():
        # 重命名参数名
        k = param_rename.get(k, k)
        if isinstance(v, str):
            resolved_params[k] = uid_to_spec.get(v, v)
        elif isinstance(v, list):
            resolved_params[k] = [
                uid_to_spec.get(item, item) if isinstance(item, str) else item
                for item in v
            ]
        else:
            resolved_params[k] = v

    param_parts = [_format_skill_param_v2(k, v) for k, v in resolved_params.items()]
    params_str = ", ".join(param_parts)
    if params_str:
        return f"partial(SkillLib.{skill}, {params_str}),"
    else:
        return f"partial(SkillLib.{skill}),"'''

if old_func in content:
    content = content.replace(old_func, new_func)
    with open(cg_path, 'w') as f:
        f.write(content)
    print("Fixed code_generator.py: renamed target_uid -> target_entity_name")
else:
    print("WARNING: _format_skill_line_v2 not found exactly")

# Fix 5: For multi-step, generate a proper load_objects override
# The issue is that when load_methods is empty, the config manager class is empty
# and the parent's load_objects gets called, which uses get_entity_config with wrong key.
# Solution: always generate a load_objects override for multi-step tasks.
# Update _generate_multi_step_code to generate load_objects.

with open(cg_path) as f:
    content = f.read()

# Find _generate_multi_step_code and update it to generate load_objects
old_gen = '''    # load_methods 为空时填入 pass
    load_methods_str = all_load_methods if all_load_methods.strip() else "        pass"'''

new_gen = '''    # load_methods 为空时生成默认 load_objects
    if all_load_methods.strip():
        load_methods_str = all_load_methods
    else:
        # 为 multi-step 任务生成默认 load_objects
        # 从 asset_status 提取所有实体的 spec
        entity_specs = list(uid_to_spec.values())
        if entity_specs:
            # 使用 target_entity (spec) 加载第一个实体
            load_methods_str = f"        def load_objects(self, target_entity):\\n"
            load_methods_str += f"            from VLABench.configs.constant import name2class_xml\\n"
            load_methods_str += f"            if target_entity is None:\\n"
            load_methods_str += f"                return\\n"
            for spec in entity_specs:
                load_methods_str += f"            entity_config = dict(\\n"
                load_methods_str += f"                name=target_entity,\\n"
                load_methods_str += f"                xml_path=name2class_xml[\\"{spec}\\"][-1],\\n"
                load_methods_str += f"                position=[0.0, 0.0, 0.8],\\n"
                load_methods_str += f"            )\\n"
                load_methods_str += f"            entity_config[\\"class\\"] = name2class_xml[\\"{spec}\\"][0]\\n"
                load_methods_str += f"            entity_config[\\"randomness\\"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])\\n"
                load_methods_str += f"            self.config[\\"task\\"][" + "\\"components\\"] = self.config[\\"task\\"].get(" + "\\"components\\", []) + [entity_config]\\n"
        else:
            load_methods_str = "        pass"'''

if old_gen in content:
    content = content.replace(old_gen, new_gen)
    with open(cg_path, 'w') as f:
        f.write(content)
    print("Fixed code_generator.py: generate load_objects for multi-step")
else:
    print("WARNING: load_methods_str not found exactly")
    # Try to find it
    idx = content.find("load_methods_str")
    if idx >= 0:
        print(f"Found at position {idx}: {content[idx:idx+200]}")

# Verify syntax
import subprocess
result = subprocess.run(['python3', '-m', 'py_compile', cg_path], capture_output=True, text=True)
if result.returncode == 0:
    print(f"✓ {cg_path} syntax OK")
else:
    print(f"✗ {cg_path} syntax error: {result.stderr.decode()}")
