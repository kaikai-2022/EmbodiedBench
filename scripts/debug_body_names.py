"""调试脚本：打印 MuJoCo 模型中的所有 body names"""
import sys
sys.path.insert(0, "/ssd/mkqin/workspace/VLABench")

import os
os.environ["VLABENCH_ROOT"] = "/ssd/mkqin/workspace/VLABench/VLABench"

import VLABench.tasks.components  # 打破 circular import

from scripts.vlabench_agent.nodes.analyzer import analyzer_node
from scripts.vlabench_agent.nodes.normalizer import normalizer_node
from scripts.vlabench_agent.nodes.asset_manager import asset_manager_node
from scripts.vlabench_agent.nodes.skill_planner import skill_planner_node
from scripts.vlabench_agent.nodes.condition_planner import condition_planner_node
from scripts.vlabench_agent.nodes.code_generator import code_generator_node
from scripts.vlabench_agent.nodes.registration import registration_node
from scripts.vlabench_agent.nodes.simulation import simulation_node

import mujoco

instruction = "Pick the tube which contains CuSO4 from the tube stand and then pour it into the small beaker"

state = {"user_instruction": instruction, "messages": [], "_log_filepath": "/tmp/debug_body.log"}
state = {**state, **analyzer_node(state)}
state = {**state, **normalizer_node({"task_analysis": state["task_analysis"], "_log_filepath": "/tmp/debug_body.log"})}
state = {**state, **asset_manager_node({"normalized_context": state["normalized_context"], "_log_filepath": "/tmp/debug_body.log"})}

skill_state = {"normalized_context": state["normalized_context"], "asset_status": state["asset_status"], "messages": [], "_log_filepath": "/tmp/debug_body.log"}
cond_state = {"normalized_context": state["normalized_context"], "asset_status": state["asset_status"], "messages": [], "_log_filepath": "/tmp/debug_body.log"}
skill_result = skill_planner_node(skill_state)
cond_result = condition_planner_node(cond_state)
state = {**state, **skill_result, **cond_result}

state = {**state, **code_generator_node({
    "normalized_context": state["normalized_context"],
    "asset_status": state["asset_status"],
    "skill_plan": state["skill_plan"],
    "condition_plan": state.get("condition_plan"),
    "task_analysis": state["task_analysis"],
    "_log_filepath": "/tmp/debug_body.log"
})}
state = {**state, **registration_node({"task_analysis": state["task_analysis"], "normalized_context": state["normalized_context"], "asset_status": state["asset_status"], "skill_plan": state["skill_plan"], "task_module_path": state["task_module_path"], "_log_filepath": "/tmp/debug_body.log"})}

print(f'Registration: {state.get("registration_success")}')

if not state.get("registration_success"):
    print("Registration failed")
    sys.exit(1)

# 创建环境以检查 body names
from VLABench.envs.dm_env import LM4ManipDMEnv
from VLABench.configs import name2config

task_name = "pick_pour"
series_name = f"{task_name}_series"
config = name2config[series_name][0]
task_class = LM4ManipDMEnv.from_config(config)

print("\n" + "="*60)
print("All Body Names in MuJoCo Model:")
print("="*60)

raw_m = task_class.physics.model._model
for i in range(raw_m.nbody):
    body_name = mujoco.mj_id2name(raw_m, mujoco.mjtObj.mjOBJ_BODY, i)
    if body_name:
        print(f"  {i}: '{body_name}'")

print("\n" + "="*60)
print("All Joint Names:")
print("="*60)
for i in range(raw_m.njnt):
    jnt_name = mujoco.mj_id2name(raw_m, mujoco.mjtObj.mjOBJ_JOINT, i)
    if jnt_name:
        jnt_type = raw_m.jnt_type[i]
        jnt_type_name = ["free", "ball", "slide", "hinge"][jnt_type] if jnt_type < 4 else "unknown"
        print(f"  {i}: '{jnt_name}' (type={jnt_type_name})")

print("\n" + "="*60)
print("Entities in task:")
print("="*60)
for name, entity in task_class.task.entities.items():
    print(f"  Entity: '{name}'")
    print(f"    - mjcf_model.model: {entity.mjcf_model.model}")
    print(f"    - class: {entity.__class__.__name__}")
    # 检查是否有 subentities
    if hasattr(entity, 'subentities'):
        print(f"    - subentities: {list(entity.subentities.keys())}")
        for subname, subent in entity.subentities.items():
            print(f"      Subentity: '{subname}'")
            print(f"        - mjcf_model.model: {subent.mjcf_model.model}")
