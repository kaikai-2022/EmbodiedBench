import os
import argparse
from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.openvla import OpenVLA
from VLABench.evaluation.model.policy.base import RandomPolicy
from VLABench.tasks import *
from VLABench.robots import *

os.environ["MUJOCO_GL"]= "egl"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=None, help="Specific tasks to run, work when eval-track is None")
    parser.add_argument('--eval-track', default=None, type=str, choices=["track_1_in_distribution", "track_2_cross_category", "track_3_common_sense", "track_4_semantic_instruction", "track_6_unseen_texture"], help="The evaluation track to run")
    parser.add_argument('--n-episode', default=1, type=int, help="The number of episodes to evaluate for a task")
    parser.add_argument('--policy', default="openvla", help="The policy to evaluate")
    parser.add_argument('--model_ckpt', default="/remote-home1/sdzhang/huggingface/openvla-7b", help="The base model checkpoint path")
    parser.add_argument('--lora_ckpt', default="/remote-home1/pjliu/openvla/weights/vlabench/select_fruit+CSv1+lora/", help="The lora checkpoint path")
    parser.add_argument('--save-dir', default="logs", help="The directory to save the evaluation results")
    parser.add_argument('--visulization', action="store_true", default=False, help="Whether to visualize the episodes")
    parser.add_argument('--metrics', nargs='+', default=["success_rate"], choices=["success_rate", "intention_score", "progress_score"], help="The metrics to evaluate")
    parser.add_argument('--host', default="localhost", type=str, help="The host to the remote server")
    parser.add_argument('--port', default=5555, type=int, help="The port to the remote server")
    parser.add_argument('--replanstep', default=4, type=int, help="The step to replan")
    parser.add_argument('--policy_config', default=None, type=str, help="Path to DP policy config YAML")
    parser.add_argument('--normalizer_path', default=None, type=str, help="Path to normalization statistics")
    args = parser.parse_args()
    return args

def evaluate(args):
    episode_config = None
    if args.eval_track is not None:
        args.save_dir = os.path.join(args.save_dir, args.eval_track)
        with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/evaluation/tracks", f"{args.eval_track}.json"), "r") as f:
            episode_config = json.load(f)
            tasks = list(episode_config.keys())
    if args.tasks is not None:
        tasks = args.tasks
    assert isinstance(tasks, list)

    evaluator = Evaluator(
        tasks=tasks,
        n_episodes=args.n_episode,
        episode_config=episode_config,
        max_substeps=1, # repeat step in simulation
        save_dir=args.save_dir,
        visulization=args.visulization,
        metrics=args.metrics
    )
    if args.policy.lower() == "openvla":
        policy = OpenVLA(
            model_ckpt=args.model_ckpt,
            lora_ckpt=args.lora_ckpt,
            norm_config_file=os.path.join(os.getenv("VLABENCH_ROOT"), "configs/model/openvla_config.json") # TODO: re-compuate the norm state by your own dataset
        )
    elif args.policy.lower() == "gr00t":
        from VLABench.evaluation.model.policy.gr00t import Gr00tPolicy
        policy = Gr00tPolicy(host=args.host, port=args.port, replan_steps=args.replanstep)
    elif args.policy.lower() == "openpi":
        from VLABench.evaluation.model.policy.openpi import OpenPiPolicy
        policy = OpenPiPolicy(host=args.host, port=args.port, replan_steps=args.replanstep)
    elif args.policy.lower() == "act":
        from VLABench.evaluation.model.policy.act import ACTPolicy
        # Check if model_ckpt is provided and valid
        use_checkpoint = (hasattr(args, 'model_ckpt') and
                         args.model_ckpt and
                         args.model_ckpt != "none" and
                         not args.model_ckpt.startswith("/remote-home"))
        policy = ACTPolicy(
            pretrained_policy_path=args.model_ckpt if use_checkpoint else None,
            camera_indices=[2, 3],  # front + wrist
            replan_steps=args.replanstep,
            device="cuda",
        )
    elif args.policy.lower() == "dp":
        from VLABench.evaluation.model.policy.dp import DPPolicy
        # Check if model_ckpt is provided and valid
        use_checkpoint = (hasattr(args, 'model_ckpt') and
                         args.model_ckpt and
                         args.model_ckpt != "none" and
                         not args.model_ckpt.startswith("/remote-home"))
        policy = DPPolicy(
            pretrained_policy_path=args.model_ckpt if use_checkpoint else None,
            policy_config_path=getattr(args, 'policy_config', None),
            normalizer_path=getattr(args, 'normalizer_path', None),
            camera_indices=[2, 3],  # front + wrist
            replan_steps=args.replanstep,
            device="cuda",
        )
    else:
        policy = RandomPolicy(None)

    result = evaluator.evaluate(policy)
    if args.eval_track:
        os.makedirs(os.path.join(args.save_dir, args.policy, args.eval_track), exist_ok=True)
        result_path = os.path.join(args.save_dir, args.policy, args.eval_track, "evaluation_result.json")
    else:
        os.makedirs(os.path.join(args.save_dir, args.policy), exist_ok=True)
        result_path = os.path.join(args.save_dir, args.policy, "evaluation_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    args = get_args()
    evaluate(args)