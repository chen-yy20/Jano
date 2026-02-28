#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys


SCRIPT_MAP = {
    "wan": {
        "jano": "run_wan/jano_generate.py",
        "pab": "run_wan/pab_generate.py",
        "teacache": "run_wan/teacache_generate.py",
        "toca": "run_wan/toca_generate.py",
    },
    "flux": {
        "jano": "run_flux/generate_flux_jano.py",
        "pab": "run_flux/generate_flux_pab.py",
        "teacache": "run_flux/generate_flux_teacache.py",
        "toca": "run_flux/generate_flux_toca.py",
    },
    "cvx": {"jano": "run_cvx/jano_generate.py"},
    "ras": {"jano": "ras_exp/jano_sd3.py", "ras": "ras_exp/ras_sd3.py"},
}


def build_parser():
    parser = argparse.ArgumentParser(description="Unified launcher for Jano scripts")
    parser.add_argument("--model", choices=SCRIPT_MAP.keys(), required=True, help="Model family")
    parser.add_argument("--method", required=True, help="Method name in selected model family")
    parser.add_argument("--launcher", choices=["python", "srun"], default="python", help="Launch backend")
    parser.add_argument("--script", default=None, help="Optional custom script path")
    parser.add_argument("--partition", default="debug", help="SLURM partition for srun")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes for srun")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="GPUs per node for srun")
    parser.add_argument("--job-name", default="jano-run", help="SLURM job name for srun")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Extra args passed to target script")
    return parser


def resolve_script(model, method, script):
    if script:
        return script
    if method not in SCRIPT_MAP[model]:
        raise ValueError(
            f"Unknown method '{method}' for model '{model}'. "
            f"Available: {', '.join(sorted(SCRIPT_MAP[model].keys()))}"
        )
    return SCRIPT_MAP[model][method]


def main():
    args = build_parser().parse_args()
    script_args = args.script_args
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    script = resolve_script(args.model, args.method, args.script)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    if env.get("PYTHONPATH") and env["PYTHONPATH"].strip():
        env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]])
    else:
        env["PYTHONPATH"] = repo_root

    if args.launcher == "python":
        cmd = [sys.executable, script, *script_args]
    else:
        env.update({"NNODES": str(args.nnodes), "GPUS_PER_NODE": str(args.gpus_per_node)})
        cmd = [
            "srun",
            "-p",
            args.partition,
            "-K",
            "-N",
            str(args.nnodes),
            "--job-name",
            args.job_name,
            "--ntasks-per-node",
            str(args.gpus_per_node),
            "--gres",
            f"gpu:{args.gpus_per_node}",
            "--export=ALL",
            "bash",
            "infer.sh",
            script,
            *script_args,
        ]

    print("Launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
