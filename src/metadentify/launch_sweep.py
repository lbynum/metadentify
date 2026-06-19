import argparse
import os
import re
import subprocess

from dotenv import load_dotenv

from metadentify.sbatch import format_sbatch_script


def initialize_wandb_sweep(yaml_config_path):
    print(f"Initializing W&B sweep from {yaml_config_path} inside Singularity...")

    singularity_env_path = os.environ.get("SINGULARITY_ENV_PATH")
    singularity_container_path = os.environ.get("SINGULARITY_CONTAINER_PATH")

    inner_command = f"source /ext3/env.sh; wandb sweep {yaml_config_path}"

    singularity_cmd = [
        "singularity",
        "exec",
        "--nv",
        "--overlay",
        singularity_env_path,
        singularity_container_path,
        "/bin/bash",
        "-c",
        inner_command,
    ]

    result = subprocess.run(singularity_cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    if result.returncode != 0:
        print(f"Error initializing sweep:\n{output}")
        raise RuntimeError("Failed to create W&B sweep.")

    match = re.search(r"wandb agent\s+([^\s]+)", output)
    if not match:
        print(f"Could not parse sweep ID from W&B output:\n{output}")
        raise ValueError("Sweep ID regex match failed.")

    sweep_id = match.group(1)
    print(f"Successfully created sweep: {sweep_id}")
    return sweep_id


def launch_sweep_agents(
    sweep_id,
    num_agents,
    num_nodes,
    tasks_per_node,
    cpus_per_task,
    num_hours,
    gigs_memory,
    num_gpus,
    slurm_directory="slurm_scripts",
):
    os.makedirs(slurm_directory, exist_ok=True)
    base_sweep_id = sweep_id.split("/")[-1]

    sweep_out_dir = os.path.join("out", base_sweep_id)
    os.makedirs(sweep_out_dir, exist_ok=True)

    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT")

    local_run_command = f"wandb agent {sweep_id} --entity {wandb_entity} --project {wandb_project}"

    for i in range(1, num_agents + 1):
        task_name = f"swp_{base_sweep_id}_{i}"
        job_name = f"swp_{base_sweep_id[:6]}_{i}"

        sbatch_script = format_sbatch_script(
            local_run_command=local_run_command,
            out_dir=sweep_out_dir,
            task_name=task_name,
            job_name=job_name,
            num_nodes=num_nodes,
            tasks_per_node=tasks_per_node,
            cpus_per_task=cpus_per_task,
            num_hours=num_hours,
            gigs_memory=gigs_memory,
            num_gpus=num_gpus,
        )

        task_path = os.path.join(slurm_directory, f"{task_name}.sbatch")
        with open(task_path, "w") as out_file:
            out_file.write(sbatch_script)

        print(f"[{i}/{num_agents}] Submitting SLURM task: {task_name}")
        subprocess.run(f"sbatch {task_path}", shell=True)

    print(f"All {num_agents} agents submitted to queue")
    print(f"SLURM logs will be saved to: {sweep_out_dir}/")


def main():
    load_dotenv()
    required_keys = [
        "SINGULARITY_ENV_PATH",
        "SINGULARITY_CONTAINER_PATH",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
    ]
    for key in required_keys:
        if not os.environ.get(key):
            raise ValueError(f"Missing {key} in .env file.")

    parser = argparse.ArgumentParser(description="W&B Sweep Launcher for SLURM")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config", type=str, help="Path to your sweep_config.yaml to start a NEW sweep"
    )
    group.add_argument(
        "--sweep_id",
        type=str,
        help="Existing W&B sweep ID (e.g., entity/project/abcd1234) to RESUME an old sweep",
    )
    parser.add_argument(
        "--num_jobs", type=int, default=4, help="Number of parallel SLURM jobs/agents to launch"
    )

    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes per agent")
    parser.add_argument("--tasks_per_node", type=int, default=1, help="Tasks per node")
    parser.add_argument("--cpus", type=int, default=14, help="CPUs per task")
    parser.add_argument("--hours", type=int, default=5, help="Wall time in hours")
    parser.add_argument("--memory", type=int, default=32, help="Memory in GB")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per agent")

    args = parser.parse_args()

    if args.config:
        target_sweep_id = initialize_wandb_sweep(args.config)
    else:
        target_sweep_id = args.sweep_id
        print(f"Attaching {args.num_jobs} new agents to existing sweep: {target_sweep_id}")

    launch_sweep_agents(
        sweep_id=target_sweep_id,
        num_agents=args.num_jobs,
        num_nodes=args.nodes,
        tasks_per_node=args.tasks_per_node,
        cpus_per_task=args.cpus,
        num_hours=args.hours,
        gigs_memory=args.memory,
        num_gpus=args.gpus,
    )


if __name__ == "__main__":
    main()
