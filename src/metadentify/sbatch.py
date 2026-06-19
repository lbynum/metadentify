import os


def format_sbatch_script(
    local_run_command,
    out_dir,
    task_name,
    job_name,
    num_nodes,
    tasks_per_node,
    cpus_per_task,
    num_hours,
    gigs_memory,
    num_gpus,
):
    slurm_user_email = os.environ.get("SLURM_USER_EMAIL")
    slurm_account_name = os.environ.get("SLURM_ACCOUNT_NAME")
    singularity_env_path = os.environ.get("SINGULARITY_ENV_PATH")
    singularity_container_path = os.environ.get("SINGULARITY_CONTAINER_PATH")

    sbatch_script = f"""#!/bin/bash

#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={num_hours}:00:00
#SBATCH --mem={gigs_memory}GB
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --job-name={job_name}
#SBATCH --mail-user={slurm_user_email}
#SBATCH --mail-type=BEGIN,END
#SBATCH --output={out_dir}/{task_name}_%j_out.txt
#SBATCH --account={slurm_account_name}
#SBATCH --signal=SIGUSR1@90

module purge

# create job-specific W&B directory to prevent sqlite database locks on shared filesystems
JOB_WANDB_DIR="./wandb_slurm_tmp/$SLURM_JOB_ID"
mkdir -p $JOB_WANDB_DIR

export WANDB_DIR=$JOB_WANDB_DIR
export WANDB_CACHE_DIR=$JOB_WANDB_DIR
export WANDB_CONFIG_DIR=$JOB_WANDB_DIR

singularity exec --nv --overlay {singularity_env_path} {singularity_container_path} \\
    /bin/bash -c "source /ext3/env.sh; {local_run_command}"

# clean up local tmp folder after job finishes
rm -rf $JOB_WANDB_DIR
"""
    return sbatch_script
