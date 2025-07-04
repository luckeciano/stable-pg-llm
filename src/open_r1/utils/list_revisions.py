# scripts/list_revisions.py
import re
import sys
import yaml
from huggingface_hub import list_repo_commits, list_models
import wandb
import math

model_id_prefix = sys.argv[1]
max_steps = int(sys.argv[2])
first_step = int(sys.argv[3])
interval = int(sys.argv[4])
wandb_entity = sys.argv[5]
wandb_project = sys.argv[6]
hf_prefix = sys.argv[7]
wandb_prefix = sys.argv[8]
pattern = re.compile(r"Training in progress, step (\d+)")

# Construct full names
wandb_run_name = f"{wandb_prefix}/{model_id_prefix}"
hf_model_prefix = f"{hf_prefix}/{model_id_prefix}"

# Initialize wandb API
api = wandb.Api(timeout=60)

def is_step_logged_in_wandb(run, step):
    try:
        for row in run.scan_history():
            step_val = row.get('train/global_step')
            metric_val = row.get('test/all/extractive_match')
            if step_val == step and metric_val is not None and not (isinstance(metric_val, float) and math.isnan(metric_val)):
                print(f"  Step {step} has extractive_match: {metric_val}")
                return True

        return False
    except Exception as e:
        print(f"  Warning: Could not check wandb history for step {step}: {e}")
        return False

# Get runs in the project that match the full run name
print(f"Fetching wandb runs from {wandb_entity}/{wandb_project} with name '{wandb_run_name}'...")
runs = api.runs(f"{wandb_entity}/{wandb_project}", filters={"display_name": wandb_run_name})

# Create a mapping from seed to run object and run path for runs matching the prefix
seed_to_run = {}
seed_to_run_path = {}
print(f"Processing {len(runs)} matching runs...")
print(runs)
for i, run in enumerate(runs):
    print(f"Processing run {i+1}/{len(runs)}: {run.id}")
    try:
        config_file = run.file("config.yaml")
        config_content = config_file.download(replace=True)
        config = yaml.safe_load(config_content)
        
        # Extract seed from config
        seed = config['seed']['value']
        print(f"  Seed: {seed}")
        run_path = f"{wandb_entity}/{wandb_project}/{run.id}"
        
        # Get a fresh run object to ensure we have the latest data
        fresh_run = api.run(run_path)
        seed_to_run[seed] = fresh_run
        seed_to_run_path[seed] = run_path
        print(f"  Found run for seed {seed}: {run_path}")
            
    except Exception as e:
        print(f"  Error processing run {run.id}: {e}")
        continue

# Iterate over the known seeds and construct model IDs
print(f"\nProcessing models for {len(seed_to_run)} seeds...")
for seed, wandb_run_path in seed_to_run_path.items():
    model_id = f"{hf_prefix}/{model_id_prefix}_{seed}"
    print(f"\nProcessing model: {model_id}")
    
    # Get the corresponding wandb run object
    wandb_run = seed_to_run[seed]
    
    # Use a dictionary to store the latest commit for each step
    step_commits = {}
    try:
        for commit in list_repo_commits(model_id):
            match = pattern.fullmatch(commit.title.strip())
            if match:
                step = int(match.group(1))
                if step >= first_step and step <= max_steps and (step - first_step) % interval == 0:
                    # Check if this step is already logged in wandb
                    if not is_step_logged_in_wandb(wandb_run, step):
                        print(f"  Found step {step} - not logged in wandb")
                        # Only store if we haven't seen this step before
                        # Since commits are ordered by date (newest first), the first one we see is the latest
                        if step not in step_commits:
                            step_commits[step] = commit.commit_id
                    else:
                        print(f"  Skipping step {step} - already logged in wandb")

        # Convert to list and sort by step
        revisions = [(step, commit_id) for step, commit_id in step_commits.items()]
        revisions.sort(key=lambda x: x[0])

        for step, commit_id in revisions:
            print(f"{model_id}:{step}:{commit_id}:{wandb_run_path}")
            
    except Exception as e:
        print(f"Error processing {model_id}: {e}")
        continue
