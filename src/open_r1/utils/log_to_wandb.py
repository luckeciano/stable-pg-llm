# scripts/log_to_wandb.py
import os
import sys
import json
import wandb
import glob

if len(sys.argv) != 4:
    print("Usage: log_to_wandb.py <wandb_run_path> <results_dir> <step>")
    sys.exit(1)

wandb_run_path = sys.argv[1]  # e.g., lucelo/maxent-eval/abc123def456
results_dir = sys.argv[2]
step = int(sys.argv[3])

print("Results dir: ", results_dir)
# Split into components
try:
    entity, project, run_id = wandb_run_path.strip().split('/')
except ValueError:
    print("WANDB_RUN_PATH must be of the form 'entity/project/run_id'")
    sys.exit(1)

run = wandb.init(entity=entity, project=project, id=run_id, resume="allow")

# First pass: collect all metric names
all_metrics = set()
for result_file in glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True):
    with open(result_file) as f:
        data = json.load(f)
    if "results" in data:
        for metric_name, metric_data in data["results"].items():
            for value_name, value in metric_data.items():
                if isinstance(value, (int, float)):
                    all_metrics.add(f"test/{metric_name}/{value_name}")

# Define all metrics that can be logged out of order
for metric_name in all_metrics:
    run.define_metric(metric_name, step_metric="train/global_step", summary="last")

for result_file in glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True):
    print(result_file)
    with open(result_file) as f:
        data = json.load(f)

    task_name = os.path.basename(result_file).replace(".json", "")

    # Extract metrics from the nested structure
    metrics = {}
    if "results" in data:
        for metric_name, metric_data in data["results"].items():
            # Assuming each metric has two values to log
            for value_name, value in metric_data.items():
                if isinstance(value, (int, float)):
                    metrics[f"test/{metric_name}/{value_name}"] = value
    else:
        print(f"Warning: No 'results' key found in {result_file}")

    # Log the metrics to wandb using train/global_step
    if metrics:
        # Add the global step to the metrics
        metrics["train/global_step"] = step
        run.log(metrics)
        print(f"Logged metrics for task {task_name} at step {step}")
    else:
        print(f"Warning: No metrics found to log for {result_file}")

print("Finished logging to WandB")
run.finish()
