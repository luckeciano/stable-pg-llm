import os
import argparse
import wandb
import pandas as pd

def download_run_metrics(wandb_entity, wandb_project, run_name):
    api = wandb.Api()
    project_path = f"{wandb_entity}/{wandb_project}"
    
    print(f"Fetching runs from project: {project_path}")
    runs = api.runs(project_path)
    matching_runs = [run for run in runs if run.name == run_name]

    if not matching_runs:
        print(f"No runs found with name: {run_name}")
        return

    print(f"Found {len(matching_runs)} matching runs.")
    
    for run in matching_runs:
        run_id = run.id
        folder_name = f"plot_data/{run.name.split('/')[-1]}/wandb_{run_id}"
        os.makedirs(folder_name, exist_ok=True)

        print(f"Downloading metrics for run: {run_id} ({run.name})")
        history = run.history(samples=100000)  # increase limit if needed
        csv_path = os.path.join(folder_name, "metrics.csv")
        history.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, required=True, help="Name of the wandb project")
    parser.add_argument("--wandb_entity", type=str, required=True, help="Wandb entity (user or team)")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run to search for")
    args = parser.parse_args()

    download_run_metrics(args.wandb_entity, args.wandb_project, args.run_name)
