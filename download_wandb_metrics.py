import os
import argparse
import wandb
import pandas as pd

def download_run_metrics(wandb_entity, wandb_project, run_name):
    """Download metrics for a single run name."""
    api = wandb.Api()
    project_path = f"{wandb_entity}/{wandb_project}"
    
    print(f"Fetching runs from project: {project_path}")
    runs = api.runs(project_path)
    matching_runs = [run for run in runs if run.name == run_name]

    if not matching_runs:
        print(f"No runs found with name: {run_name}")
        return False

    print(f"Found {len(matching_runs)} matching runs.")
    
    success_count = 0
    for run in matching_runs:
        try:
            run_id = run.id
            folder_name = f"plot_data/{run.name.split('/')[-1]}/wandb_{run_id}"
            os.makedirs(folder_name, exist_ok=True)

            print(f"Downloading metrics for run: {run_id} ({run.name})")
            history = run.history(samples=100000)  # increase limit if needed
            csv_path = os.path.join(folder_name, "metrics.csv")
            history.to_csv(csv_path, index=False)
            print(f"Saved to {csv_path}")
            success_count += 1
        except Exception as e:
            print(f"Error downloading run {run.id} ({run.name}): {e}")
            continue
    
    return success_count > 0

def download_multiple_runs(wandb_entity, wandb_project, run_names):
    """Download metrics for multiple run names sequentially."""
    print(f"Starting download for {len(run_names)} run names...")
    
    successful_downloads = 0
    failed_downloads = 0
    
    for i, run_name in enumerate(run_names, 1):
        print(f"\n--- Processing run {i}/{len(run_names)}: {run_name} ---")
        try:
            success = download_run_metrics(wandb_entity, wandb_project, run_name)
            if success:
                successful_downloads += 1
                print(f"✓ Successfully processed: {run_name}")
            else:
                failed_downloads += 1
                print(f"✗ No runs found for: {run_name}")
        except Exception as e:
            failed_downloads += 1
            print(f"✗ Error processing {run_name}: {e}")
    
    print(f"\n--- Download Summary ---")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"Total processed: {len(run_names)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, required=True, help="Name of the wandb project")
    parser.add_argument("--wandb_entity", type=str, required=True, help="Wandb entity (user or team)")
    parser.add_argument("--run_name", type=str, help="Name of a single run to search for")
    parser.add_argument("--run_names", type=str, help="Comma-separated list of run names to download")
    args = parser.parse_args()

    if args.run_names:
        # Download multiple runs
        run_names = [name.strip() for name in args.run_names.split(",") if name.strip()]
        if not run_names:
            print("Error: No valid run names provided in --run_names")
            exit(1)
        download_multiple_runs(args.wandb_entity, args.wandb_project, run_names)
    elif args.run_name:
        # Download single run (backward compatibility)
        download_run_metrics(args.wandb_entity, args.wandb_project, args.run_name)
    else:
        print("Error: Must provide either --run_name or --run_names")
        parser.print_help()
        exit(1)
