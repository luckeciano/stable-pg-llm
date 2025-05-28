# scripts/list_revisions.py
import re
import sys
from huggingface_hub import list_repo_commits

model_id = sys.argv[1]
max_steps = int(sys.argv[2])
first_step = int(sys.argv[3])
interval = int(sys.argv[4])
pattern = re.compile(r"Training in progress, step (\d+)")

# Use a dictionary to store the latest commit for each step
step_commits = {}
for commit in list_repo_commits(model_id):
    match = pattern.fullmatch(commit.title.strip())
    if match:
        step = int(match.group(1))
        if step >= first_step and step <= max_steps and (step - first_step) % interval == 0:
            # Only store if we haven't seen this step before
            # Since commits are ordered by date (newest first), the first one we see is the latest
            if step not in step_commits:
                step_commits[step] = commit.commit_id

# Convert to list and sort by step
revisions = [(step, commit_id) for step, commit_id in step_commits.items()]
revisions.sort(key=lambda x: x[0])

for step, commit_id in revisions:
    print(f"{step}:{commit_id}")
