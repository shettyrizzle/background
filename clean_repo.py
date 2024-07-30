import subprocess

# Path to the repository
repo_path = "C:\\Users\\satvi\\OneDrive\\Desktop\\BG"

# Run git-filter-repo to remove the large file with --force
subprocess.run([
    "git", "filter-repo", "--path", "unet_model.h5", "--invert-paths", "--force"
], cwd=repo_path, check=True)

# Clean the repository
subprocess.run(["git", "reflog", "expire", "--expire=now", "--all"], cwd=repo_path, check=True)
subprocess.run(["git", "gc", "--prune=now", "--aggressive"], cwd=repo_path, check=True)

# Push changes to the remote repository
subprocess.run(["git", "push", "--force", "origin", "beta"], cwd=repo_path, check=True)
