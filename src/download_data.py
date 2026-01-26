import os
from huggingface_hub import snapshot_download

def download_pills_dataset():
    """Download the 'gwenxin/pills_inside_bottles' dataset from Hugging Face."""
    repo_id = "gwenxin/pills_inside_bottles"
    local_dir = "data/raw/pills_inside_bottles"
    
    print(f"Downloading {repo_id} to {local_dir}...")
    
    # Check if download is already complete (simple check)
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"Directory {local_dir} is not empty. Assuming already downloaded.")
        return

    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="dataset")
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_pills_dataset()
