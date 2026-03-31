"""Push dataset2_split to Hugging Face Hub as to505to505/dataset2."""

from huggingface_hub import HfApi

REPO_ID = "to505to505/dataset2"
LOCAL_DIR = "/home/dsa/stenosis/data/dataset2_split"

api = HfApi()

# Create the repo if it doesn't exist (type="dataset")
api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

# Upload the entire folder (large folder mode for reliability)
api.upload_large_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type="dataset",
)

print(f"Done! Dataset uploaded to https://huggingface.co/datasets/{REPO_ID}")
