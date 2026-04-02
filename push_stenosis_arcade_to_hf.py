"""Push stenosis_arcade to Hugging Face Hub as to505to505/stenosis_arcade."""

from huggingface_hub import HfApi

REPO_ID = "to505to505/stenosis_arcade"
LOCAL_DIR = "/home/dsa/stenosis/data/stenosis_arcade"

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
