import argparse

from huggingface_hub import HfApi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a local folder to the Hugging Face Hub.")
    parser.add_argument("--folder_path", required=True, help="Local folder to upload.")
    parser.add_argument("--repo_id", required=True, help="Target Hub repo, for example OpenEnvisionLab/ARR-RPO.")
    parser.add_argument("--repo_type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--path_in_repo", default=None, help="Optional target path inside the Hub repo.")
    args = parser.parse_args()

    HfApi().upload_folder(
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
    )
