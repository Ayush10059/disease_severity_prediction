import os
import getpass
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd


BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
BRONZE_CXR_DIR = "bronze/mimic_cxr"
LIST_FILE = "bronze/mimic_cxr/cohort_image_filenames.txt"

cohort_path = "silver/balanced_multimodal_cohort.parquet"
out_list_path = "bronze/mimic_cxr/cohort_image_filenames.txt"

def image_path():
    df = pd.read_parquet(cohort_path)

    # Extract unique relative paths (e.g., files/p10/p10000032/s50414267/....jpg)
    image_paths = df["jpg_rel_path"].dropna().unique()

    with open(out_list_path, "w") as f:
        for path in image_paths:
            f.write(f"{path}\n")

    print(f"Exported {len(image_paths)} unique image paths to {out_list_path}")

def main():
    username = "abajra"
    password = getpass.getpass(f"PhysioNet Password for {username}: ")

    with open(LIST_FILE, "r") as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    session = requests.Session()
    session.auth = (username, password)

    session.headers.update({
        "User-Agent": "Wget/1.21.4",
        "Accept": "*/*",
        "Connection": "keep-alive",
    })

    def fetch_one(rel_path: str):
        local_path = os.path.join(BRONZE_CXR_DIR, rel_path)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return "skipped"

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        resp = session.get(BASE_URL + rel_path, stream=True, timeout=30)
        if resp.status_code == 200:
            temp_path = local_path + ".part"
            with open(temp_path, "wb") as out_f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        out_f.write(chunk)
            os.replace(temp_path, local_path)
            return "downloaded"
        return f"err_{resp.status_code}"

    stats = {"downloaded": 0, "skipped": 0, "errors": 0}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_one, p): p for p in rel_paths}
        with tqdm(total=len(rel_paths), desc="Downloading CXRs", unit="img") as pbar:
            for future in as_completed(futures):
                res = future.result()
                if res == "downloaded":
                    stats["downloaded"] += 1
                elif res == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["errors"] += 1
                pbar.set_postfix(dl=stats["downloaded"], skip=stats["skipped"], err=stats["errors"])
                pbar.update(1)
                
if __name__ == "__main__":
    image_path()
    main()
