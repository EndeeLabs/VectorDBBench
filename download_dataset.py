import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ===================================================================
# ======================== USER CONFIGURATION =======================

# 1. SELECT DATASET
DATASET_NAME = "cohere"  # Options: "cohere", "laion"

# 2. ROOT DIRECTORY
ROOT_DIR = "/home/admin/vectordataset"

# ===================================================================

# 3. DATASET DETAILS
DATASET_CONFIG = {
    "cohere": {
        "base_url": "https://assets.zilliz.com/benchmark/cohere_large_10m/",
        "rel_path": "cohere/cohere_large_10m",  # Subfolder inside ROOT_DIR
        "train_prefix": "shuffle_train",
        "train_count": 10,
        "train_suffix": "of-10.parquet",
        "extra_files": ["scalar_labels.parquet"] 
    },
    "laion": {
        "base_url": "https://assets.zilliz.com/benchmark/laion_large_100m/",
        "rel_path": "laion/laion_large_100m",   # Subfolder inside ROOT_DIR
        "train_prefix": "train",
        "train_count": 100,
        "train_suffix": "of-100.parquet",
        "extra_files": []
    }
}

# 4. GLOBAL SETTINGS
MAX_WORKERS = 10 


def get_file_list(config):
    """Generates the list of files based on the dataset config."""
    files = ["test.parquet", "neighbors.parquet"]

    # Add extra files (like scalar_labels) if they exist in config
    files.extend(config["extra_files"])

    # Generate Training Files
    # Example: shuffle_train-00-of-10.parquet OR train-00-of-100.parquet
    prefix = config["train_prefix"]
    count = config["train_count"]
    suffix = config["train_suffix"]

    for i in range(count):
        # Format usually needs leading zeros (00, 01..99)
        files.append(f"{prefix}-{i:02d}-{suffix}")
    
    return files

def download_file(args):
    """
    Downloads a single file. args is (filename, base_url, full_local_path)
    """
    filename, base_url, dest_dir = args
    local_file_path = os.path.join(dest_dir, filename)
    url = base_url + filename

    # 1. Skip if fully downloaded (File > 1KB)
    if os.path.exists(local_file_path) and os.path.getsize(local_file_path) > 1024:
        return f"✅ Skipped (Exists): {filename}"

    try:
        # 2. Stream download
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code == 404:
                return f"❌ Missing (404): {filename}"
            
            r.raise_for_status()
            
            # Write to file
            with open(local_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                    f.write(chunk)
                  
        return f"⬇️ Downloaded: {filename}"
        
    except Exception as e:
        return f"⚠️ Error {filename}: {str(e)}"

if __name__ == "__main__":
    if DATASET_NAME not in DATASET_CONFIG:
        print(f"❌ Error: Invalid dataset name '{DATASET_NAME}'")
        exit(1)

    cfg = DATASET_CONFIG[DATASET_NAME]
    
    # Construct the full destination path
    full_dest_path = os.path.join(ROOT_DIR, cfg["rel_path"])
    
    # Ensure directory exists
    os.makedirs(full_dest_path, exist_ok=True)
    
    files_to_download = get_file_list(cfg)

    print(f"--- Unified Downloader: {DATASET_NAME.upper()} ---")
    print(f"Source:      {cfg['base_url']}")
    print(f"Destination: {full_dest_path}")
    print(f"File Count:  {len(files_to_download)}")
    print(f"Threads:     {MAX_WORKERS}\n")
    
    # Pack arguments: (filename, base_url, full_dest_path)
    map_args = [(f, cfg["base_url"], full_dest_path) for f in files_to_download]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(download_file, map_args), 
            total=len(files_to_download), 
            unit="file"
        ))

    failures = [r for r in results if "Error" in r or "Missing" in r]
    if failures:
        print("\nSummary of Issues:")
        for fail in failures:
            print(fail)
    else:
        print("\n🎉 All files processed successfully!")






# # =================================================================== #
# # ==================== FOR COHERE 10M DATASET ======================= # 
# # =================================================================== #

# import os
# import requests
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm

# # --- CONFIGURATION ---
# # Base URL for Cohere 10M dataset
# BASE_URL = "https://assets.zilliz.com/benchmark/cohere_large_10m/"

# # Your SSD Path (Updated based on your terminal output)
# LOCAL_DIR = "/home/admin/vectordataset/cohere/cohere_large_10m"

# # Number of parallel downloads
# MAX_WORKERS = 10

# # --- FILE LIST GENERATION ---
# files = [
#     "test.parquet", 
#     "neighbors.parquet", 
#     "scalar_labels.parquet" # Included based on your ls output
# ]

# # Generate the 10 training parts (00 to 09)
# # Note: Cohere 10M uses the prefix 'shuffle_train-' and has 10 parts
# for i in range(10):
#     files.append(f"shuffle_train-{i:02d}-of-10.parquet")

# def download_file(filename):
#     local_path = os.path.join(LOCAL_DIR, filename)
#     url = BASE_URL + filename
    
#     # 1. Skip if fully downloaded (File > 1KB)
#     if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
#         return f"✅ Skipped (Exists): {filename}"

#     try:
#         # 2. Stream download
#         with requests.get(url, stream=True, timeout=60) as r:
#             if r.status_code == 404:
#                 return f"❌ Missing (404): {filename}"
            
#             r.raise_for_status()
            
#             # Write to file
#             with open(local_path, 'wb') as f:
#                 for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
#                     f.write(chunk)
                  
#         return f"⬇️ Downloaded: {filename}"
        
#     except Exception as e:
#         return f"⚠️ Error {filename}: {str(e)}"

# if __name__ == "__main__":
#     # Ensure directory exists
#     os.makedirs(LOCAL_DIR, exist_ok=True)
    
#     print(f"--- Cohere 10M Downloader ---")
#     print(f"Source: {BASE_URL}")
#     print(f"Destination: {LOCAL_DIR}")
#     print(f"Files to check: {len(files)}")
#     print(f"Parallel threads: {MAX_WORKERS}\n")
    
#     # Start Parallel Download
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         # We use tqdm to show a progress bar for the *number of files* completed
#         results = list(tqdm(executor.map(download_file, files), total=len(files), unit="file"))

#     # Print summary of missing or failed files
#     failures = [r for r in results if "Error" in r or "Missing" in r]
#     if failures:
#         print("\nSummary of Issues:")
#         for fail in failures:
#             print(fail)
#     else:
#         print("\n🎉 All files processed successfully!")






# # =================================================================== #
# # ==================== FOR LAION 100M DATASET ======================= # 
# # =================================================================== #

# import os
# import requests
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm

# # --- CONFIGURATION ---
# # Base URL for Laion 100M dataset
# BASE_URL = "https://assets.zilliz.com/benchmark/laion_large_100m/"

# # Your SSD Path (Updated based on your terminal output)
# LOCAL_DIR = "/home/admin/vectordataset/laion/laion_large_100m"

# # Number of parallel downloads
# MAX_WORKERS = 10

# # --- FILE LIST GENERATION ---
# files = [
#     "test.parquet",
#     "neighbors.parquet",
#     # "scalar_labels.parquet"  # Uncomment if needed (often missing for 100M)
# ]

# # Generate the 100 training parts (00 to 99)
# for i in range(100):
#     files.append(f"train-{i:02d}-of-100.parquet")


# def download_file(filename):
#     local_path = os.path.join(LOCAL_DIR, filename)
#     url = BASE_URL + filename

#     # 1. Skip if fully downloaded (File > 1KB)
#     if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
#         return f"✅ Skipped (Exists): {filename}"

#     try:
#         # 2. Stream download
#         with requests.get(url, stream=True, timeout=60) as r:
#             if r.status_code == 404:
#                 return f"❌ Missing (404): {filename}"

#             r.raise_for_status()

#             # Write to file
#             with open(local_path, 'wb') as f:
#                 for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
#                     f.write(chunk)

#         return f"⬇️ Downloaded: {filename}"

#     except Exception as e:
#         return f"⚠️ Error {filename}: {str(e)}"


# if __name__ == "__main__":
#     # Ensure directory exists
#     os.makedirs(LOCAL_DIR, exist_ok=True)

#     print(f"--- LAION 100M Downloader ---")
#     print(f"Source: {BASE_URL}")
#     print(f"Destination: {LOCAL_DIR}")
#     print(f"Files to check: {len(files)}")
#     print(f"Parallel threads: {MAX_WORKERS}\n")

#     # Start Parallel Download
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         # We use tqdm to show a progress bar for the *number of files* completed
#         results = list(
#             tqdm(
#                 executor.map(download_file, files),
#                 total=len(files),
#                 unit="file"
#             )
#         )

#     # Print summary of missing or failed files
#     failures = [r for r in results if "Error" in r or "Missing" in r]
#     if failures:
#         print("\nSummary of Issues:")
#         for fail in failures:
#             print(fail)
#     else:
#         print("\n🎉 All files processed successfully!")
