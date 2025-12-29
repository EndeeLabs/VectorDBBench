# =================================================================== #
# ==================== FOR COHERE 10M DATASET ======================= # 
# =================================================================== #

import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- CONFIGURATION ---
# Base URL for Cohere 10M dataset
BASE_URL = "https://assets.zilliz.com/benchmark/cohere_large_10m/"

# Your SSD Path (Updated based on your terminal output)
LOCAL_DIR = "/home/admin/vectordataset/cohere/cohere_large_10m"

# Number of parallel downloads
MAX_WORKERS = 10

# --- FILE LIST GENERATION ---
files = [
    "test.parquet", 
    "neighbors.parquet", 
    "scalar_labels.parquet" # Included based on your ls output
]

# Generate the 10 training parts (00 to 09)
# Note: Cohere 10M uses the prefix 'shuffle_train-' and has 10 parts
for i in range(10):
    files.append(f"shuffle_train-{i:02d}-of-10.parquet")

def download_file(filename):
    local_path = os.path.join(LOCAL_DIR, filename)
    url = BASE_URL + filename
    
    # 1. Skip if fully downloaded (File > 1KB)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
        return f"✅ Skipped (Exists): {filename}"

    try:
        # 2. Stream download
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code == 404:
                return f"❌ Missing (404): {filename}"
            
            r.raise_for_status()
            
            # Write to file
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                    f.write(chunk)
                  
        return f"⬇️ Downloaded: {filename}"
        
    except Exception as e:
        return f"⚠️ Error {filename}: {str(e)}"

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    print(f"--- Cohere 10M Downloader ---")
    print(f"Source: {BASE_URL}")
    print(f"Destination: {LOCAL_DIR}")
    print(f"Files to check: {len(files)}")
    print(f"Parallel threads: {MAX_WORKERS}\n")
    
    # Start Parallel Download
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # We use tqdm to show a progress bar for the *number of files* completed
        results = list(tqdm(executor.map(download_file, files), total=len(files), unit="file"))

    # Print summary of missing or failed files
    failures = [r for r in results if "Error" in r or "Missing" in r]
    if failures:
        print("\nSummary of Issues:")
        for fail in failures:
            print(fail)
    else:
        print("\n🎉 All files processed successfully!")










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
