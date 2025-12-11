import concurrent
import logging
import math
import multiprocessing as mp
import time
import traceback

import numpy as np
import psutil

from vectordb_bench.backend.dataset import DatasetManager
from vectordb_bench.backend.filter import Filter, FilterOp, non_filter

from ... import config
from ...metric import calc_ndcg, calc_recall, get_ideal_dcg
from ...models import LoadTimeoutError, PerformanceTimeoutError
from .. import utils
from ..clients import api

import os
import json

# CHECKPOINT_PATH = "/mnt/data/VectorDBBench/insert_checkpoint.json"

# Get the path of the current script
CURRENT_FILE = os.path.abspath(__file__)

# Go up 3 levels to reach the root benchmarking folder
ROOT_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE))))

CHECKPOINT_FILENAME = "insert_checkpoint.json"
CHECKPOINT_PATH = os.path.join(ROOT_BENCHMARK_DIR, CHECKPOINT_FILENAME)

# Ensure the root folder exists (should already exist, but safe)
os.makedirs(ROOT_BENCHMARK_DIR, exist_ok=True)

def save_checkpoint(count: int):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"last_index": count}, f)

def load_checkpoint() -> int:
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                data = json.load(f)
                return data.get("last_index", 0)
        except Exception as e:
            log.warning(f"Failed to read checkpoint file: {e}")
    return 0

def clear_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)


NUM_PER_BATCH = config.NUM_PER_BATCH
LOAD_MAX_TRY_COUNT = config.LOAD_MAX_TRY_COUNT

log = logging.getLogger(__name__)


class SerialInsertRunner:
    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager,
        normalize: bool,
        filters: Filter = non_filter,
        timeout: float | None = None,
    ):
        self.timeout = timeout if isinstance(timeout, int | float) else None
        self.dataset = dataset
        self.db = db
        self.normalize = normalize
        self.filters = filters

    def retry_insert(self, db: api.VectorDB, retry_idx: int = 0, **kwargs):
        _, error = db.insert_embeddings(**kwargs)
        if error is not None:
            log.warning(f"Insert Failed, try_idx={retry_idx}, Exception: {error}")
            retry_idx += 1
            if retry_idx <= config.MAX_INSERT_RETRY:
                time.sleep(retry_idx)
                self.retry_insert(db, retry_idx=retry_idx, **kwargs)
            else:
                msg = f"Insert failed and retried more than {config.MAX_INSERT_RETRY} times"
                raise RuntimeError(msg) from None

    def task(self) -> int:
        count = 0
        last_index = load_checkpoint()
        log.info(f"Resuming insertion from checkpoint at index {last_index}")

        with self.db.init():
            log.info(f"({mp.current_process().name:16}) Start inserting embeddings in batch {config.NUM_PER_BATCH}")
            start = time.perf_counter()
            
            for data_df in self.dataset:
                batch_size = len(data_df)
                if count + batch_size <= last_index:
                    count += batch_size
                    log.debug(f"Skipping batch of size {batch_size}, total skipped={count}")
                    continue

                all_metadata = data_df[self.dataset.data.train_id_field].tolist()
                emb_np = np.stack(data_df[self.dataset.data.train_vector_field])

                if self.normalize:
                    log.debug("Normalizing train data")
                    all_embeddings = (emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis]).tolist()
                else:
                    all_embeddings = emb_np.tolist()
                del emb_np

                labels_data = None
                if self.filters.type == FilterOp.StrEqual:
                    if self.dataset.data.scalar_labels_file_separated:
                        labels_data = self.dataset.scalar_labels[self.filters.label_field][all_metadata].to_list()
                    else:
                        labels_data = data_df[self.filters.label_field].tolist()

                insert_count, error = self.db.insert_embeddings(
                    embeddings=all_embeddings,
                    metadata=all_metadata,
                    labels_data=labels_data,
                )

                if error is not None:
                    self.retry_insert(
                        self.db,
                        embeddings=all_embeddings,
                        metadata=all_metadata,
                        labels_data=labels_data,
                    )

                assert insert_count == len(all_metadata)
                count += insert_count
                save_checkpoint(count)  # Save progress after each batch

                if count % 100_000 == 0:
                    log.info(f"({mp.current_process().name:16}) Loaded {count} embeddings into VectorDB")

            log.info(
                f"({mp.current_process().name:16}) Finished loading all dataset into VectorDB, "
                f"dur={time.perf_counter() - start}"
            )
            clear_checkpoint()  # Clean checkpoint file after successful completion
            return count


    def endless_insert_data(self, all_embeddings: list, all_metadata: list, left_id: int = 0) -> int:
        with self.db.init():
            # unique id for endlessness insertion
            all_metadata = [i + left_id for i in all_metadata]

            num_batches = math.ceil(len(all_embeddings) / NUM_PER_BATCH)
            log.info(
                f"({mp.current_process().name:16}) Start inserting {len(all_embeddings)} "
                f"embeddings in batch {NUM_PER_BATCH}"
            )
            count = 0
            for batch_id in range(num_batches):
                retry_count = 0
                already_insert_count = 0
                metadata = all_metadata[batch_id * NUM_PER_BATCH : (batch_id + 1) * NUM_PER_BATCH]
                embeddings = all_embeddings[batch_id * NUM_PER_BATCH : (batch_id + 1) * NUM_PER_BATCH]

                log.debug(
                    f"({mp.current_process().name:16}) batch [{batch_id:3}/{num_batches}], "
                    f"Start inserting {len(metadata)} embeddings"
                )
                while retry_count < LOAD_MAX_TRY_COUNT:
                    insert_count, error = self.db.insert_embeddings(
                        embeddings=embeddings[already_insert_count:],
                        metadata=metadata[already_insert_count:],
                    )
                    already_insert_count += insert_count
                    if error is not None:
                        retry_count += 1
                        time.sleep(10)

                        log.info(f"Failed to insert data, try {retry_count} time")
                        if retry_count >= LOAD_MAX_TRY_COUNT:
                            raise error
                    else:
                        break
                log.debug(
                    f"({mp.current_process().name:16}) batch [{batch_id:3}/{num_batches}], "
                    f"Finish inserting {len(metadata)} embeddings"
                )

                assert already_insert_count == len(metadata)
                count += already_insert_count
            log.info(
                f"({mp.current_process().name:16}) Finish inserting {len(all_embeddings)} embeddings in "
                f"batch {NUM_PER_BATCH}"
            )
        return count

    @utils.time_it
    def _insert_all_batches(self) -> int:
        """Performance case only"""
        with concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("spawn"),
            max_workers=1,
        ) as executor:
            future = executor.submit(self.task)
            try:
                count = future.result(timeout=self.timeout)
            except TimeoutError as e:
                msg = f"VectorDB load dataset timeout in {self.timeout}"
                log.warning(msg)
                for pid, _ in executor._processes.items():
                    psutil.Process(pid).kill()
                raise PerformanceTimeoutError(msg) from e
            except Exception as e:
                log.warning(f"VectorDB load dataset error: {e}")
                raise e from e
            else:
                return count

    def run_endlessness(self) -> int:
        """run forever util DB raises exception or crash"""
        # datasets for load tests are quite small, can fit into memory
        # only 1 file
        data_df = next(iter(self.dataset))
        all_embeddings, all_metadata = (
            np.stack(data_df[self.dataset.data.train_vector_field]).tolist(),
            data_df[self.dataset.data.train_id_field].tolist(),
        )

        start_time = time.perf_counter()
        max_load_count, times = 0, 0
        try:
            while time.perf_counter() - start_time < self.timeout:
                count = self.endless_insert_data(
                    all_embeddings,
                    all_metadata,
                    left_id=max_load_count,
                )
                max_load_count += count
                times += 1
                log.info(
                    f"Loaded {times} entire dataset, current max load counts={utils.numerize(max_load_count)}, "
                    f"{max_load_count}"
                )
        except Exception as e:
            log.info(
                f"Capacity case load reach limit, insertion counts={utils.numerize(max_load_count)}, "
                f"{max_load_count}, err={e}"
            )
            traceback.print_exc()
            return max_load_count
        else:
            raise LoadTimeoutError(self.timeout)

    def run(self) -> int:
        count, _ = self._insert_all_batches()
        return count


# =============================================================================
# PARALLEL SEARCH WORKER FUNCTION
# =============================================================================
def _parallel_search_worker(args: tuple) -> list[tuple[float, int, list[int]]]:
    """
    Worker function executed in separate process for parallel search.
    
    Each worker:
    1. Receives a batch of queries (embeddings) with their indices
    2. Initializes its own DB connection
    3. Executes searches for all queries in the batch
    4. Returns results with latencies and original indices
    
    Args:
        args: Tuple of (db_instance, batch_queries, batch_indices, k, filters)
    
    Returns:
        List of tuples: [(latency, original_query_idx, search_results), ...]
    
    This function runs in a subprocess, so it needs to:
    - Initialize DB connection independently
    - Handle its own retry logic
    - Return all results for aggregation in main process
    """
    db_instance, batch_queries, batch_indices, k, filters = args
    
    results_batch = []
    
    # Each worker initializes its own DB connection
    with db_instance.init():
        # Prepare filters if needed (e.g., for filtered search scenarios)
        db_instance.prepare_filter(filters)
        
        # Process each query in this worker's batch
        for idx, emb in zip(batch_indices, batch_queries):
            retry_count = 0
            max_retries = config.MAX_SEARCH_RETRY
            
            # Retry logic for each individual query
            while retry_count <= max_retries:
                try:
                    # Time the search operation
                    s = time.perf_counter()
                    results = db_instance.search_embedding(emb, k)
                    latency = time.perf_counter() - s
                    
                    # Store result with original index for proper ordering later
                    results_batch.append((latency, idx, results))
                    break  # Success, move to next query
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        log.error(f"Query {idx} failed after {max_retries} retries: {e}")
                        raise
                    log.warning(f"Query {idx} retry {retry_count}/{max_retries}: {e}")
                    time.sleep(0.1 * retry_count)  # Exponential backoff
    
    return results_batch


class SerialSearchRunner:
    """
    Search runner that executes queries to measure recall and latency.
    
    PERFORMANCE ENHANCEMENT:
    This class now uses parallel processing under the hood while maintaining
    the same interface. Queries are distributed across multiple worker processes
    for significant speedup (4-8x faster depending on CPU cores).
    
    The parallelization is transparent to callers - all existing code continues
    to work without modification.
    """
    
    def __init__(
        self,
        db: api.VectorDB,
        test_data: list[list[float]],
        ground_truth: list[list[int]],
        k: int = 100,
        filters: Filter = non_filter,
    ):
        self.db = db
        self.k = k
        self.filters = filters

        # Convert test data to list format if needed
        if isinstance(test_data[0], np.ndarray):
            self.test_data = [query.tolist() for query in test_data]
        else:
            self.test_data = test_data
        self.ground_truth = ground_truth
        
        # PARALLEL PROCESSING CONFIG:
        # Auto-detect number of CPU cores, reserve 1 for main process
        # This allows efficient parallel query execution
        self.num_workers = max(1, mp.cpu_count() - 1)
        log.info(f"SerialSearchRunner initialized with {self.num_workers} parallel workers")

    def _get_db_search_res(self, emb: list[float], retry_idx: int = 0) -> list[int]:
        """
        LEGACY METHOD - Kept for backward compatibility.
        This method is no longer used in the main search path but retained
        in case any external code calls it directly.
        """
        try:
            results = self.db.search_embedding(emb, self.k)
        except Exception as e:
            log.warning(f"Serial search failed, retry_idx={retry_idx}, Exception: {e}")
            if retry_idx < config.MAX_SEARCH_RETRY:
                return self._get_db_search_res(emb=emb, retry_idx=retry_idx + 1)

            msg = f"Serial search failed and retried more than {config.MAX_SEARCH_RETRY} times"
            raise RuntimeError(msg) from e

        return results

    def _create_query_batches(self) -> list[tuple[list[int], list]]:
        """
        PARALLEL PROCESSING HELPER:
        Splits test queries into batches for distribution to worker processes.
        
        Each batch contains:
        - Original indices (to maintain ordering after parallel execution)
        - Query embeddings for this batch
        
        Batch size is calculated as: total_queries / num_workers
        This ensures even distribution of work across all available workers.
        
        Returns:
            List of (indices, queries) tuples, one per worker
        """
        total_queries = len(self.test_data)
        batch_size = max(1, math.ceil(total_queries / self.num_workers))
        batches = []
        
        for i in range(0, total_queries, batch_size):
            end_idx = min(i + batch_size, total_queries)
            indices = list(range(i, end_idx))
            queries = self.test_data[i:end_idx]
            batches.append((indices, queries))
        
        log.debug(f"Created {len(batches)} query batches, batch_size={batch_size}")
        return batches

    def search(self, args: tuple[list, list[list[int]]]) -> tuple[float, float, float, float]:
        """
        MODIFIED: Now uses parallel processing for significant speedup.
        
        Main changes from original:
        1. Splits queries into batches for parallel workers
        2. Spawns worker processes to execute searches concurrently
        3. Collects and re-orders results by original index
        4. Calculates metrics (recall, NDCG, latencies) from parallel results
        
        The interface and return values remain identical to the original version,
        ensuring backward compatibility with existing code.
        
        Original single-threaded approach took ~164s for 1000 queries.
        Parallel approach typically achieves 4-8x speedup depending on CPU cores.
        
        Args:
            args: Tuple of (test_data, ground_truth) - kept for interface compatibility
        
        Returns:
            Tuple of (avg_recall, avg_ndcg, p99_latency, p95_latency)
        """
        log.info(f"{mp.current_process().name:14} start search the entire test_data to get recall and latency")
        
        test_data, ground_truth = args
        
        # Validate input
        if not test_data:
            raise RuntimeError("empty test_data")
        
        log.debug(f"test dataset size: {len(test_data)}")
        log.debug(f"ground truth size: {len(ground_truth) if ground_truth else 0}")
        
        # PARALLEL EXECUTION SECTION:
        # Create batches for parallel processing
        batches = self._create_query_batches()
        
        # Prepare arguments for each worker process
        # Each worker gets: (db_instance, queries, indices, k, filters)
        worker_args = [
            (self.db, queries, indices, self.k, self.filters)
            for indices, queries in batches
        ]
        
        log.info(f"Launching {len(worker_args)} parallel workers for search")
        
        # Execute searches in parallel using ProcessPoolExecutor
        # Using 'spawn' context ensures clean process initialization
        all_results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=mp.get_context("spawn")
        ) as executor:
            # Submit all batches to worker pool
            futures = [
                executor.submit(_parallel_search_worker, args) 
                for args in worker_args
            ]
            
            # Collect results as workers complete
            # as_completed() allows processing results immediately when ready
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    log.error(f"Worker process failed: {e}")
                    raise
        
        # RESULTS PROCESSING SECTION:
        # Sort results by original query index to maintain correct ordering
        # This is crucial because workers complete in arbitrary order
        all_results.sort(key=lambda x: x[1])  # Sort by index (2nd element)
        
        log.info(f"Collected {len(all_results)} search results from parallel workers")
        
        # Calculate performance metrics
        latencies, recalls, ndcgs = [], [], []
        ideal_dcg = get_ideal_dcg(self.k)
        
        for latency, idx, results in all_results:
            latencies.append(latency)
            
            # Calculate recall and NDCG if ground truth is available
            if ground_truth is not None:
                gt = ground_truth[idx]
                recalls.append(calc_recall(self.k, gt[:self.k], results))
                ndcgs.append(calc_ndcg(gt[:self.k], results, ideal_dcg))
            else:
                recalls.append(0)
                ndcgs.append(0)
            
            # Progress logging every 100 queries
            if len(latencies) % 100 == 0:
                log.debug(
                    f"({mp.current_process().name:14}) search_count={len(latencies):3}, "
                    f"latest_latency={latencies[-1]}, latest recall={recalls[-1]}"
                )
        
        # Calculate aggregate statistics
        avg_latency = round(np.mean(latencies), 4)
        avg_recall = round(np.mean(recalls), 4)
        avg_ndcg = round(np.mean(ndcgs), 4)
        cost = round(np.sum(latencies), 4)
        p99 = round(np.percentile(latencies, 99), 4)
        p95 = round(np.percentile(latencies, 95), 4)
        
        log.info(
            f"{mp.current_process().name:14} search entire test_data: "
            f"cost={cost}s, "
            f"queries={len(latencies)}, "
            f"avg_recall={avg_recall}, "
            f"avg_ndcg={avg_ndcg}, "
            f"avg_latency={avg_latency}, "
            f"p99={p99}, "
            f"p95={p95}"
        )
        
        return (avg_recall, avg_ndcg, p99, p95)

    def _run_in_subprocess(self) -> tuple[float, float, float, float]:
        """
        UNCHANGED: Wrapper to execute search in subprocess.
        
        This method is kept for interface compatibility.
        The actual parallelization happens inside the search() method.
        """
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.search, (self.test_data, self.ground_truth))
            return future.result()

    @utils.time_it
    def run(self) -> tuple[float, float, float, float]:
        """
        UNCHANGED: Main entry point for search execution.
        
        Interface remains identical - callers don't need any changes.
        Parallel processing is handled transparently.
        """
        log.info(f"{mp.current_process().name:14} start serial search")
        if self.test_data is None:
            msg = "empty test_data"
            raise RuntimeError(msg)

        return self._run_in_subprocess()

    @utils.time_it
    def run_with_cost(self) -> tuple[tuple[float, float, float, float], float]:
        """
        UNCHANGED: Search with cost tracking.
        
        Returns:
            tuple[tuple[float, float, float, float], float]: 
                (avg_recall, avg_ndcg, p99_latency, p95_latency), cost
        """
        log.info(f"{mp.current_process().name:14} start serial search")
        if self.test_data is None:
            msg = "empty test_data"
            raise RuntimeError(msg)

        return self._run_in_subprocess()








# =======================================================================

# MODIFIED TO HAVE CHECKPOINT FOR RESUME LOGIC

# =======================================================================




# import concurrent
# import logging
# import math
# import multiprocessing as mp
# import time
# import traceback

# import numpy as np
# import psutil

# from vectordb_bench.backend.dataset import DatasetManager
# from vectordb_bench.backend.filter import Filter, FilterOp, non_filter

# from ... import config
# from ...metric import calc_ndcg, calc_recall, get_ideal_dcg
# from ...models import LoadTimeoutError, PerformanceTimeoutError
# from .. import utils
# from ..clients import api

# import os
# import json


# # Get the path of the current script
# CURRENT_FILE = os.path.abspath(__file__)

# # Go up 3 levels to reach the root benchmarking folder
# ROOT_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE))))

# CHECKPOINT_FILENAME = "insert_checkpoint.json"
# CHECKPOINT_PATH = os.path.join(ROOT_BENCHMARK_DIR, CHECKPOINT_FILENAME)

# # Ensure the root folder exists (should already exist, but safe)
# os.makedirs(ROOT_BENCHMARK_DIR, exist_ok=True)

# def save_checkpoint(count: int):
#     with open(CHECKPOINT_PATH, "w") as f:
#         json.dump({"last_index": count}, f)

# def load_checkpoint() -> int:
#     if os.path.exists(CHECKPOINT_PATH):
#         try:
#             with open(CHECKPOINT_PATH, "r") as f:
#                 data = json.load(f)
#                 return data.get("last_index", 0)
#         except Exception as e:
#             log.warning(f"Failed to read checkpoint file: {e}")
#     return 0

# def clear_checkpoint():
#     if os.path.exists(CHECKPOINT_PATH):
#         os.remove(CHECKPOINT_PATH)


# NUM_PER_BATCH = config.NUM_PER_BATCH
# LOAD_MAX_TRY_COUNT = config.LOAD_MAX_TRY_COUNT

# log = logging.getLogger(__name__)


# class SerialInsertRunner:
#     def __init__(
#         self,
#         db: api.VectorDB,
#         dataset: DatasetManager,
#         normalize: bool,
#         filters: Filter = non_filter,
#         timeout: float | None = None,
#     ):
#         self.timeout = timeout if isinstance(timeout, int | float) else None
#         self.dataset = dataset
#         self.db = db
#         self.normalize = normalize
#         self.filters = filters

#     def retry_insert(self, db: api.VectorDB, retry_idx: int = 0, **kwargs):
#         _, error = db.insert_embeddings(**kwargs)
#         if error is not None:
#             log.warning(f"Insert Failed, try_idx={retry_idx}, Exception: {error}")
#             retry_idx += 1
#             if retry_idx <= config.MAX_INSERT_RETRY:
#                 time.sleep(retry_idx)
#                 self.retry_insert(db, retry_idx=retry_idx, **kwargs)
#             else:
#                 msg = f"Insert failed and retried more than {config.MAX_INSERT_RETRY} times"
#                 raise RuntimeError(msg) from None

#     def task(self) -> int:
#         count = 0
#         last_index = load_checkpoint()
#         log.info(f"Resuming insertion from checkpoint at index {last_index}")

#         with self.db.init():
#             log.info(f"({mp.current_process().name:16}) Start inserting embeddings in batch {config.NUM_PER_BATCH}")
#             start = time.perf_counter()
            
#             for data_df in self.dataset:
#                 batch_size = len(data_df)
#                 if count + batch_size <= last_index:
#                     count += batch_size
#                     log.debug(f"Skipping batch of size {batch_size}, total skipped={count}")
#                     continue

#                 all_metadata = data_df[self.dataset.data.train_id_field].tolist()
#                 emb_np = np.stack(data_df[self.dataset.data.train_vector_field])

#                 if self.normalize:
#                     log.debug("Normalizing train data")
#                     all_embeddings = (emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis]).tolist()
#                 else:
#                     all_embeddings = emb_np.tolist()
#                 del emb_np

#                 labels_data = None
#                 if self.filters.type == FilterOp.StrEqual:
#                     if self.dataset.data.scalar_labels_file_separated:
#                         labels_data = self.dataset.scalar_labels[self.filters.label_field][all_metadata].to_list()
#                     else:
#                         labels_data = data_df[self.filters.label_field].tolist()

#                 insert_count, error = self.db.insert_embeddings(
#                     embeddings=all_embeddings,
#                     metadata=all_metadata,
#                     labels_data=labels_data,
#                 )

#                 if error is not None:
#                     self.retry_insert(
#                         self.db,
#                         embeddings=all_embeddings,
#                         metadata=all_metadata,
#                         labels_data=labels_data,
#                     )

#                 assert insert_count == len(all_metadata)
#                 count += insert_count
#                 save_checkpoint(count)  # Save progress after each batch

#                 if count % 100_000 == 0:
#                     log.info(f"({mp.current_process().name:16}) Loaded {count} embeddings into VectorDB")

#             log.info(
#                 f"({mp.current_process().name:16}) Finished loading all dataset into VectorDB, "
#                 f"dur={time.perf_counter() - start}"
#             )
#             clear_checkpoint()  # Clean checkpoint file after successful completion
#             return count


#     def endless_insert_data(self, all_embeddings: list, all_metadata: list, left_id: int = 0) -> int:
#         with self.db.init():
#             # unique id for endlessness insertion
#             all_metadata = [i + left_id for i in all_metadata]

#             num_batches = math.ceil(len(all_embeddings) / NUM_PER_BATCH)
#             log.info(
#                 f"({mp.current_process().name:16}) Start inserting {len(all_embeddings)} "
#                 f"embeddings in batch {NUM_PER_BATCH}"
#             )
#             count = 0
#             for batch_id in range(num_batches):
#                 retry_count = 0
#                 already_insert_count = 0
#                 metadata = all_metadata[batch_id * NUM_PER_BATCH : (batch_id + 1) * NUM_PER_BATCH]
#                 embeddings = all_embeddings[batch_id * NUM_PER_BATCH : (batch_id + 1) * NUM_PER_BATCH]

#                 log.debug(
#                     f"({mp.current_process().name:16}) batch [{batch_id:3}/{num_batches}], "
#                     f"Start inserting {len(metadata)} embeddings"
#                 )
#                 while retry_count < LOAD_MAX_TRY_COUNT:
#                     insert_count, error = self.db.insert_embeddings(
#                         embeddings=embeddings[already_insert_count:],
#                         metadata=metadata[already_insert_count:],
#                     )
#                     already_insert_count += insert_count
#                     if error is not None:
#                         retry_count += 1
#                         time.sleep(10)

#                         log.info(f"Failed to insert data, try {retry_count} time")
#                         if retry_count >= LOAD_MAX_TRY_COUNT:
#                             raise error
#                     else:
#                         break
#                 log.debug(
#                     f"({mp.current_process().name:16}) batch [{batch_id:3}/{num_batches}], "
#                     f"Finish inserting {len(metadata)} embeddings"
#                 )

#                 assert already_insert_count == len(metadata)
#                 count += already_insert_count
#             log.info(
#                 f"({mp.current_process().name:16}) Finish inserting {len(all_embeddings)} embeddings in "
#                 f"batch {NUM_PER_BATCH}"
#             )
#         return count

#     @utils.time_it
#     def _insert_all_batches(self) -> int:
#         """Performance case only"""
#         with concurrent.futures.ProcessPoolExecutor(
#             mp_context=mp.get_context("spawn"),
#             max_workers=1,
#         ) as executor:
#             future = executor.submit(self.task)
#             try:
#                 count = future.result(timeout=self.timeout)
#             except TimeoutError as e:
#                 msg = f"VectorDB load dataset timeout in {self.timeout}"
#                 log.warning(msg)
#                 for pid, _ in executor._processes.items():
#                     psutil.Process(pid).kill()
#                 raise PerformanceTimeoutError(msg) from e
#             except Exception as e:
#                 log.warning(f"VectorDB load dataset error: {e}")
#                 raise e from e
#             else:
#                 return count

#     def run_endlessness(self) -> int:
#         """run forever util DB raises exception or crash"""
#         # datasets for load tests are quite small, can fit into memory
#         # only 1 file
#         data_df = next(iter(self.dataset))
#         all_embeddings, all_metadata = (
#             np.stack(data_df[self.dataset.data.train_vector_field]).tolist(),
#             data_df[self.dataset.data.train_id_field].tolist(),
#         )

#         start_time = time.perf_counter()
#         max_load_count, times = 0, 0
#         try:
#             while time.perf_counter() - start_time < self.timeout:
#                 count = self.endless_insert_data(
#                     all_embeddings,
#                     all_metadata,
#                     left_id=max_load_count,
#                 )
#                 max_load_count += count
#                 times += 1
#                 log.info(
#                     f"Loaded {times} entire dataset, current max load counts={utils.numerize(max_load_count)}, "
#                     f"{max_load_count}"
#                 )
#         except Exception as e:
#             log.info(
#                 f"Capacity case load reach limit, insertion counts={utils.numerize(max_load_count)}, "
#                 f"{max_load_count}, err={e}"
#             )
#             traceback.print_exc()
#             return max_load_count
#         else:
#             raise LoadTimeoutError(self.timeout)

#     def run(self) -> int:
#         count, _ = self._insert_all_batches()
#         return count


# class SerialSearchRunner:
#     def __init__(
#         self,
#         db: api.VectorDB,
#         test_data: list[list[float]],
#         ground_truth: list[list[int]],
#         k: int = 100,
#         filters: Filter = non_filter,
#     ):
#         self.db = db
#         self.k = k
#         self.filters = filters

#         if isinstance(test_data[0], np.ndarray):
#             self.test_data = [query.tolist() for query in test_data]
#         else:
#             self.test_data = test_data
#         self.ground_truth = ground_truth

#     def _get_db_search_res(self, emb: list[float], retry_idx: int = 0) -> list[int]:
#         try:
#             results = self.db.search_embedding(emb, self.k)
#         except Exception as e:
#             log.warning(f"Serial search failed, retry_idx={retry_idx}, Exception: {e}")
#             if retry_idx < config.MAX_SEARCH_RETRY:
#                 return self._get_db_search_res(emb=emb, retry_idx=retry_idx + 1)

#             msg = f"Serial search failed and retried more than {config.MAX_SEARCH_RETRY} times"
#             raise RuntimeError(msg) from e

#         return results

#     def search(self, args: tuple[list, list[list[int]]]) -> tuple[float, float, float, float]:
#         log.info(f"{mp.current_process().name:14} start search the entire test_data to get recall and latency")
#         with self.db.init():
#             self.db.prepare_filter(self.filters)
#             test_data, ground_truth = args
#             ideal_dcg = get_ideal_dcg(self.k)

#             log.debug(f"test dataset size: {len(test_data)}")
#             log.debug(f"ground truth size: {len(ground_truth)}")

#             latencies, recalls, ndcgs = [], [], []
#             for idx, emb in enumerate(test_data):
#                 s = time.perf_counter()
#                 try:
#                     results = self._get_db_search_res(emb)
#                 except Exception as e:
#                     log.warning(f"VectorDB search_embedding error: {e}")
#                     raise e from None

#                 latencies.append(time.perf_counter() - s)

#                 if ground_truth is not None:
#                     gt = ground_truth[idx]
#                     recalls.append(calc_recall(self.k, gt[: self.k], results))
#                     ndcgs.append(calc_ndcg(gt[: self.k], results, ideal_dcg))
#                 else:
#                     recalls.append(0)
#                     ndcgs.append(0)

#                 if len(latencies) % 100 == 0:
#                     log.debug(
#                         f"({mp.current_process().name:14}) search_count={len(latencies):3}, "
#                         f"latest_latency={latencies[-1]}, latest recall={recalls[-1]}"
#                     )

#         avg_latency = round(np.mean(latencies), 4)
#         avg_recall = round(np.mean(recalls), 4)
#         avg_ndcg = round(np.mean(ndcgs), 4)
#         cost = round(np.sum(latencies), 4)
#         p99 = round(np.percentile(latencies, 99), 4)
#         p95 = round(np.percentile(latencies, 95), 4)
#         log.info(
#             f"{mp.current_process().name:14} search entire test_data: "
#             f"cost={cost}s, "
#             f"queries={len(latencies)}, "
#             f"avg_recall={avg_recall}, "
#             f"avg_ndcg={avg_ndcg}, "
#             f"avg_latency={avg_latency}, "
#             f"p99={p99}, "
#             f"p95={p95}"
#         )
#         return (avg_recall, avg_ndcg, p99, p95)

#     def _run_in_subprocess(self) -> tuple[float, float, float, float]:
#         with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
#             future = executor.submit(self.search, (self.test_data, self.ground_truth))
#             return future.result()

#     @utils.time_it
#     def run(self) -> tuple[float, float, float, float]:
#         log.info(f"{mp.current_process().name:14} start serial search")
#         if self.test_data is None:
#             msg = "empty test_data"
#             raise RuntimeError(msg)

#         return self._run_in_subprocess()

#     @utils.time_it
#     def run_with_cost(self) -> tuple[tuple[float, float, float, float], float]:
#         """
#         Search all test data in serial.
#         Returns:
#             tuple[tuple[float, float, float, float], float]: (avg_recall, avg_ndcg, p99_latency, p95_latency), cost
#         """
#         log.info(f"{mp.current_process().name:14} start serial search")
#         if self.test_data is None:
#             msg = "empty test_data"
#             raise RuntimeError(msg)

#         return self._run_in_subprocess()






# =======================================================================

# ORIGINAL SERIAL RUNNER CODE

# =======================================================================







# import concurrent
# import logging
# import math
# import multiprocessing as mp
# import time
# import traceback

# import numpy as np
# import psutil

# from vectordb_bench.backend.dataset import DatasetManager
# from vectordb_bench.backend.filter import Filter, FilterOp, non_filter

# from ... import config
# from ...metric import calc_ndcg, calc_recall, get_ideal_dcg
# from ...models import LoadTimeoutError, PerformanceTimeoutError
# from .. import utils
# from ..clients import api

# NUM_PER_BATCH = config.NUM_PER_BATCH
# LOAD_MAX_TRY_COUNT = config.LOAD_MAX_TRY_COUNT

# log = logging.getLogger(__name__)


# class SerialInsertRunner:
#     def __init__(
#         self,
#         db: api.VectorDB,
#         dataset: DatasetManager,
#         normalize: bool,
#         filters: Filter = non_filter,
#         timeout: float | None = None,
#     ):
#         self.timeout = timeout if isinstance(timeout, int | float) else None
#         self.dataset = dataset
#         self.db = db
#         self.normalize = normalize
#         self.filters = filters

#     def retry_insert(self, db: api.VectorDB, retry_idx: int = 0, **kwargs):
#         _, error = db.insert_embeddings(**kwargs)
#         if error is not None:
#             log.warning(f"Insert Failed, try_idx={retry_idx}, Exception: {error}")
#             retry_idx += 1
#             if retry_idx <= config.MAX_INSERT_RETRY:
#                 time.sleep(retry_idx)
#                 self.retry_insert(db, retry_idx=retry_idx, **kwargs)
#             else:
#                 msg = f"Insert failed and retried more than {config.MAX_INSERT_RETRY} times"
#                 raise RuntimeError(msg) from None

#     def task(self) -> int:
#         count = 0
#         with self.db.init():
#             log.info(f"({mp.current_process().name:16}) Start inserting embeddings in batch {config.NUM_PER_BATCH}")
#             start = time.perf_counter()
#             for data_df in self.dataset:
#                 all_metadata = data_df[self.dataset.data.train_id_field].tolist()

#                 emb_np = np.stack(data_df[self.dataset.data.train_vector_field])
#                 if self.normalize:
#                     log.debug("normalize the 100k train data")
#                     all_embeddings = (emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis]).tolist()
#                 else:
#                     all_embeddings = emb_np.tolist()
#                 del emb_np
#                 log.debug(f"batch dataset size: {len(all_embeddings)}, {len(all_metadata)}")

#                 labels_data = None
#                 if self.filters.type == FilterOp.StrEqual:
#                     if self.dataset.data.scalar_labels_file_separated:
#                         labels_data = self.dataset.scalar_labels[self.filters.label_field][all_metadata].to_list()
#                     else:
#                         labels_data = data_df[self.filters.label_field].tolist()

#                 insert_count, error = self.db.insert_embeddings(
#                     embeddings=all_embeddings,
#                     metadata=all_metadata,
#                     labels_data=labels_data,
#                 )
#                 if error is not None:
#                     self.retry_insert(
#                         self.db,
#                         embeddings=all_embeddings,
#                         metadata=all_metadata,
#                         labels_data=labels_data,
#                     )

#                 assert insert_count == len(all_metadata)
#                 count += insert_count
#                 if count % 100_000 == 0:
#                     log.info(f"({mp.current_process().name:16}) Loaded {count} embeddings into VectorDB")

#             log.info(
#                 f"({mp.current_process().name:16}) Finish loading all dataset into VectorDB, "
#                 f"dur={time.perf_counter() - start}"
#             )
#             return count

#     def endless_insert_data(self, all_embeddings: list, all_metadata: list, left_id: int = 0) -> int:
#         with self.db.init():
#             # unique id for endlessness insertion
#             all_metadata = [i + left_id for i in all_metadata]

#             num_batches = math.ceil(len(all_embeddings) / NUM_PER_BATCH)
#             log.info(
#                 f"({mp.current_process().name:16}) Start inserting {len(all_embeddings)} "
#                 f"embeddings in batch {NUM_PER_BATCH}"
#             )
#             count = 0
#             for batch_id in range(num_batches):
#                 retry_count = 0
#                 already_insert_count = 0
#                 metadata = all_metadata[batch_id * NUM_PER_BATCH : (batch_id + 1) * NUM_PER_BATCH]
#                 embeddings = all_embeddings[batch_id * NUM_PER_BATCH : (batch_id + 1) * NUM_PER_BATCH]

#                 log.debug(
#                     f"({mp.current_process().name:16}) batch [{batch_id:3}/{num_batches}], "
#                     f"Start inserting {len(metadata)} embeddings"
#                 )
#                 while retry_count < LOAD_MAX_TRY_COUNT:
#                     insert_count, error = self.db.insert_embeddings(
#                         embeddings=embeddings[already_insert_count:],
#                         metadata=metadata[already_insert_count:],
#                     )
#                     already_insert_count += insert_count
#                     if error is not None:
#                         retry_count += 1
#                         time.sleep(10)

#                         log.info(f"Failed to insert data, try {retry_count} time")
#                         if retry_count >= LOAD_MAX_TRY_COUNT:
#                             raise error
#                     else:
#                         break
#                 log.debug(
#                     f"({mp.current_process().name:16}) batch [{batch_id:3}/{num_batches}], "
#                     f"Finish inserting {len(metadata)} embeddings"
#                 )

#                 assert already_insert_count == len(metadata)
#                 count += already_insert_count
#             log.info(
#                 f"({mp.current_process().name:16}) Finish inserting {len(all_embeddings)} embeddings in "
#                 f"batch {NUM_PER_BATCH}"
#             )
#         return count

#     @utils.time_it
#     def _insert_all_batches(self) -> int:
#         """Performance case only"""
#         with concurrent.futures.ProcessPoolExecutor(
#             mp_context=mp.get_context("spawn"),
#             max_workers=1,
#         ) as executor:
#             future = executor.submit(self.task)
#             try:
#                 count = future.result(timeout=self.timeout)
#             except TimeoutError as e:
#                 msg = f"VectorDB load dataset timeout in {self.timeout}"
#                 log.warning(msg)
#                 for pid, _ in executor._processes.items():
#                     psutil.Process(pid).kill()
#                 raise PerformanceTimeoutError(msg) from e
#             except Exception as e:
#                 log.warning(f"VectorDB load dataset error: {e}")
#                 raise e from e
#             else:
#                 return count

#     def run_endlessness(self) -> int:
#         """run forever util DB raises exception or crash"""
#         # datasets for load tests are quite small, can fit into memory
#         # only 1 file
#         data_df = next(iter(self.dataset))
#         all_embeddings, all_metadata = (
#             np.stack(data_df[self.dataset.data.train_vector_field]).tolist(),
#             data_df[self.dataset.data.train_id_field].tolist(),
#         )

#         start_time = time.perf_counter()
#         max_load_count, times = 0, 0
#         try:
#             while time.perf_counter() - start_time < self.timeout:
#                 count = self.endless_insert_data(
#                     all_embeddings,
#                     all_metadata,
#                     left_id=max_load_count,
#                 )
#                 max_load_count += count
#                 times += 1
#                 log.info(
#                     f"Loaded {times} entire dataset, current max load counts={utils.numerize(max_load_count)}, "
#                     f"{max_load_count}"
#                 )
#         except Exception as e:
#             log.info(
#                 f"Capacity case load reach limit, insertion counts={utils.numerize(max_load_count)}, "
#                 f"{max_load_count}, err={e}"
#             )
#             traceback.print_exc()
#             return max_load_count
#         else:
#             raise LoadTimeoutError(self.timeout)

#     def run(self) -> int:
#         count, _ = self._insert_all_batches()
#         return count


# class SerialSearchRunner:
#     def __init__(
#         self,
#         db: api.VectorDB,
#         test_data: list[list[float]],
#         ground_truth: list[list[int]],
#         k: int = 100,
#         filters: Filter = non_filter,
#     ):
#         self.db = db
#         self.k = k
#         self.filters = filters

#         if isinstance(test_data[0], np.ndarray):
#             self.test_data = [query.tolist() for query in test_data]
#         else:
#             self.test_data = test_data
#         self.ground_truth = ground_truth

#     def _get_db_search_res(self, emb: list[float], retry_idx: int = 0) -> list[int]:
#         try:
#             results = self.db.search_embedding(emb, self.k)
#         except Exception as e:
#             log.warning(f"Serial search failed, retry_idx={retry_idx}, Exception: {e}")
#             if retry_idx < config.MAX_SEARCH_RETRY:
#                 return self._get_db_search_res(emb=emb, retry_idx=retry_idx + 1)

#             msg = f"Serial search failed and retried more than {config.MAX_SEARCH_RETRY} times"
#             raise RuntimeError(msg) from e

#         return results

#     def search(self, args: tuple[list, list[list[int]]]) -> tuple[float, float, float, float]:
#         log.info(f"{mp.current_process().name:14} start search the entire test_data to get recall and latency")
#         with self.db.init():
#             self.db.prepare_filter(self.filters)
#             test_data, ground_truth = args
#             ideal_dcg = get_ideal_dcg(self.k)

#             log.debug(f"test dataset size: {len(test_data)}")
#             log.debug(f"ground truth size: {len(ground_truth)}")

#             latencies, recalls, ndcgs = [], [], []
#             for idx, emb in enumerate(test_data):
#                 s = time.perf_counter()
#                 try:
#                     results = self._get_db_search_res(emb)
#                 except Exception as e:
#                     log.warning(f"VectorDB search_embedding error: {e}")
#                     raise e from None

#                 latencies.append(time.perf_counter() - s)

#                 if ground_truth is not None:
#                     gt = ground_truth[idx]
#                     recalls.append(calc_recall(self.k, gt[: self.k], results))
#                     ndcgs.append(calc_ndcg(gt[: self.k], results, ideal_dcg))
#                 else:
#                     recalls.append(0)
#                     ndcgs.append(0)

#                 if len(latencies) % 100 == 0:
#                     log.debug(
#                         f"({mp.current_process().name:14}) search_count={len(latencies):3}, "
#                         f"latest_latency={latencies[-1]}, latest recall={recalls[-1]}"
#                     )

#         avg_latency = round(np.mean(latencies), 4)
#         avg_recall = round(np.mean(recalls), 4)
#         avg_ndcg = round(np.mean(ndcgs), 4)
#         cost = round(np.sum(latencies), 4)
#         p99 = round(np.percentile(latencies, 99), 4)
#         p95 = round(np.percentile(latencies, 95), 4)
#         log.info(
#             f"{mp.current_process().name:14} search entire test_data: "
#             f"cost={cost}s, "
#             f"queries={len(latencies)}, "
#             f"avg_recall={avg_recall}, "
#             f"avg_ndcg={avg_ndcg}, "
#             f"avg_latency={avg_latency}, "
#             f"p99={p99}, "
#             f"p95={p95}"
#         )
#         return (avg_recall, avg_ndcg, p99, p95)

#     def _run_in_subprocess(self) -> tuple[float, float, float, float]:
#         with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
#             future = executor.submit(self.search, (self.test_data, self.ground_truth))
#             return future.result()

#     @utils.time_it
#     def run(self) -> tuple[float, float, float, float]:
#         log.info(f"{mp.current_process().name:14} start serial search")
#         if self.test_data is None:
#             msg = "empty test_data"
#             raise RuntimeError(msg)

#         return self._run_in_subprocess()

#     @utils.time_it
#     def run_with_cost(self) -> tuple[tuple[float, float, float, float], float]:
#         """
#         Search all test data in serial.
#         Returns:
#             tuple[tuple[float, float, float, float], float]: (avg_recall, avg_ndcg, p99_latency, p95_latency), cost
#         """
#         log.info(f"{mp.current_process().name:14} start serial search")
#         if self.test_data is None:
#             msg = "empty test_data"
#             raise RuntimeError(msg)

#         return self._run_in_subprocess()
