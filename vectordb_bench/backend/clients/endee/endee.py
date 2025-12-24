import logging
from contextlib import contextmanager

from endee import endee

from ..api import DBCaseConfig, EmptyDBCaseConfig, IndexType, VectorDB
from .config import EndeeConfig

log = logging.getLogger(__name__)


class Endee(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        drop_old: bool = False,
        **kwargs,
    ):
        print(db_config)

        self.token = db_config.get("token", "")
        self.region = db_config.get("region", "india-west-1")
        self.base_url = db_config.get("base_url")
        
        # self.collection_name = db_config.get("collection_name", "")
        self.collection_name = (db_config.get("collection_name") or db_config.get("index_name"))
        
        if not self.collection_name:
            import uuid
            self.collection_name = f"endee_bench_{uuid.uuid4().hex[:8]}"
        
        self.space_type = db_config.get("space_type", "cosine")
        # self.use_fp16 = db_config.get("use_fp16", False)
        self.precision = db_config.get("precision")
        self.version = db_config.get("version")
        self.M = db_config.get("m")
        self.ef_con = db_config.get("ef_con")
        self.ef_search = db_config.get("ef_search")
        self.nd = endee.Endee(token=self.token)

        # Dynamically set the URL
        if self.base_url:
            self.nd.set_base_url(self.base_url)
            log.info(f"Targeting server: {self.base_url}")

        # Using Base url
        # self.nd.base_url = "http://10.128.0.3:8080/api/v1"
        # self.nd.set_base_url("http://10.128.0.3:8080/api/v1")
        # self.nd.set_base_url("http://3.85.217.253:80/api/v1")
        # self.nd.set_base_url("http://10.128.0.5:8080/api/v1")
        # self.nd.set_base_url("http://54.89.169.48:80/api/v1")


        # BASE_URL="http://3.85.217.253:80/api/v1"

        # try:
        #     indices = self.nd.list_indexes().get("indices", [])
        #     index_names = [index["name"] for index in indices] if indices else []
            
        #     if drop_old and self.collection_name in index_names:
        #         self._drop_index(self.collection_name)
        #         self._create_index(dim)
        #     elif self.collection_name not in index_names:
        #         self._create_index(dim)
        # except Exception as e:
        #     log.error(f"Error connecting to Endee API: {e}")
        #     raise

        try:
            index_name = self.collection_name  # assign before use
            indices = self.nd.list_indexes().get("indices", [])
            index_names = [index["name"] for index in indices] if indices else []
            try:
                self.index = self.nd.get_index(name=index_name)
                log.info(f"Connected to existing Endee index: '{index_name}'")

            except Exception as fetch_error:
                log.warning(f"Index '{index_name}' not found. Creating new index...")
                try:
                    self._create_index(dim)
                    self.index = self.nd.get_index(name=index_name)
                    log.info(f"Successfully created and connected to index: '{index_name}'")

                except Exception as create_error:
                    # If error is "index already exists", just get the index again
                    if "already exists" in str(create_error).lower() or "conflict" in str(create_error).lower():
                        log.warning(f"Index '{index_name}' already exists despite previous error. Fetching it again.")
                        self.index = self.nd.get_index(name=index_name)
                    else:
                        log.error(f"Failed to create Endee index: {create_error}")
                        raise
        except Exception as e:
            log.error(f"Error accessing or creating Endee index '{index_name}': {e}")
            raise



    def _create_index(self, dim):
        try:
            resp = self.nd.create_index(
                name=self.collection_name,
                dimension=dim,
                space_type=self.space_type,
                # use_fp16=self.use_fp16,
                precision=self.precision,
                version=self.version,
                M=self.M,
                ef_con=self.ef_con
            )
            log.info(f"Created new Endee index: {resp}")
        except Exception as e:
            log.error(f"Failed to create Endee index: {e}")
            raise

    def _drop_index(self, collection_name):
        try:
            res = self.nd.delete_index(collection_name)
            log.info(res)
        except Exception as e:
            log.error(f"Failed to drop Endee index: {e}")
            raise

    @classmethod
    def config_cls(cls) -> type[EndeeConfig]:
        return EndeeConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> type[DBCaseConfig]:
        return EmptyDBCaseConfig

    @contextmanager
    def init(self):
        try:
            # log.info(f"Token: {self.token}")
            nd = endee.Endee(token=self.token)
            # Uncomment below to use base_url
            # nd.base_url = "http://10.128.0.3:8080/api/v1"
            # nd.set_base_url("http://10.128.0.3:8080/api/v1")
            # nd.set_base_url("http://3.85.217.253:80/api/v1")
            # nd.set_base_url("http://10.128.0.5:8080/api/v1")
            # nd.set_base_url("http://54.89.169.48:80/api/v1")

            if self.base_url:
                nd.set_base_url(self.base_url)
            self.nd = nd

            self.index = nd.get_index(name=self.collection_name)
            yield
        except Exception as e:
            if hasattr(e, 'response') and e.response is not None:
                log.error(f"HTTP Status: {e.response.status_code}, Body: {e.response.text}")
            log.error(f"Error initializing Endee client: {e}")
            # log.error("Error initializing Endee client", exc_info=True)
            raise
        finally:
            pass

    def optimize(self, data_size: int | None = None):
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception): # type: ignore
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            batch_vectors = [
                {
                    "id": str(metadata[i]),
                    "vector": embeddings[i],
                    "meta": {"id": str(metadata[i])}
                }
                for i in range(len(embeddings))
            ]
            self.index.upsert(batch_vectors)
            insert_count = len(batch_vectors)
            
        except Exception as e:
            return (insert_count, e)
            
        return (len(embeddings), None)

    def search_embedding(
        self,
        query: list[float],
        k: int = 30,
        filters: dict | None = None,
        **kwargs,
    ) -> list[int]:
        try:
            filter_expr = None
            if filters and "id" in filters:
                filter_expr = {"id": filters["id"]}
                
            results = self.index.query(
                vector=query,
                top_k=k,
                filter=filter_expr,
                ef=self.ef_search,
                include_vectors=False
            )
            
            return [int(result["id"]) for result in results]
            
        except Exception as e:
            log.warning(f"Error querying Endee index: {e}")
            raise

    def describe_index(self) -> dict:
        """Get information about the current index."""
        try:
            all_indices = self.nd.list_indexes().get("indices", [])
            
            for idx in all_indices:
                if idx.get("name") == self.collection_name:
                    return idx
                    
            return {}
        except Exception as e:
            log.warning(f"Error describing Endee index: {e}")
            return {}
