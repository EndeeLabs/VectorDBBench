from pydantic import SecretStr
from typing import Optional
from ..api import DBConfig


class EndeeConfig(DBConfig):
    token: SecretStr
    region: Optional[str] = "india-west-1"
    base_url: str = "http://127.0.0.1:8080/api/v1"  # Default value
    space_type: str ="cosine"
    # use_fp16: bool = False
    precision: str = "medium"
    version: Optional[int] = 1
    m: Optional[int] = 16
    ef_con: Optional[int] = 128
    ef_search: Optional[int] = 128
    # collection_name: str
    index_name: str
    
    def to_dict(self) -> dict:
        return {
            "token": self.token.get_secret_value(),
            "region": self.region,
            "base_url": self.base_url,
            "space_type": self.space_type,
            # "use_fp16": self.use_fp16,
            "precision": self.precision,
            "version": self.version,
            "m": self.m,
            "ef_con": self.ef_con,
            "ef_search": self.ef_search,
            # "collection_name": self.collection_name,
            "index_name": self.index_name,
        }