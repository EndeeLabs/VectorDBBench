import click
import logging
import uuid
from typing import Annotated

from vectordb_bench.cli.cli import (
    CommonTypedDict,
    benchmark_runner,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    parse_task_stages,
)
from .. import DB
from .config import EndeeConfig
from ..api import EmptyDBCaseConfig

log = logging.getLogger(__name__)


class EndeeTypedDict(CommonTypedDict):
    token: Annotated[
        str, 
        click.option("--token", type=str, required=True, help="Endee API token")
    ]
    region: Annotated[
        str, 
        click.option("--region", type=str, default=None, help="Endee region", show_default=True)
    ]
    base_url: Annotated[
        str, 
        click.option("--base-url", type=str, default="http://127.0.0.1:8080/api/v1", help="API server URL", show_default=True)
    ]
    space_type: Annotated[
        str, 
        click.option("--space-type", type=click.Choice(["cosine", "l2", "dot_product"]), default="cosine", help="Distance metric", show_default=True)
    ]
    precision: Annotated[
        str, 
        # click.option("--precision", type=click.Choice(["medium", "high", "ultra-high", "fp16", "small"]), default="medium", help="Quant Level", show_default=True)
        click.option("--precision", type=click.Choice(["binary", "int4d", "int8d", "int16d", "float16", "float32"]), default="int8d", help="Quant Level", show_default=True)
    ]
    version: Annotated[
        int, 
        click.option("--version", type=int, default=None, help="Index version", show_default=True)
    ]
    m: Annotated[
        int, 
        click.option("--m", type=int, default=None, help="HNSW M parameter", show_default=True)
    ]
    ef_con: Annotated[
        int, 
        click.option("--ef-con", type=int, default=None, help="HNSW construction parameter", show_default=True)
    ]
    ef_search: Annotated[
        int, 
        click.option("--ef-search", type=int, default=None, help="HNSW search parameter", show_default=True)
    ]
    index_name: Annotated[
        str, 
        click.option("--index-name", type=str, required=True, help="Endee index name (will use a random name if not provided)", show_default=True)
    ]



@click.command()
@click_parameter_decorators_from_typed_dict(EndeeTypedDict)
def Endee(**parameters):
    stages = parse_task_stages(
        parameters["drop_old"],
        parameters["load"],
        parameters["search_serial"],
        parameters["search_concurrent"],
    )
    
    # Generate a random collection name if not provided
    collection_name = parameters["index_name"]
    if not collection_name:
        collection_name = f"endee_bench_{uuid.uuid4().hex[:8]}"

    # Filter out None values before creating config
    params_for_vecx = {k: v for k, v in parameters.items() if v is not None}
    db_config = EndeeConfig(**params_for_vecx)

    custom_case_config = get_custom_case_config(parameters)

    # Create task config
    from vectordb_bench.models import TaskConfig, CaseConfig, CaseType, ConcurrencySearchConfig
    
    # Create an instance of EmptyDBCaseConfig instead of passing None
    db_case_config = EmptyDBCaseConfig()
    
    task = TaskConfig(
        db=DB.Endee,
        db_config=db_config,  # Use the EndeeConfig instance directly
        db_case_config=db_case_config,
        case_config=CaseConfig(
            case_id=CaseType[parameters["case_type"]],
            k=parameters["k"],
            concurrency_search_config=ConcurrencySearchConfig(
                concurrency_duration=parameters["concurrency_duration"],
                num_concurrency=[int(s) for s in parameters["num_concurrency"]],
            ),
            custom_case=custom_case_config,
        ),
        stages=stages,
    )
    
    # Use the run method of the benchmark_runner object
    if not parameters["dry_run"]:
        benchmark_runner.run([task])
        
        # Wait for task to complete if needed
        import time
        from vectordb_bench.interface import global_result_future
        from concurrent.futures import wait
        
        time.sleep(5)
        if global_result_future:
            wait([global_result_future]) 