from datetime import datetime
from typing import List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    String,
    graph,
    op,
)
from workspaces.config import REDIS, S3, S3_FILE
from workspaces.resources import mock_s3_resource, redis_resource, s3_resource
from workspaces.types import Aggregation, Stock


@op(
    config_schema={"s3_key": String},
    required_resource_keys={"s3"},
)
def get_s3_data(context) -> List[Stock]:
    """Bring in data, process into custom data type. Input is provided via config_schema"""
    stock_strings = context.resources.s3.get_data(context.op_config["s3_key"])
    return [Stock.from_list(i) for i in stock_strings]

@op
def process_data(context, stocks: List[Stock]) -> Aggregation:
    """Take the output of get_s3_data (list of Stock) and output a type Aggregation).
    Take the list of stocks and determine the Stock with the greatest high value"""
    max_value = stocks[0].high
    max_value_date = stocks[0].date
    for stock in stocks:
        if stock.high > max_value:
            max_value = stock.high
            max_value_date = stock.date
    return Aggregation(date=max_value_date, high=max_value)


@op(
    ins={"aggregation": In(dagster_type=Aggregation)},
    required_resource_keys={"redis"},
)
def put_redis_data(context, aggregation) -> Nothing:
    """Upload an Aggregation to Redis"""
    context.resources.redis.put_data(str(aggregation.date), str(aggregation.high))    



@op(
    ins={"aggregation": In(dagster_type=Aggregation)},
    required_resource_keys={"s3"},
)
def put_s3_data(context, aggregation) -> Nothing:
    """Upload an Aggregation to S3 File"""
    context.resources.s3.put_data(str(aggregation.date), aggregation)

@graph
def machine_learning_graph():
    aggregated = process_data(get_s3_data())
    put_redis_data(aggregated)
    put_s3_data(aggregated)


local = {
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
    config=local,
    resource_defs={"s3": mock_s3_resource, "redis": ResourceDefinition.mock_resource()},
)

machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
    config=docker,
    resource_defs={"s3": s3_resource, "redis": redis_resource},
)
