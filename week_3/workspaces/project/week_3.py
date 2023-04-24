from datetime import datetime
from typing import List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    RetryPolicy,
    RunRequest,
    ScheduleDefinition,
    SensorEvaluationContext,
    SkipReason,
    String,
    graph,
    op,
    schedule,
    sensor,
    static_partitioned_config,
)
from workspaces.config import REDIS, S3
from workspaces.project.sensors import get_s3_keys
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
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}


docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}

@static_partitioned_config(partition_keys=[str(i) for i in range(1, 11)])
def docker_config(partition_key: str):
    return {
        "resources": {
            "s3": {"config": S3},
            "redis": {"config": REDIS},
        },
        "ops": {"get_s3_data": {"config": {"s3_key": f"prefix/stock_{partition_key}.csv"}}}
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
    op_retry_policy=RetryPolicy(max_retries=10, delay=1)
)



machine_learning_schedule_local = ScheduleDefinition(cron_schedule='*/15 * * * *', job=machine_learning_job_local)


@schedule(cron_schedule='0 * * * *', job=machine_learning_job_docker)
def machine_learning_schedule_docker():
    for partition_key in docker_config.get_partition_keys():
        yield RunRequest(run_key=partition_key, run_config=docker_config.get_run_config(partition_key))


@sensor(job=machine_learning_job_docker)
def machine_learning_sensor_docker(context):
    s3_keys = get_s3_keys(
        bucket='dagster',
        prefix='prefix',
        endpoint_url='http://localstack:4566'
    )
    if not s3_keys:
        yield SkipReason("No new s3 files found in bucket.")

    for s3_key in s3_keys:
        yield RunRequest(
            run_key=s3_key,
            run_config={
                "resources": {
                    "s3": {"config": S3},
                    "redis": {"config": REDIS},
                },
                "ops": {
                    "get_s3_data": {
                        "config": {
                            "s3_key": s3_key
                        }
                    }
                },
            },
        )
