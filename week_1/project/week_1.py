import csv
from datetime import datetime
from typing import Iterator, List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    String,
    job,
    op,
    usable_as_dagster_type,
)
from pydantic import BaseModel


@usable_as_dagster_type(description="Stock data")
class Stock(BaseModel):
    date: datetime
    close: float
    volume: int
    open: float
    high: float
    low: float

    @classmethod
    def from_list(cls, input_list: List[str]):
        """Do not worry about this class method for now"""
        return cls(
            date=datetime.strptime(input_list[0], "%Y/%m/%d"),
            close=float(input_list[1]),
            volume=int(float(input_list[2])),
            open=float(input_list[3]),
            high=float(input_list[4]),
            low=float(input_list[5]),
        )


@usable_as_dagster_type(description="Aggregation of stock data")
class Aggregation(BaseModel):
    date: datetime
    high: float


def csv_helper(file_name: str) -> Iterator[Stock]:
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield Stock.from_list(row)


@op(config_schema={"s3_key": String})
def get_s3_data_op(context) -> List[Stock]:
    """Bring in data, process into custom data type. Input is provided via config_schema"""
    return list(csv_helper(context.op_config["s3_key"]))


@op
def process_data_op(context, stocks: List[Stock]) -> Aggregation:
    """Take the output of get_s3_data (list of Stock) and output a type Aggregation).
    Take the list of stocks and determine the Stock with the greatest high value"""
    max_value = stocks[0].high
    max_value_date = stocks[0].date
    for stock in stocks:
        if stock.high > max_value:
            max_value = stock.high
            max_value_date = stock.date
    return Aggregation(date=max_value_date, high=max_value)


@op(ins={"aggregation": In(dagster_type=Aggregation)})
def put_redis_data_op(context, aggregation) -> Nothing:
    """Upload an Aggregation to Redis"""
    pass


@op(ins={"aggregation": In(dagster_type=Aggregation)})
def put_s3_data_op(context, aggregation) -> Nothing:
    """Upload an Aggregation to S3 File"""
    pass

@job
def machine_learning_job():
    aggregated = process_data_op(get_s3_data_op())
    put_redis_data_op(aggregated)
    put_s3_data_op(aggregated)