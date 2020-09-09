import asyncio

import click
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
from pandas import DataFrame


@click.group()
def cli():
    pass


def init_mongo(mongo_url: str):
    motor_client: AsyncIOMotorClient = AsyncIOMotorClient(mongo_url)
    return motor_client["data"]


async def insert_data(record, collection, db):
    result = await db[collection].insert_one(record)
    print(result.inserted_id)


async def find_data(collection, db):
    cursor = db[collection].find({}, projection={"_id": 0})
    data = []

    async for c in cursor:
        print(c["ID"])
        data.append(c)

    return data


def callback(future):
    return future.result()


@cli.command(name="upload_to_mongo")
@click.option("--mongo_url")
@click.option("--file_path")
@click.option("--file_type")
def upload_data_to_mongo(mongo_url: str, file_path: str, file_type: str):
    df: DataFrame = pd.read_csv(file_path, sep="###__###", header=None)

    if file_type == "train":
        df.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_List']
    elif file_type == "test":
        df.columns = ['ID', 'Query_List']

    res: dict = df.to_dict(orient="records")

    db = init_mongo(mongo_url)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait([insert_data(r, file_type, db) for r in res]))


@cli.command(name="upload_to_mongo")
@click.option("--mongo_url")
@click.option("--file_type")
def download_data_from_mongo(mongo_url: str, file_type: str):
    db = init_mongo(mongo_url)

    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(find_data(file_type, db))
    loop.run_until_complete(future)

    return DataFrame(future.result())


if __name__ == "__main__":
    cli()
