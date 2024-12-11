import logging as logger
from datetime import datetime
from airflow.decorators import task, dag


@dag(start_date=datetime(2021, 1, 1), schedule=None, catchup=False)
def test_dag():

    @task
    def start():
        logger.info("Start")
        return 1

    @task
    def hello_world(start:int = 0):
        if start:
            logger.info("Hello world")
        else:
            logger.info("not started")

        return 1

    @task
    def end(val: int = 0):
        return val

    s = start()
    e = hello_world(s)
    end(e)


test_dag()
