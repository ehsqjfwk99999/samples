from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.task_group import TaskGroup

default_args = {"start_date": datetime(2020, 1, 1)}


def _branch_task(ti):
    if True:
        return 'dag_success'
    return 'dag_fail'

with DAG(dag_id="simple_dag",default_args=default_args,schedule_interval=None,catchup=False) as dag:
    forefront_task =   DummyOperator(task_id="forefront_task")

    with TaskGroup('sleeping_taskgroup') as sleeping_taskgroup:
        sleep_3=BashOperator(task_id='sleep_3', bash_command='sleep 3')
        sleep_6=BashOperator(task_id='sleep_6', bash_command='sleep 6')
        sleep_9=BashOperator(task_id='sleep_9', bash_command='sleep 9')
    
    gather_sleeping_taskgroup=BashOperator(task_id='gather_sleeping_taskgroup', bash_command='sleep 5')

    branch_task=BranchPythonOperator(task_id='branch_task',python_callable=_branch_task)

    dag_success = DummyOperator(task_id="dag_success")
    dag_fail = DummyOperator(task_id="dag_fail")

forefront_task >> sleeping_taskgroup >> gather_sleeping_taskgroup >> branch_task>> [dag_success,dag_fail]
