# imports 
import pandas as pd

# import google.auth
# from google.cloud import bigquery



# this is my billing project
# NOTE:  You shoudl replace below with your own billing project
PROJECT_ID = 'ba-820-lt'


# make the query and get the data
SQL = "select * from `questrom.datasets.diamonds` limit 5"


# we can also do this via pandas 
df = pd.read_gbq(SQL, PROJECT_ID)
df.head()
