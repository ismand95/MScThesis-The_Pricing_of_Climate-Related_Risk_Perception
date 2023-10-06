import sqlalchemy
from sqlalchemy.sql import text
import pandas as pd
import re
import os
import boto3
import time
from datetime import datetime
from dateutil import tz

BUCKET_NAME = 'personal-default-jedimaster-data'


def update_database(db_path='sqlite:///JediMaster.db'):
    '''
    Updates the key in S3 by overwriting the key - only if local database
    is newer (last modified) than S3 version.
    '''
    # initialize S3 AWS session
    session = boto3.Session(profile_name='jedimaster')
    s3_client = session.client('s3')

    # extract DB file-name from `sqlite:///` string
    db_file = re.findall(string=db_path, pattern=r'\w+.db')[0]

    # get last-modified stamp of local db
    local_modified = time.ctime(os.path.getmtime(db_file))
    local_modified = datetime.strptime(local_modified, '%a %b %d %H:%M:%S %Y')

    # get last-modified stamp of S3 db
    s3_modified = s3_client.head_object(
        Bucket=BUCKET_NAME, Key=db_file)['LastModified']

    # convert to local tz timestamp
    s3_modified = s3_modified.astimezone(tz.tzlocal())
    s3_modified = s3_modified.replace(tzinfo=None)

    if local_modified > s3_modified:
        print('Local database newer than S3 database, uploading...')

        # s3_client.upload_file(file_name, bucket, object_name)
        s3_client.upload_file(db_file, BUCKET_NAME, db_file)

    else:
        print('Local database older than S3 database, downloading...')

        # Download from S3
        session = boto3.Session(profile_name='jedimaster')
        s3_client = session.client('s3')
        s3_client.download_file(BUCKET_NAME, db_file, db_file)


def connect_to_db(db_path='sqlite:///JediMaster.db'):
    '''
    Connects to SQLite database, if present.
    Otherwise fetches from S3 using the `boto3` API.
    '''

    # extract DB file-name from `sqlite:///` string
    db_file = re.findall(string=db_path, pattern=r'\w+.db')[0]

    # static path from DeepNote S3 integration
    deepnote_path = '/datasets/aws-s3-integration/'

    if os.path.isfile(f'{deepnote_path}{db_file}'):
        return sqlalchemy.create_engine(f'sqlite:///{deepnote_path}{db_file}')

    elif os.path.isfile(db_file):
        return sqlalchemy.create_engine(f'sqlite:///{db_file}')

    else:
        # Download from S3
        session = boto3.Session(profile_name='jedimaster')
        s3_client = session.client('s3')
        s3_client.download_file(BUCKET_NAME, db_file, db_file)

        return sqlalchemy.create_engine(f'sqlite:///{db_file}')


def vacuum_db(engine):
    with engine.connect() as connection:
        connection.execute(text('VACUUM'))

    return True


def read_db(engine, statement, idx_col='index', dt_col='date'):
    df = pd.read_sql(
        statement,
        con=engine,
        index_col=idx_col,
        parse_dates=[dt_col]
    )

    if idx_col != 'index':
        df = df.drop(columns=['index'])

    # remove index name
    df.index.name = None

    return df
