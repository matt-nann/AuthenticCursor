import psycopg2
from contextlib import contextmanager
import pandas as pd

from src import getSecret

@contextmanager
def db_connection():
    conn = psycopg2.connect(
        host=getSecret('SQL_Host'),
        port=getSecret('SQL_Port'),
        user=getSecret('SQL_User'),
        password=getSecret('SQL_Password'),
        database=getSecret('SQL_Database')
    )
    try:
        yield conn
    except:
        conn.rollback()
    finally:
        conn.commit()
        conn.close()

def get_dataframe_from_query(query):
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute(query)
        df = pd.DataFrame(cur.fetchall())
        df.columns = [desc[0] for desc in cur.description]
        return df

