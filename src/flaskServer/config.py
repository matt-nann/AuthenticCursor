import os
from src import getSecret, isRunningInCloud
# Grabs the folder where the script runs.
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SQLALCHEMY_SCHEMA = 'authentic_cursor'


    # Secret key for session management. You can generate random strings here: https://randomkeygen.com/
    # A secret key is required to use CSRF.
    SECRET_KEY = "*_gzD'>'U-)7~(=<QonaVmni6j==VW?ItW#<kkP/axVZo!'PjSPKfe;%$I?e)=L"

    DEBUG = False if isRunningInCloud() else True

    if isRunningInCloud():
        # module not found error if using the uri provided by heroku
        SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL'].replace('postgres://', 'postgresql://')
    else:
        SQL_username = str(getSecret('SQL_User'))
        SQL_password = str(getSecret('SQL_Password'))
        SQL_host = str(getSecret('SQL_Host'))
        SQL_port = str(getSecret('SQL_Port'))
        SQL_database = str(getSecret('SQL_Database'))
        SQLALCHEMY_DATABASE_URI = 'postgresql://' + SQL_username + ':' + SQL_password + '@' + SQL_host + ':' + SQL_port + '/' + SQL_database