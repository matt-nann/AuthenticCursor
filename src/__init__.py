import os
from dotenv import load_dotenv
import string
import random
import secrets

FLASK_PORT = 8080
CLOUD_URL = 'https://authentic_cursor.herokuapp.com' # TODO replace with your own website url

load_dotenv()

def getSecret(secret):
    return os.environ.get(secret)

def isRunningInCloud():
    return os.environ.get('RUNNING') == 'cloud'

def baseUrl():
    if isRunningInCloud():
        return CLOUD_URL
    else:
        return 'http://127.0.0.1:'+str(FLASK_PORT)