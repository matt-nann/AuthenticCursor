from src import FLASK_PORT
from src.flaskServer import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=FLASK_PORT,threaded=True)