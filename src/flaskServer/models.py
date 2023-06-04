import requests
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Enum
from enum import Enum as PyEnum

from .config import Config

db = SQLAlchemy()

class BaseModel(db.Model):
    __abstract__ = True
    __table_args__ = {'schema': Config.SQLALCHEMY_SCHEMA}

#################### Quiz data #######################

class Quiz(BaseModel):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    questions = db.relationship('Question', backref='quiz', lazy=True)
    results = db.relationship('Result', backref='quiz', lazy=True)

class Question(BaseModel):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    quiz_id = db.Column(db.Integer, db.ForeignKey(f'{Config.SQLALCHEMY_SCHEMA}.quiz.id'), nullable=False)
    options = db.relationship('Option', backref='question', lazy=True)

class Option(BaseModel):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(255), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey(f'{Config.SQLALCHEMY_SCHEMA}.question.id'), nullable=False)

class Result(BaseModel):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    quiz_id = db.Column(db.Integer, db.ForeignKey(f'{Config.SQLALCHEMY_SCHEMA}.quiz.id'), nullable=False)
    imagePath = db.Column(db.String(255), nullable=False)
    resultRanges = db.relationship('ResultRange', backref='result', lazy=True)
    
class ResultRange(BaseModel):
    id = db.Column(db.Integer, primary_key=True)
    min = db.Column(db.Integer, nullable=False)
    max = db.Column(db.Integer, nullable=False)
    result_id = db.Column(db.Integer, db.ForeignKey(f'{Config.SQLALCHEMY_SCHEMA}.result.id'), nullable=False)

#################### Models for gathering mouse movement data #######################

class Visitor(BaseModel):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ip_address = db.Column(db.String(120))
    country = db.Column(db.String(120))
    region = db.Column(db.String(120))
    city = db.Column(db.String(120))
    zip = db.Column(db.String(10))
    isp = db.Column(db.String(120))
    user_agent = db.Column(db.String(300))
    first_visit = db.Column(db.DateTime)

    def query_ip(self, ip_address):
        # https://ip-api.com/docs/api:json
        url = 'http://ip-api.com/json/' + ip_address
        response = requests.get(url)
        data = response.json()
        print(data)
        return data

    def __init__(self, ip_address, user_agent):
        self.ip_address = ip_address
        self.user_agent = user_agent
        data = self.query_ip(ip_address)
        if data['status'] == 'fail':
            self.country = 'Unknown'
            self.region = 'Unknown'
            self.city = 'Unknown'
            self.zip = 'Unknown'
            self.isp = 'Unknown'
        else:
            self.country = data['country']
            self.region = data['regionName']
            self.city = data['city']
            self.zip = data['zip']
            self.isp = data['isp']
        self.first_visit = datetime.now()

class TargetType(PyEnum):
    RADIO = "Radio Option"
    NEXTPAGE = "Next Page"

class InputTarget(BaseModel):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    x = db.Column(db.Float)
    y = db.Column(db.Float)
    width = db.Column(db.Float)
    height = db.Column(db.Float)
    target_type = db.Column(Enum(TargetType), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey(f'{Config.SQLALCHEMY_SCHEMA}.question.id'), nullable=True)
    option_id = db.Column(db.Integer, db.ForeignKey(f'{Config.SQLALCHEMY_SCHEMA}.option.id'), nullable=True)

def addingButtonsAndStates():
    import json
    from src.flaskServer.models import db
    from src.flaskServer.models import VisitorButton, VisitorInputState
    from runFlask import app

    with app.app_context():
        buttons = ['NoButton','Left','Right']
        states = ['Move','Pressed']
        for button in buttons:
            if VisitorButton.query.filter_by(name=button).first() is None:
                db.session.add(VisitorButton(button))
        for state in states:
            if VisitorInputState.query.filter_by(name=state).first() is None:
                db.session.add(VisitorInputState(state))
        db.session.commit()

class VisitorButton(BaseModel):
    id = db.Column(db.SmallInteger, primary_key=True, autoincrement=True)
    name = db.Column(db.String(20), nullable=False)
    def __init__(self, name):
        self.name = name

class VisitorInputState(BaseModel):
    id = db.Column(db.SmallInteger, primary_key=True, autoincrement=True)
    name = db.Column(db.String(20), nullable=False)
    def __init__(self, name):
        self.name = name

class MouseTrajectory(BaseModel):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    visitor_id = db.Column(db.Integer, db.ForeignKey(f'{Config.SQLALCHEMY_SCHEMA}.visitor.id'), nullable=False)
    target_id = db.Column(db.Integer, db.ForeignKey(f'{Config.SQLALCHEMY_SCHEMA}.input_target.id'), nullable=False)

    def __init__(self, visitor_id, target_id):
        self.visitor_id = visitor_id
        self.target_id = target_id

class VisitorAction(BaseModel):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    sequence_id = db.Column(db.Integer, db.ForeignKey(f'{Config.SQLALCHEMY_SCHEMA}.mouse_trajectory.id'), nullable=False)
    clientTimestamp = db.Column(db.Float, nullable=False)
    recordTimestamp = db.Column(db.BigInteger, nullable=False)
    button = db.Column(db.SmallInteger, db.ForeignKey(f'{Config.SQLALCHEMY_SCHEMA}.visitor_button.id'), nullable=False)
    state = db.Column(db.SmallInteger, db.ForeignKey(f'{Config.SQLALCHEMY_SCHEMA}.visitor_input_state.id'), nullable=False)
    x = db.Column(db.SmallInteger)
    y = db.Column(db.SmallInteger)

    def __init__(self, sequence_id, recordTimestamp, clientTimestamp, button, state, x, y):
        self.sequence_id = sequence_id
        self.recordTimestamp = recordTimestamp
        self.clientTimestamp = clientTimestamp
        self.button = button
        self.state = state
        self.x = x
        self.y = y