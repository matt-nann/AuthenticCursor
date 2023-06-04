import json
from src.flaskServer.models import db
from src.flaskServer.models import Quiz, Question, Option
from runFlask import app

with app.app_context():

    def load_quiz_data(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)

        # check if the quiz title is already present
        quiz = Quiz.query.filter_by(title=data['title']).first()
        if quiz:
            return

        quiz = Quiz(title=data['title'])
        db.session.add(quiz)
        db.session.commit()

        for question_data in data['questions']:
            question = Question(text=question_data['question'], quiz_id=quiz.id)
            db.session.add(question)
            db.session.commit()

            for option in question_data['options']:
                option = Option(text=option, question_id=question.id)
                db.session.add(option)
                db.session.commit()

    load_quiz_data('data/chefQuiz.json')
    load_quiz_data('data/plantQuiz.json')
    load_quiz_data('data/financeQuiz.json')

import json
from src.flaskServer.models import *

import json
from src.flaskServer.models import db
from src.flaskServer.models import Quiz, Question, Option
from runFlask import app

import json
import os
base = 'src/flaskServer/static/img/'
resultImages = []
for imageName in os.listdir(base):
    if imageName.endswith('.png'):
        resultImages.append(imageName)

with app.app_context():

    def load_quiz_results(json_file):
        # Load quiz JSON
        with open(json_file, 'r') as file:
            quiz_json = json.load(file)


        quiz = Quiz.query.filter_by(title=quiz_json["title"]).first()

        if not quiz:
            return
        if 'results' not in quiz_json:
            return
        # Create Result instances for each result in the JSON
        for result_name in quiz_json["results"]:
            if Result.query.filter_by(name=result_name).first():
                continue
            imagePath = [resultImage for resultImage in resultImages if result_name.replace(' ', '_') in resultImage]
            if len(imagePath) == 0:
                raise Exception('No image found for result: ' + result_name)
            result = Result(name=result_name, quiz=quiz, imagePath=imagePath[0])
            db.session.add(result)

        # Create ResultRange instances for each result range in the JSON
        for i, result_range in enumerate(quiz_json["grading"]["result_ranges"]):
            result = Result.query.filter_by(name=quiz_json["results"][result_range["result"]]).first()
            if ResultRange.query.filter_by(min=result_range["min"], max=result_range["max"], result_id=result.id).first():
                continue
            result_range = ResultRange(min=result_range["min"], max=result_range["max"], result_id=result.id)
            db.session.add(result_range)

        db.session.commit()
    
    load_quiz_results('data/chefQuiz.json')
    load_quiz_results('data/plantQuiz.json')
    load_quiz_results('data/financeQuiz.json')