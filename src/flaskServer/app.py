import os
from flask import Flask, render_template, request, flash, redirect, url_for, abort, jsonify, session
from flask_migrate import Migrate
import logging
from logging import Formatter, FileHandler
import json
from flask_wtf.csrf import CSRFProtect
from markupsafe import Markup
import math

from src.flaskServer.mouseData import MouseData
from src.flaskServer.config import Config
from src.flaskServer.forms import *
from src.flaskServer.models import db, Quiz, ResultRange, Visitor, TargetType

def json_script(obj, var_name):
    json_str = json.dumps(obj)
    script = f'<script>var {var_name} = {json_str};</script>'
    return Markup(script)

def get_attr(obj, attr_name):
    return getattr(obj, attr_name)
def has_attr(obj, attr_name):
    return hasattr(obj, attr_name)

def zip_lists(list1, list2):
    return zip(list1, list2)

def create_app():

    app = Flask(__name__)
    csrf = CSRFProtect(app)
    csrf._exempt_views.add('dash.dash.dispatch') # TODO possibly a security risk, but it's a hack to get dash callbacks working within a flask app with CSRF protection

    # Add the filter to the Jinja2 environment
    app.jinja_env.filters['json_script'] = json_script
    app.jinja_env.filters['getattr'] = get_attr
    app.jinja_env.filters['hasattr'] = has_attr
    app.jinja_env.filters['zip'] = zip_lists

    app.config.from_object(Config)

    db.init_app(app)
    migrate = Migrate(app, db)
    with app.app_context():
        db.create_all()

    def recordVisitor():
        user_agent = request.headers.get('User-Agent')
        try:
            ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
        except:
            ip_address = request.remote_addr
        visitor = Visitor.query.filter_by(ip_address=ip_address, user_agent=user_agent).first()
        if visitor:
            return visitor
        else:
            visitor = Visitor(ip_address=ip_address, user_agent=user_agent)
            db.session.add(visitor)
            db.session.commit()
        return visitor
    
    @app.route('/')
    def home():
        # temporarily the start of the mouse movement data collection
        visitor = recordVisitor()
        session['visitor_id'] = visitor.id
        return render_template('pages/quizStart.html')
        # return render_template('pages/home.html')

    @app.route('/dump', methods=['POST'])
    def dump():
        print('dumping data')
        data = request.get_json()
        print(data)
        return jsonify({'success': True})
    
    @app.route('/quiz/<int:quiz_id>/<int:page>', methods=['GET', 'POST'])
    def quiz(quiz_id, page):
        quiz = Quiz.query.get(quiz_id)
        if not quiz:
            abort(404)
        questions = quiz.questions
        questions_per_page = 3
        total_questions = len(questions) 
        total_pages = math.ceil(total_questions / questions_per_page)
        start = (page - 1) * questions_per_page
        end = start + questions_per_page
        current_questions = questions[start:end]
        questionIds = [question.id for question in current_questions]
        form = QuizForm(questions=current_questions)
        for f_question, question in zip(form.questions, current_questions):
            f_question.question.label.text = question.text
            f_question.question.choices = [(str(idx) + "_" + str(option.question_id) + "_" + str(option.id), option.text) for idx, option in enumerate(question.options)]
        # fill with None for the number of questions
        selected_choices = [None for _ in range(len(current_questions))]
        if request.method == 'POST':
            valid, selected_choices = form.validate_custom(questionIds)
            print("selected_choices: ", selected_choices, "valid: ", valid)
            if valid:
                session[f'answers_quiz_{quiz_id}_page_{page}'] = selected_choices
                if page < total_pages:
                    # Redirect to the next page
                    return redirect(url_for('quiz', quiz_id=quiz_id, page=page + 1))
                else:
                    for i in range(1, page):
                        selected_choices += session[f'answers_quiz_{quiz_id}_page_{i}']
                    session[f'answers_quiz_{quiz_id}'] = selected_choices
                    return redirect(url_for('quiz_results', quiz_id=quiz_id))
            else:
                flash('Please answer all questions')
        print("selected_choices by passed: ", selected_choices)
        return render_template('forms/quiz.html', form=form, TargetTypes=TargetType, selected_choices=selected_choices)
    
    @app.route('/quiz/<int:quiz_id>/results/', methods=['GET', 'POST'])
    def quiz_results(quiz_id):
        if not session.get('visitor_id'):
            return redirect(url_for('home'))
        quiz = Quiz.query.get(quiz_id)
        answers = session[f'answers_quiz_{quiz_id}']
        if not quiz.results: # last financial form doesn't have quiz data
            return render_template('pages/complete.html')
        totalScore = sum([answer[0] + 1 for answer in answers])
        resultName = ''
        resultImagePath = ''
        for result in quiz.results:
            try:
                resultRange = ResultRange.query.where(ResultRange.result_id == result.id).all()[0]
            except:
                print('no result range for result ', result.id, 'for visitor ', session['visitor_id'])
                continue
            if totalScore >= resultRange.min and totalScore <= resultRange.max:
                resultName = result.name
                resultImagePath = result.imagePath
                break
        return render_template('forms/quiz_results.html', quizTitle=quiz.title, quiz_id= quiz.id,
                               resultName=resultName, resultImagePath=resultImagePath)

    @app.errorhandler(500)
    def internal_error(error):
        #db_session.rollback()
        return render_template('errors/500.html'), 500
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('errors/404.html'), 404
   
    mouseData = MouseData(db, csrf)
    setattr(app, 'mouseData', mouseData)
    mouseData.add_routes(app, db)
    
    if not app.debug:
        file_handler = FileHandler('error.log')
        file_handler.setFormatter(
            Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
        )
        app.logger.setLevel(logging.INFO)
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.info('errors')

    return app