from flask_wtf import FlaskForm
from wtforms import FieldList, FormField, SubmitField, FieldList, FormField, RadioField
from wtforms.validators import DataRequired

from .models import TargetType

class QuestionForm(FlaskForm):
    question = RadioField(label='question', choices=[], validators=[DataRequired()])

class QuizForm(FlaskForm):
    def __init__(self, questions=None, *args, **kwargs):
        super(QuizForm, self).__init__(*args, **kwargs)
        startingSize = len(self.questions)
        if questions:
            for i, question in enumerate(questions):
                # when submitting the form it adds additional empty questions
                if i < startingSize:
                    field = self.questions[i]
                else:
                    field = QuestionForm()
                    # for whatever reason this doesn't work
                    # field.question.label = question.text
                    # field.question.choices = [(str(idx), option.text) for idx, option in enumerate(question.options)]
                    self.questions.append_entry(field)

    questions = FieldList(FormField(QuestionForm), min_entries=0)
    submit = SubmitField('Submit', render_kw={'target_type': TargetType.NEXTPAGE.value})

    def validate_custom(self, questionIds):
        selected_choices = []
        valid = True
        selected_choices = [None] * len(self.questions)
        for question in self.questions:
            if not isinstance(question.question.data, str) or (isinstance(question.question.data, str) and question.question.data[0] == "<"):
                valid = False
            else:
                idx, questionId, optionId = question.question.data.split("_")
                ids_idx = questionIds.index(int(questionId))
                selected_choices[ids_idx] = (int(idx), int(questionId), int(optionId))
        return valid, selected_choices