
from flask import request, make_response, jsonify, session
import json
import pandas as pd

import json
from src import isRunningInCloud
from src.flaskServer.models import VisitorAction, MouseTrajectory, VisitorInputState, VisitorButton, InputTarget, TargetType

class MouseData:
    def __init__(self, db, csrf):
        self.db = db
        self.csrf = csrf
        self.enum_dict = {member.value: member for member in TargetType}

    def get_target_id(self, targetDict):
        target_type = self.enum_dict[targetDict['target_type']]
        kwargs = {x:targetDict[x] for x in ['x', 'y', 'width', 'height', 'question_id', 'option_id']}
        kwargs = {'target_type': target_type, **kwargs}
        target = InputTarget.query.filter_by(**kwargs).first()
        if target is None:
            target = InputTarget(**kwargs)
            self.db.session.add(target)
            self.db.session.commit()
        return target.id
    
    def deleteMouseData(self):
        # delete everything from VisitorAction, TargetSequence, InputTarget
        confirmation = input('Are you sure you want to delete all VisitorAction, TargetSequence, InputTarget records? (y/n)')
        if confirmation == 'y':
            second_confirmation = input('Are you really sure? (yes/no)')
            if second_confirmation == 'yes':
                VisitorAction.query.delete()
                MouseTrajectory.query.delete()
                InputTarget.query.delete()
                self.db.session.commit()

    def add_routes(self, app, db):
        
        @app.route('/saveMouseDataSequence', methods=['POST'])
        def saveMouseDataSequence():
            try:
                df = pd.DataFrame(request.get_json())
                visitor_id = session['visitor_id']
                actions = []
                df.sort_values(by=['recordTimestamp'], inplace=True, ascending=False)
                targetPressedEvent = df.iloc[0] 
                if pd.notnull(targetPressedEvent['target']):
                    target_id = self.get_target_id(targetPressedEvent['target'])
                else:
                    raise Exception('Last event must be a target pressed event')
                sequence = MouseTrajectory(target_id = target_id, visitor_id = visitor_id)
                db.session.add(sequence)

                buttons = VisitorButton.query.all()
                buttons = {button.name:button.id for button in buttons}
                states = VisitorInputState.query.all()
                states = {state.name:state.id for state in states}
                df['button'] = df['button'].apply(lambda x: buttons[x])
                df['state'] = df['state'].apply(lambda x: states[x])
                for row in df.to_dict('records'):
                    action = VisitorAction(sequence.id, row['recordTimestamp'], row['clientTimestamp'], row['button'], row['state'], row['x'], row['y'])
                    actions.append(action)
                db.session.bulk_save_objects(actions)
                db.session.commit()
            except Exception as e:
                print(e)
                return make_response(jsonify({'message': 'error'}), 500)
            return make_response(jsonify({'message': 'success'}), 200)
        
        @app.route('/deleteMouseData', methods=['GET'])
        def deleteMouseData():
            if not isRunningInCloud():
                self.deleteMouseData()
            return make_response(jsonify({'message': 'success'}), 200)

        return app