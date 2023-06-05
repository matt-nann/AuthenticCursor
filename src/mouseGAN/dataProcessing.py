import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib
import matplotlib.colors as colors
import torch
import pickle

sqrt3 = np.sqrt(3)
sqrt5 = np.sqrt(5)

def wind_mouse(start_x, start_y, dest_x, dest_y, G_0=9, W_0=3, M_0=20, D_0=12, move_mouse=lambda x,y: None):
    # https://ben.land/post/2021/04/25/windmouse-human-mouse-movement/
    '''
    Simulates a mouse moving from start_x, start_y to dest_x, dest_y using the WindMouse algorithm. Used to generate fake training data.

    Calls the move_mouse kwarg with each new step.
    Released under the terms of the GPLv3 license.
    G_0 - magnitude of the gravitational fornce
    W_0 - magnitude of the wind force fluctuations
    M_0 - maximum step size (velocity clip threshold)
    D_0 - distance where wind behavior changes from random to damped
    '''
    current_x, current_y = start_x, start_y
    v_x = v_y = W_x = W_y = 0
    offsets = []
    # print(f"start: {start_x},{start_y} dest: {dest_x},{dest_y}, dist: {np.hypot(dest_x-start_x,dest_y-start_y)}")
    total_steps = 0
    total_pixel_distance = 0
    while (dist:=np.hypot(dest_x-start_x,dest_y-start_y)) >= 1:
        # print(f"dist: {dist}") 
        W_mag = min(W_0, dist)
        # print(f"dist: {dist}, W_mag: {W_mag}")
        if dist >= D_0:
            W_x = W_x/sqrt3 + (2*np.random.random()-1)*W_mag/sqrt5
            W_y = W_y/sqrt3 + (2*np.random.random()-1)*W_mag/sqrt5
        else:
            W_x /= sqrt3
            W_y /= sqrt3
            if M_0 < 3:
                M_0 = np.random.random()*3 + 3
            else:
                M_0 /= sqrt5
        # print(f"W_x: {W_x}, W_y: {W_y}")
        v_x += W_x + G_0*(dest_x-start_x)/dist
        v_y += W_y + G_0*(dest_y-start_y)/dist
        v_mag = np.hypot(v_x, v_y)
        if v_mag > M_0:
            # print(f"v_mag: {v_mag}, M_0: {M_0}")
            v_clip = M_0/2 + np.random.random()*M_0/2
            v_x = (v_x/v_mag) * v_clip
            v_y = (v_y/v_mag) * v_clip
        # print(f"v_x: {v_x}, v_y: {v_y}")
        start_x += v_x
        start_y += v_y
        move_x = start_x
        move_y = start_y
        if current_x != move_x or current_y != move_y:
            move_mouse(move_x - current_x, move_y - current_y)
            total_pixel_distance += np.hypot(move_x - current_x, move_y - current_y)
            total_steps += 1
            offsets.append((current_x, current_y))
            current_x, current_y = move_x, move_y
    return total_steps, total_pixel_distance, offsets

class MouseGAN_Data:
    dataLocal = 'data/local'
    trajColumns = ['dx','dy']
    allButtonColumns = ['left','top','width','height','start_x','start_y']
    targetColumns = ['width','height','start_x','start_y']
    FIXED_TIMESTEP = 0.008
    def __init__(self, USE_FAKE_DATA=False, 
                equal_length=True, lowerLimit = 50, upperLimit = 85):
        self.equal_length = equal_length
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.USE_FAKE_DATA = USE_FAKE_DATA
        self.shapes = []
        self.list_of_all_arrows = []

    def collectRawMouseTrajectories(self):
        from src.flaskServer.models import VisitorAction, MouseTrajectory, InputTarget
        from runFlask import app

        with app.app_context():
            query_results = VisitorAction.query.all()
            data = [result.__dict__ for result in query_results]
            df_moves = pd.DataFrame(data)
            df_moves = df_moves.drop(['_sa_instance_state'], axis=1)
            # df_moves = df_moves.drop(['state','button','recordTimestamp','id'], axis=1)
            df_moves = df_moves.drop(['recordTimestamp','id','state'], axis=1)
            sequence_id = list(df_moves['sequence_id'].unique().astype(int))
            sequence_id = [int(i) for i in sequence_id]
            query_results = MouseTrajectory.query.filter(MouseTrajectory.id.in_(sequence_id)).all()
            data = [result.__dict__ for result in query_results]
            df_trajectory = pd.DataFrame(data)
            df_trajectory = df_trajectory.drop(['_sa_instance_state'], axis=1)
            df_trajectory = df_trajectory.rename(columns={'id':'sequence_id'})
            # join the input target with the target id
            targetIds = df_trajectory['target_id'].unique().astype(int)
            targetIds = [int(i) for i in targetIds]
            buttonTargets = InputTarget.query.filter(InputTarget.id.in_(targetIds)).all()
            data = [result.__dict__ for result in buttonTargets]
            df_targets = pd.DataFrame(data)
            df_targets = df_targets.drop(['_sa_instance_state'], axis=1)
            df_targets = df_targets.rename(columns={'id':'target_id'})
            df_trajectory = df_trajectory.merge(df_targets, on='target_id')
            df_trajectory = df_trajectory.rename(columns={'x':'left', 'y':'top'})

        # display(df_moves.head(3))
        # display(df_trajectory.head(3))
        self.df_moves = df_moves
        self.df_trajectory = df_trajectory
        return df_moves, df_trajectory

    def loadFakeWindMouseData(self):
        with open(self.dataLocal + '/synthetic_trajectories.pkl', 'rb') as f:
            self.fake_trajectories = pickle.load(f)
        with open(self.dataLocal + '/synthetic_buttonTargets.pkl', 'rb') as f:
            self.fake_buttonTargets = pickle.load(f)

    def plotTrajectory(self, df_sequence, df_target, sequence_id, fig=None):
            if fig:
                self.fig = fig
            elif self.SHOW_ONE:
                self.fig = go.Figure()
                self.shapes, self.list_of_all_arrows = [], []
            u = df_sequence['x'].diff(1).fillna(0)
            v = df_sequence['y'].diff(1).fillna(0)
            self.fig.add_trace(go.Scatter(x=df_sequence['x'], y=df_sequence['y'],
                mode='lines+markers',
                marker=dict(
                            size=5, 
                            # symbol= "arrow-bar-up", angleref="previous",
                            # size=15,
                            # color='grey',),
                            color=df_sequence['velocity'], colorscale='Viridis', showscale=True, colorbar=dict(title="Velocity")),
                
                ))

            min_velocity, max_velocity = df_sequence['velocity'].min(), df_sequence['velocity'].max()
            cmap = matplotlib.colormaps['viridis']
            values = np.linspace(0, 1, 10000)
            colors_map = cmap(values)
            colorscale = [colors.rgb2hex(color) for color in colors_map]

            for i in range(0, len(df_sequence['x']), 1):  
                color_idx = (df_sequence['velocity'].iloc[i] - min_velocity) / (max_velocity - min_velocity)
                color_idx *= len(colorscale) - 1
                if pd.isna(color_idx):
                    color_idx = 0
                arrowcolor = colorscale[int(np.round(color_idx))]
                # print('x:', df_sequence['x'].iloc[i], 'y:', df_sequence['y'].iloc[i], 'u:', u.iloc[i], 'v:', v.iloc[i])
                arrow = go.layout.Annotation(dict(
                                    ax=df_sequence['x'].iloc[i],
                                    ay=df_sequence['y'].iloc[i],
                                    x=df_sequence['x'].iloc[i] + u.iloc[i],
                                    y=df_sequence['y'].iloc[i] + v.iloc[i],
                                    xref="x", yref="y", axref="x", ayref="y",
                                    showarrow=True,
                                    arrowhead=5, arrowsize=2, arrowwidth=1,
                                    arrowcolor=arrowcolor))
                self.list_of_all_arrows.append(arrow)

            # if left and top are not provided then the button is assumed to be centered already
            if 'left' not in df_target.columns:
                left, top = -df_target['width'].iloc[0]/2, -df_target['height'].iloc[0]/2
            else:
                left, top = df_target['left'].iloc[0], df_target['top'].iloc[0]
            width, height = df_target['width'].iloc[0], df_target['height'].iloc[0]
            x0, y0 = left, top
            x1, y1 = left + width, top + height
            square = go.layout.Shape(
                type='rect',
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color='black', width=2),
                fillcolor='rgba(0, 0, 255, 0.3)',
            )
            # add a text annoation to the rect
            text = go.layout.Annotation(dict(x=(x0+x1)/2,
                                            y=(y0+y1)/2,
                                            xref="x", yref="y",
                                            text='clicked button',
                                            showarrow=False,))
            self.fig.update_layout(
                shapes=[square],
            )
            if self.SHOW_ONE:
                minLimit = min(min(df_sequence['x']), min(df_sequence['y']))
                minLimit = min(x0, y0, minLimit)-10
                maxLimit = max(max(df_sequence['x']), max(df_sequence['y']))
                maxLimit = max(x1, y1, maxLimit)+10
                self.fig.update_layout(
                            title = 'Example Collected Mouse Trajectory',
                            # title='Sequence {}'.format(sequence_id),
                            annotations= self.list_of_all_arrows + [text],
                            shapes=[square],
                xaxis=dict(range=[minLimit, maxLimit], autorange=False),
                yaxis=dict(range=[minLimit, maxLimit], autorange=False),
                            xaxis_title='x', yaxis_title='y', width=500, height=500, margin=dict(l=0, r=0, b=0, t=30))
                self.fig.show()
            elif self.SHOW_ALL:
                self.shapes.append(square)

            # fig.show()
            return self.fig

    def remove_consecutive_zero_velocity_rows(self,df):
        """
        remove any trailing rows with zero movement that unnecessarily extend the trajectory
        """
        df_reversed = df.iloc[::-1]
        first_non_zero_index = df_reversed.loc[(df_reversed['dx'] != 0) | (df_reversed['dy'] != 0)].index[0]
        df = df.loc[:first_non_zero_index].copy()
        return df

    def regularize_TimeSeries(self,df, df_target):
        """
        needs raw_x, raw_y, timeDelta columns
        """
        total_time = df['timeDelta'].cumsum().iloc[-1]
        # Generate new time points
        new_time_points = np.arange(0, total_time, self.FIXED_TIMESTEP)
        new_time_points = np.append(new_time_points, new_time_points[-1]+ self.FIXED_TIMESTEP)
        df['cumulativeTime'] = df['timeDelta'].cumsum()
        df_reg = pd.DataFrame()
        # Interpolate raw_x and raw_y for the new time points
        for column in ['raw_x', 'raw_y']:
            # Create a new Series with cumulativeTime as the index
            s = pd.Series(df[column].values, index=df['cumulativeTime'])
            # Interpolate the Series at the new time points
            new_s = s.reindex(s.index.union(new_time_points)).interpolate(method='index').loc[new_time_points]
            df_reg[column] = new_s.values

        # df_reg['raw_x'].iloc[0] = df_target['start_x'].iloc[0]
        # df_reg['raw_y'].iloc[0] = df_target['start_y'].iloc[0]
        
        # display(new_df)
        df_reg['distance'] = np.sqrt(df_reg['raw_x'].diff(1)**2 + df_reg['raw_y'].diff(1)**2)
        df_reg['velocity'] = df_reg['distance'] / self.FIXED_TIMESTEP
        df_reg['time'] = new_time_points
        # the first row is where the trajectory starts, prior to this point there was no movement (assumption)
        df_reg['dx'] = df_reg['raw_x'].diff(1)
        df_reg['dy'] = df_reg['raw_y'].diff(1)
        df_reg['x'] = df_reg['raw_x']
        df_reg['y'] = df_reg['raw_y']

        for col in ['velocity','distance','dx','dy']:
            df_reg[col].iloc[0] = 0
        return df_reg
    
    def saveTrajData(self, df_regulSeq, df_target):
        self.input_trajectories.append(df_regulSeq[self.trajColumns].values)
        self.buttonTargets.append(df_target[self.targetColumns].values)
        for i, name in enumerate(self.trajColumns):
            self.trajectoryValues[i] += list(df_regulSeq[name].values)
        for i, name in enumerate(self.targetColumns):
            self.targetValues[i] += list(df_target[name].values)

    def convertToAbsolute(self, df_sequence, df_target):
        start_x = df_target['start_x'].iloc[0]
        start_y = df_target['start_y'].iloc[0]
        df_abs = df_sequence.copy()
        df_abs['temp_x'] = df_abs['dx']
        df_abs['temp_y'] = df_abs['dy']
        # absolute positioning, where origin is button target
        df_abs.at[0,'temp_x'] += start_x
        df_abs.at[0,'temp_y'] += start_y
        # to convert it into original raw screen pixels
        # df_abs['temp_x'].iloc[0] += left
        # # df_abs['temp_y'].iloc[0] += top
        if 'velocity' not in df_abs.columns:
            df_abs['velocity'] = np.sqrt(df_abs['dx']**2 + df_abs['dy']**2)
        df_abs['x'] = (df_abs['temp_x'] ).cumsum()
        df_abs['y'] = (df_abs['temp_y'] ).cumsum()
        return df_abs

    def processMouseData(self, SHOW_ALL=False, SHOW_ONE=False, num_sequences=10, samples=None):
        self.fig = go.Figure()
        self.shapes, self.list_of_all_arrows = [], []
        self.SHOW_ALL = SHOW_ALL
        self.SHOW_ONE = SHOW_ONE
        self.trajectoryValues = [[],[]]
        self.targetValues = [[],[],[],[]]
        counter = 0
        self.input_trajectories = []
        self.buttonTargets = []

        percentPrint = 0.1

        if self.USE_FAKE_DATA:
            # raw_x	raw_y velocity
            # width	height	start_x	start_y
            if samples is None:
                sampleIndexes = range(len(self.fake_trajectories))
            else:
                sampleIndexes = np.random.choice(len(self.fake_trajectories), samples, replace=False)
            for i in sampleIndexes:
                trajectory = self.fake_trajectories[i]
                buttonTarget = self.fake_buttonTargets[i]
                f_trajectory = trajectory.copy()
                # check if first rows of f_trajectory are duplicates
                if f_trajectory.iloc[0].equals(f_trajectory.iloc[1]):
                    f_trajectory = f_trajectory.iloc[1:].copy()
                if self.equal_length:
                    if len(f_trajectory) < self.lowerLimit:
                        continue
                    f_trajectory = f_trajectory.iloc[-self.lowerLimit:].copy()
                    new_start_x, new_start_y = f_trajectory['raw_x'].iloc[0], f_trajectory['raw_y'].iloc[0]
                    buttonTarget['start_x'] = new_start_x
                    buttonTarget['start_y'] = new_start_y
                f_trajectory['dx'] = f_trajectory['raw_x'].diff(1)
                f_trajectory['dy'] = f_trajectory['raw_y'].diff(1)
                for col in ['dx','dy']:
                    f_trajectory[col].iloc[0] = 0
                # keeping the first row but setting all the movements columns to zero
                f_trajectory.reset_index(inplace=True, drop=True)
                for col in ['velocity','distance','dx','dy','timeDelta']:
                    f_trajectory.at[0, col] = 0
                # df_cleanedSeq = self.remove_consecutive_zero_velocity_rows(f_trajectory)
                f_trajectory = f_trajectory[self.trajColumns]
                self.saveTrajData(f_trajectory, buttonTarget)
                counter += 1
                if counter % int(len(self.fake_trajectories)*percentPrint) == 0:
                    print('processed fake data', counter, ' out of ', len(self.fake_trajectories))

        else:
            # filter out some they are roughly all the same length to start
            if self.equal_length:
                df_moves = self.df_moves
                df_moves = df_moves[df_moves['sequence_id'].isin(df_moves['sequence_id'].value_counts()[(df_moves['sequence_id'].value_counts() > self.lowerLimit) & (df_moves['sequence_id'].value_counts() < self.upperLimit)].index)]

            for sequence_id, df_sequence in df_moves.groupby('sequence_id'):
                df_sequence = df_sequence.sort_values(by=['clientTimestamp'])
                df_sequence.drop(['sequence_id'], axis=1, inplace=True)
                df_sequence = df_sequence[['clientTimestamp','x','y']].copy()
                df_sequence['clientTimestamp'] -= df_sequence['clientTimestamp'].iloc[0]
                df_sequence['clientTimestamp'] /= 1000
                df_sequence['timeDelta'] = df_sequence['clientTimestamp'].diff(1) # in seconds
                df_sequence['distance'] = np.sqrt(df_sequence['x'].diff(1)**2 + df_sequence['y'].diff(1)**2)
                df_sequence['velocity'] = df_sequence['distance'] / df_sequence['timeDelta']

                start_x, start_y = df_sequence['x'].iloc[0], df_sequence['y'].iloc[0]
                # end_x, end_y = df_sequence['x'].iloc[-1], df_sequence['y'].iloc[-1]

                df_target = self.df_trajectory[self.df_trajectory['sequence_id'] == sequence_id].copy()
                width = df_target['width'].iloc[0]
                height = df_target['height'].iloc[0]
                left = df_target['left'].iloc[0]
                top = df_target['top'].iloc[0]
                df_target['start_x'] = start_x - left - width/2
                df_target['start_y'] = start_y - top - height/2
                # display(df_target)
                # print(width, height, left, top, start_x, start_y)
                df_target = df_target[self.allButtonColumns].copy()
                
                df_sequence['dx'] = df_sequence['x'].diff(1)
                df_sequence['dy'] = df_sequence['y'].diff(1)

                df_sequence['raw_x'] = df_sequence['x'] - left - width/2
                df_sequence['raw_y'] = df_sequence['y'] - top - height/2

                df_sequence['distance'] = np.sqrt(df_sequence['dx']**2 + df_sequence['dy']**2)
                df_sequence['velocity'] = df_sequence['distance'] / df_sequence['timeDelta']

                # TODO eventually remove making them all same size to start
                if self.equal_length:
                    df_EquSeq = df_sequence.iloc[-self.lowerLimit:].copy()
                    new_start_x, new_start_y = df_EquSeq['x'].iloc[0], df_EquSeq['y'].iloc[0]
                    df_target['start_x'] = new_start_x - left - width/2
                    df_target['start_y'] = new_start_y - top - height/2

                # keeping the first row but setting all the movements columns to zero
                # df_sequence = df_sequence.iloc[1:]
                df_EquSeq.reset_index(inplace=True, drop=True)
                for col in ['velocity','distance','dx','dy','timeDelta']:
                    df_EquSeq.at[0, col] = 0

                df_cleanedSeq = self.remove_consecutive_zero_velocity_rows(df_EquSeq)
                # display(df_cleanedSeq)
                df_regulSeq = self.regularize_TimeSeries(df_cleanedSeq, df_target)
                self.saveTrajData(df_EquSeq, df_target)
                # display(df_regulSeq)
                
                if SHOW_ONE or SHOW_ALL:
                    self.plotTrajectory(df_regulSeq, df_target[self.targetColumns], sequence_id)
                if SHOW_ONE and counter > num_sequences:
                    break
                counter += 1

                df_sequence = df_sequence[self.trajColumns]
                df_target = df_target[self.targetColumns]

            if SHOW_ALL:
                self.fig.update_layout(title='Sequence {}'.format(sequence_id),
                                annotations=self.list_of_all_arrows,
                                shapes=self.shapes,
                    # xaxis=dict(range=[0, 2000], autorange=False),
                    # yaxis=dict(range=[0, 1000], autorange=False),
                                xaxis_title='x', yaxis_title='y', width=1000, height=1000, margin=dict(l=0, r=0, b=0, t=30))
                self.fig.show()

                # display(df_target)
                # display(df_sequence.head(3))
            # print(df_sequence)
            # print(df_target)

            # chop off 1 if not even
            if len(self.input_trajectories) % 2 != 0:
                self.input_trajectories = self.input_trajectories[:-1]
                self.buttonTargets = self.buttonTargets[:-1]
        
        norm_input_trajectories, norm_buttonTargets = self.normalize(self.input_trajectories, self.buttonTargets)
        print('Number of sequences:', len(self.input_trajectories))
        return norm_input_trajectories, norm_buttonTargets
        # return df_regulSeq, df_target, df_target['start_x'].iloc[0], df_target['start_y'].iloc[0], left, top
    
    def _calcDistributionMetrics(self):
        # every column is getting normalized individually
        self.trajectoryValues = np.array(self.trajectoryValues)
        self.targetValues = np.array(self.targetValues)
        self.mean_traj = self.trajectoryValues.mean(axis=1)
        self.std_traj = self.trajectoryValues.std(axis=1)
        self.mean_button = self.targetValues.mean(axis=1)
        self.std_button = self.targetValues.std(axis=1)

    def normalize(self, input_trajectories, buttonTargets):
        self._calcDistributionMetrics()
        norm_input_trajectories = []
        norm_buttonTargets = []
        for i in range(len(input_trajectories)):
            norm_traj = (input_trajectories[i] - self.mean_traj) / self.std_traj
            norm_input_trajectories.append(torch.tensor(norm_traj, dtype=torch.float32))
            if np.isnan(norm_input_trajectories[-1]).any():
                raise ValueError('nan found in norm_input_trajectories')
            norm_target = (buttonTargets[i] - self.mean_button) / self.std_button
            norm_buttonTargets.append(torch.tensor(norm_target, dtype=torch.float32))
        return norm_input_trajectories, norm_buttonTargets
    
    def denormalize(self, norm_input_trajectories, norm_buttonTargets):
        input_trajectories = []
        buttonTargets = []
        for i in range(len(norm_input_trajectories)):
            traj = norm_input_trajectories[i] * self.std_traj + self.mean_traj
            input_trajectories.append(traj)
            target = norm_buttonTargets[i] * self.std_button + self.mean_button
            buttonTargets.append(target)
        return input_trajectories, buttonTargets
    
    def createFakeWindMouseDataset(self, save=False):
        # fig = go.Figure()
        AXIAL_RESOLUTION = 2000
        theta = np.linspace(0, 2*np.pi, AXIAL_RESOLUTION)
        low_radius = 100
        high_radius = 1000
        TOTAL_SAMPLES = 50000
        maxRadius = 0
        max_width = 200
        min_width = 50
        max_height = 100
        min_height = 25

        trajectories = []
        buttonTargets = []

        for i in range(TOTAL_SAMPLES // AXIAL_RESOLUTION):
            radius = np.random.random() * (high_radius - low_radius) + low_radius
            maxRadius = max(maxRadius, radius)
            x = radius * np.cos(theta) 
            y = radius * np.sin(theta)
            counter = 0
            width = np.random.random() * (max_width - min_width) + min_width
            height = np.random.random() * (max_height - min_height) + min_height
            for (x1,y1) in zip(x,y):
                dest_x = np.random.random() * width - width/2
                dest_y = np.random.random() * height - height/2
                total_steps, total_pixel_distance, absolute = wind_mouse(x1, y1,dest_x, dest_y, G_0=6, W_0=10, M_0=12, D_0=12)
                absolute = [(x1,y1)] + absolute
                absolute.append((dest_x, dest_y))
                absolute = np.array(absolute)
                df_trajecttory = pd.DataFrame(absolute, columns=['raw_x', 'raw_y'])
                df_trajecttory['velocity'] = np.concatenate(([0],np.hypot(np.diff(absolute[:,0]), np.diff(absolute[:,1]))))
                # textArray = ['x:' + str(round(x1,1)) + ' y:' + str(round(y1,1)) + ' xf: ' + str(round(dest_x,1)) + ' yf: ' + str(round(dest_y,1)) + ' v:' + str(round(vel)) for vel in velocity]
                target = pd.DataFrame([[width, height, x1, y1]],
                                    columns=['width', 'height','start_x', 'start_y'])
                trajectories.append(df_trajecttory)
                buttonTargets.append(target)
                # if counter == 0:
                # fig.add_trace(go.Scatter(x=df_trajecttory['raw_x'], y=df_trajecttory['raw_y'], mode='lines+markers',
                #     # text = ,
                #     line=dict(width=1, color='grey'),
                #     marker=dict(
                #             size=5, 
                #             color=df_trajecttory['velocity'],colorscale='Viridis',showscale=True,)
                #             ))
                counter += 1
                
        # fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=2)))

        # fig.update_layout(
        #     width=800,
        #     height=800,
        #     xaxis=dict(
        #         range=[-maxRadius*1.1, maxRadius*1.1],)
        #     ,yaxis=dict(
        #         range=[-maxRadius*1.1, maxRadius*1.1],)
        # )
        # fig.show()
        if save:
            with open(self.dataLocal + '/synthetic_trajectories.pkl', 'wb') as f:
                pickle.dump(trajectories, f)
            with open(self.dataLocal + '/synthetic_buttonTargets.pkl', 'wb') as f:
                pickle.dump(buttonTargets, f)
        else:
            self.fake_buttonTargets = buttonTargets
            self.fake_trajectories = trajectories

    def plotFakeSamples(self):
        # create a sample of indexes 
        trajectories = self.fake_trajectories
        buttonTargets = self.fake_buttonTargets
        idx = np.random.choice(np.arange(len(trajectories)), 10, replace=False)

        fig = go.Figure()
        shapes=[]
        for i in idx:
            trajectory = trajectories[i]
            button = buttonTargets[i]
            fig.add_trace(go.Scatter(x=trajectory['raw_x'], y=trajectory['raw_y'], mode='lines+markers',
                # text = ,
                line=dict(width=1, color='grey'),
                marker=dict(
                        size=5, 
                        color=trajectory['velocity'],colorscale='Viridis',showscale=True,)
                        ))
            fig.add_trace(go.Scatter(x=button['start_x'], y=button['start_y'], mode='markers', marker=dict(size=2)))
            
            x0 = - button['width'][0]/2
            y0 = - button['height'][0]/2
            x1 = button['width'][0]/2
            y1 = button['height'][0]/2
            square = go.layout.Shape(
                type='rect',
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color='black', width=2),
                # fillcolor='rgba(0, 0, 255, 0.3)',
            )
            # shapes.append(square)

        fig.update_layout(
            xaxis=dict(
                range=[-800,800],)
            ,yaxis=dict(
                range=[-800,800],)
            ,
            shapes=shapes,
            width=800,
            height=800,
        )
        fig.show()

    
# x_center = left + df_target['width'].iloc[0]/2
# y_center = top + df_target['height'].iloc[0]/2
# print(x_center, y_center)
# # for creating the original raw data again
# print(start_x, start_y, left, top)
# print(start_x + left, start_y + top)
# df_sequence = df_sequence.copy()
# df_sequence['cum_x'] = 0
# df_sequence['cum_y'] = 0
# df_sequence['temp_x'] = df_sequence['dx']
# df_sequence['temp_y'] = df_sequence['dy']
# # absolute positioning, where origin is button target
# df_sequence['temp_x'].iloc[0] += start_x
# df_sequence['temp_y'].iloc[0] += start_y
# # to convert it into original raw screen pixels
# df_sequence['temp_x'].iloc[0] += left
# df_sequence['temp_y'].iloc[0] += top
# df_sequence['cum_x'] = (df_sequence['temp_x'] ).cumsum()
# df_sequence['cum_y'] = (df_sequence['temp_y'] ).cumsum()