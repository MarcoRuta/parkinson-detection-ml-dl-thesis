import numpy as np
import pandas as pd
import os
import seaborn as sns
from math import sqrt

header_row = ["X", "Y", "Z", "Pressure", "GripAngle", "Timestamp", "Test_ID"]

control_data_path = 'E:/Desktop/Parkinson_py/dataset/numerical_dataset/hw_dataset/control'
parkinson_data_path = 'E:/Desktop/Parkinson_py/dataset/numerical_dataset/hw_dataset/parkinson'

control_file_list = [os.path.join( control_data_path, x ) for x in os.listdir( control_data_path )]
parkinson_file_list = [os.path.join( parkinson_data_path, x ) for x in os.listdir( parkinson_data_path )]


# conta il numero di strokes
def get_no_strokes(df):
    pressure_data = df['Pressure'].to_numpy()
    on_surface = (pressure_data > 600).astype(int)
    return ((np.roll(on_surface, 1) - on_surface) != 0).astype(int).sum()


# calcola e ritorna Vel, magnitude, timestamp_diff, horz_Vel, vert_Vel, magnitude_vel, magnitude_horz_vel, magnitude_vert_vel
def find_velocity(f):
    data_pat = f
    Vel = []
    horz_Vel = []
    horz_vel_mag = []
    vert_vel_mag = []
    vert_Vel = []
    magnitude = []
    timestamp_diff = []

    t = 0
    for i in range( len( data_pat ) - 2 ):
        if t + 10 <= len( data_pat ) - 1:
            Vel.append( ((data_pat['X'].to_numpy()[t + 10] - data_pat['X'].to_numpy()[t]) / (
                    data_pat['Timestamp'].to_numpy()[t + 10] - data_pat['Timestamp'].to_numpy()[t]),
                         (data_pat['Y'].to_numpy()[t + 10] - data_pat['Y'].to_numpy()[t]) / (
                                 data_pat['Timestamp'].to_numpy()[t + 10] - data_pat['Timestamp'].to_numpy()[
                             t])) )
            horz_Vel.append( (data_pat['X'].to_numpy()[t + 10] - data_pat['X'].to_numpy()[t]) / (
                    data_pat['Timestamp'].to_numpy()[t + 10] - data_pat['Timestamp'].to_numpy()[t]) )

            vert_Vel.append( (data_pat['Y'].to_numpy()[t + 10] - data_pat['Y'].to_numpy()[t]) / (
                    data_pat['Timestamp'].to_numpy()[t + 10] - data_pat['Timestamp'].to_numpy()[t]) )
            magnitude.append( sqrt( ((data_pat['X'].to_numpy()[t + 10] - data_pat['X'].to_numpy()[t]) / (
                    data_pat['Timestamp'].to_numpy()[t + 10] - data_pat['Timestamp'].to_numpy()[t])) ** 2 + (((
                                                                                                                      data_pat[
                                                                                                                          'Y'].to_numpy()[
                                                                                                                          t + 10] -
                                                                                                                      data_pat[
                                                                                                                          'Y'].to_numpy()[
                                                                                                                          t]) / (
                                                                                                                      data_pat[
                                                                                                                          'Timestamp'].to_numpy()[
                                                                                                                          t + 10] -
                                                                                                                      data_pat[
                                                                                                                          'Timestamp'].to_numpy()[
                                                                                                                          t])) ** 2) ) )
            timestamp_diff.append( data_pat['Timestamp'].to_numpy()[t + 10] - data_pat['Timestamp'].to_numpy()[t] )
            horz_vel_mag.append( abs( horz_Vel[len( horz_Vel ) - 1] ) )
            vert_vel_mag.append( abs( vert_Vel[len( vert_Vel ) - 1] ) )
            t = t + 10
        else:
            break
    magnitude_vel = np.mean( magnitude )
    magnitude_horz_vel = np.mean( horz_vel_mag )
    magnitude_vert_vel = np.mean( vert_vel_mag )
    return Vel, magnitude, timestamp_diff, horz_Vel, vert_Vel, magnitude_vel, magnitude_horz_vel, magnitude_vert_vel


# calcola e ritorna  accl, magnitude, horz_Accl, vert_Accl, timestamp_diff, magnitude_acc, magnitude_horz_acc, magnitude_vert_acc
def find_acceleration(f):
    '''
    change in direction and its velocity

    '''
    Vel, magnitude, timestamp_diff, horz_Vel, vert_Vel, magnitude_vel, magnitude_horz_vel, magnitude_vert_vel = find_velocity(
        f )
    accl = []
    horz_Accl = []
    vert_Accl = []
    magnitude = []
    horz_acc_mag = []
    vert_acc_mag = []
    for i in range( len( Vel ) - 2 ):
        accl.append(
            ((Vel[i + 1][0] - Vel[i][0]) / timestamp_diff[i], (Vel[i + 1][1] - Vel[i][1]) / timestamp_diff[i]) )
        horz_Accl.append( (horz_Vel[i + 1] - horz_Vel[i]) / timestamp_diff[i] )
        vert_Accl.append( (vert_Vel[i + 1] - vert_Vel[i]) / timestamp_diff[i] )
        horz_acc_mag.append( abs( horz_Accl[len( horz_Accl ) - 1] ) )
        vert_acc_mag.append( abs( vert_Accl[len( vert_Accl ) - 1] ) )
        magnitude.append( sqrt( ((Vel[i + 1][0] - Vel[i][0]) / timestamp_diff[i]) ** 2 + (
                (Vel[i + 1][1] - Vel[i][1]) / timestamp_diff[i]) ** 2 ) )

    magnitude_acc = np.mean( magnitude )
    magnitude_horz_acc = np.mean( horz_acc_mag )
    magnitude_vert_acc = np.mean( vert_acc_mag )
    return accl, magnitude, horz_Accl, vert_Accl, timestamp_diff, magnitude_acc, magnitude_horz_acc, magnitude_vert_acc


# calcola e ritorna jerk, magnitude, hrz_jerk, vert_jerk, timestamp_diff, magnitude_jerk, magnitude_horz_jerk, magnitude_vert_jerk
def find_jerk(f):
    accl, magnitude, horz_Accl, vert_Accl, timestamp_diff, magnitude_acc, magnitude_horz_acc, magnitude_vert_acc = find_acceleration(
        f )
    jerk = []
    hrz_jerk = []
    vert_jerk = []
    magnitude = []
    horz_jerk_mag = []
    vert_jerk_mag = []

    for i in range( len( accl ) - 2 ):
        jerk.append(
            ((accl[i + 1][0] - accl[i][0]) / timestamp_diff[i], (accl[i + 1][1] - accl[i][1]) / timestamp_diff[i]) )
        hrz_jerk.append( (horz_Accl[i + 1] - horz_Accl[i]) / timestamp_diff[i] )
        vert_jerk.append( (vert_Accl[i + 1] - vert_Accl[i]) / timestamp_diff[i] )
        horz_jerk_mag.append( abs( hrz_jerk[len( hrz_jerk ) - 1] ) )
        vert_jerk_mag.append( abs( vert_jerk[len( vert_jerk ) - 1] ) )
        magnitude.append( sqrt( ((accl[i + 1][0] - accl[i][0]) / timestamp_diff[i]) ** 2 + (
                (accl[i + 1][1] - accl[i][1]) / timestamp_diff[i]) ** 2 ) )

    magnitude_jerk = np.mean( magnitude )
    magnitude_horz_jerk = np.mean( horz_jerk_mag )
    magnitude_vert_jerk = np.mean( vert_jerk_mag )
    return jerk, magnitude, hrz_jerk, vert_jerk, timestamp_diff, magnitude_jerk, magnitude_horz_jerk, magnitude_vert_jerk


# calcola e ritorna il numero di variazioni di accellerazione per semicerchio
def NCA_per_halfcircle(f):
    data_pat = f
    Vel, magnitude, timestamp_diff, horz_Vel, vert_Vel, magnitude_vel, magnitude_horz_vel, magnitude_vert_vel = find_velocity(
        f )
    accl = []
    nca = []
    temp_nca = 0
    basex = data_pat['X'].to_numpy()[0]
    for i in range( len( Vel ) - 2 ):
        if data_pat['X'].to_numpy()[i] == basex:
            nca.append( temp_nca )
            # print ('tempNCa::',temp_nca)
            temp_nca = 0
            continue

        accl.append(
            ((Vel[i + 1][0] - Vel[i][0]) / timestamp_diff[i], (Vel[i + 1][1] - Vel[i][1]) / timestamp_diff[i]) )
        if accl[len( accl ) - 1] != (0, 0):
            temp_nca += 1
    nca.append( temp_nca )
    nca = list( filter( (2).__ne__, nca ) )
    nca_Val = np.sum( nca ) / np.count_nonzero( nca )
    return nca, nca_Val


# calcola e ritorna il numero di variazioni di velocità per semicerchio
def NCV_per_halfcircle(f):
    data_pat = f
    Vel = []
    ncv = []
    temp_ncv = 0
    basex = data_pat['X'].to_numpy()[0]
    for i in range( len( data_pat ) - 2 ):
        if data_pat['X'].to_numpy()[i] == basex:
            ncv.append( temp_ncv )
            temp_ncv = 0
            continue

        Vel.append( ((data_pat['X'].to_numpy()[i + 1] - data_pat['X'].to_numpy()[i]) / (
                data_pat['Timestamp'].to_numpy()[i + 1] - data_pat['Timestamp'].to_numpy()[i]),
                     (data_pat['Y'].to_numpy()[i + 1] - data_pat['Y'].to_numpy()[i]) / (
                             data_pat['Timestamp'].to_numpy()[i + 1] - data_pat['Timestamp'].to_numpy()[i])) )
        if Vel[len( Vel ) - 1] != (0, 0):
            temp_ncv += 1
    ncv.append( temp_ncv )
    # ncv = list(filter((2).__ne__, ncv))
    ncv_Val = np.sum( ncv ) / np.count_nonzero( ncv )
    return ncv, ncv_Val


# calcola e ritorna la velocità in un punto
def get_speed(df):
    total_dist = 0
    duration = df['Timestamp'].to_numpy()[-1]
    coords = df[['X', 'Y', 'Z']].to_numpy()
    for i in range( 10, df.shape[0] ):
        temp = np.linalg.norm( coords[i, :] - coords[i - 10, :] )
        total_dist += temp
    speed = total_dist / duration
    return speed


# calcola e ritorna il tempo in cui la penna non è stata in contatto con il supporto
def get_in_air_time(data):
    data = data['Pressure'].to_numpy()
    return (data < 600).astype( int ).sum()


# calcola e ritorna il tempo in cui la penna non è in contatto con il supporto
def get_on_surface_time(data):
    data = data['Pressure'].to_numpy()
    return (data > 600).astype( int ).sum()


# calcola e ritorna le seguenti features ['no_strokes_st', 'no_strokes_dy', 'speed_st', 'speed_dy', 'magnitude_vel_st',
#                     'magnitude_horz_vel_st', 'magnitude_vert_vel_st', 'magnitude_vel_dy', 'magnitude_horz_vel_dy',
#                     'magnitude_vert_vel_dy', 'magnitude_acc_st', 'magnitude_horz_acc_st', 'magnitude_vert_acc_st',
#                     'magnitude_acc_dy', 'magnitude_horz_acc_dy', 'magnitude_vert_acc_dy', 'magnitude_jerk_st',
#                     'magnitude_horz_jerk_st', 'magnitude_vert_jerk_st', 'magnitude_jerk_dy', 'magnitude_horz_jerk_dy',
#                     'magnitude_vert_jerk_dy', 'ncv_st', 'ncv_dy', 'nca_st', 'nca_dy', 'in_air_stcp', 'on_surface_st',
#                     'on_surface_dy', 'target']
# come parametri ha il file da cui estrarre le features e 1 o 0 se test PD o Control
def get_features(f, parkinson_target):
    global header_row
    df = pd.read_csv( f, sep=';', header=None, names=header_row )

    df_static = df[df["Test_ID"] == 0]  # static test
    df_dynamic = df[df["Test_ID"] == 1]  # dynamic test

    initial_timestamp = df['Timestamp'][0]
    df['Timestamp'] = df['Timestamp'] - initial_timestamp  # offset timestamps


    data_point = []
    data_point.append( get_no_strokes( df_static ) if df_static.shape[0] else 0 )  # no. of strokes for static test
    data_point.append( get_no_strokes( df_dynamic ) if df_dynamic.shape[0] else 0 )  # no. of strokes for dynamic test
    data_point.append( get_speed( df_static ) if df_static.shape[0] else 0 )  # speed for static test
    data_point.append( get_speed( df_dynamic ) if df_dynamic.shape[0] else 0 )  # speed for dynamic test

    # magnitudes of velocity, horizontal velocity e vertical velocity static test
    Vel, magnitude, timestamp_diff, horz_Vel, vert_Vel, magnitude_vel, magnitude_horz_vel, magnitude_vert_vel = find_velocity(
        df ) if df_static.shape[0] else (0, 0, 0, 0, 0, 0, 0, 0)
    data_point.extend( [magnitude_vel, magnitude_horz_vel, magnitude_vert_vel] )

    # magnitudes of velocity, horizontal velocity e vertical velocity dynamic test
    Vel, magnitude, timestamp_diff, horz_Vel, vert_Vel, magnitude_vel, magnitude_horz_vel, magnitude_vert_vel = find_velocity(
        df_dynamic ) if df_dynamic.shape[0] else (0, 0, 0, 0, 0, 0, 0, 0)
    data_point.extend( [magnitude_vel, magnitude_horz_vel, magnitude_vert_vel] )

    # magnitudes of acceleration, horizontal acceleration e vertical acceleration static test
    accl, magnitude, horz_Accl, vert_Accl, timestamp_diff, magnitude_acc, magnitude_horz_acc, magnitude_vert_acc = find_acceleration(
        df_static ) if df_static.shape[0] else (0, 0, 0, 0, 0, 0, 0, 0)
    data_point.extend( [magnitude_acc, magnitude_horz_acc, magnitude_vert_acc] )

    # magnitudes of acceleration, horizontal acceleration e vertical acceleration dynamic test
    accl, magnitude, horz_Accl, vert_Accl, timestamp_diff, magnitude_acc, magnitude_horz_acc, magnitude_vert_acc = find_acceleration(
        df_dynamic ) if df_dynamic.shape[0] else (0, 0, 0, 0, 0, 0, 0, 0)
    data_point.extend( [magnitude_acc, magnitude_horz_acc, magnitude_vert_acc] )

    # magnitudes of jerk, horizontal jerk e vertical jerk static test
    jerk, magnitude, hrz_jerk, vert_jerk, timestamp_diff, magnitude_jerk, magnitude_horz_jerk, magnitude_vert_jerk = find_jerk(
        df_static ) if df_static.shape[0] else (0, 0, 0, 0, 0, 0, 0, 0)
    data_point.extend( [magnitude_jerk, magnitude_horz_jerk, magnitude_vert_jerk] )

    # magnitudes of jerk, horizontal jerk e vertical jerk dynamic test
    jerk, magnitude, hrz_jerk, vert_jerk, timestamp_diff, magnitude_jerk, magnitude_horz_jerk, magnitude_vert_jerk = find_jerk(
        df_dynamic ) if df_dynamic.shape[0] else (0, 0, 0, 0, 0, 0, 0, 0)
    data_point.extend( [magnitude_jerk, magnitude_horz_jerk, magnitude_vert_jerk] )

    # NCV static test
    ncv, ncv_Val = NCV_per_halfcircle( df_static ) if df_static.shape[0] else (0, 0)
    data_point.append( ncv_Val )

    # NCV dynamic test
    ncv, ncv_Val = NCV_per_halfcircle( df_dynamic ) if df_dynamic.shape[0] else (0, 0)
    data_point.append( ncv_Val )

    # NCA static test
    nca, nca_Val = NCA_per_halfcircle( df_static ) if df_static.shape[0] else (0, 0)
    data_point.append( nca_Val )

    # NCA dynamic test
    nca, nca_Val = NCA_per_halfcircle( df_dynamic ) if df_dynamic.shape[0] else (0, 0)
    data_point.append( nca_Val )


    # on surface time static test
    data_point.append( get_on_surface_time( df_static ) if df_static.shape[0] else 0 )

    # on surface time dynamic test
    data_point.append( get_on_surface_time( df_dynamic ) if df_dynamic.shape[0] else 0 )

    # traget. 1 parkinson. 0 control.
    data_point.append( parkinson_target )

    return data_point


# i dati presenti nelle righe tra 13 e 18 (ho eseguito data check manuale dato che sono 78 istanze) non sono veritieri
# per non eliminare le istanze ho deciso di modificare i campi a 0 (dove lo 0 non è possibile) con la media della colonna
def data_cleaning(df):
    features = list( df.columns.values )
    for n in [13, 14, 15, 16, 17, 18]:
        for f in features:
            if int( df.loc[n][[f]] ) == 0 and str( f ) != 'target' and str( f ) != 'no_strokes_st' and str(
                    f ) != 'no_strokes_dy':
                df.loc[n][[f]] = df[[f]].mean()
    return df


if __name__ == '__main__':

    raw = []

    for x in parkinson_file_list:
        raw.append( get_features( x, 1 ) )

    for x in control_file_list:
        raw.append( get_features( x, 0 ) )

    raw = np.array( raw )

    features_headers = ['no_strokes_st', 'no_strokes_dy', 'speed_st', 'speed_dy', 'magnitude_vel_st',
                        'magnitude_horz_vel_st', 'magnitude_vert_vel_st', 'magnitude_vel_dy', 'magnitude_horz_vel_dy',
                        'magnitude_vert_vel_dy', 'magnitude_acc_st', 'magnitude_horz_acc_st', 'magnitude_vert_acc_st',
                        'magnitude_acc_dy', 'magnitude_horz_acc_dy', 'magnitude_vert_acc_dy', 'magnitude_jerk_st',
                        'magnitude_horz_jerk_st', 'magnitude_vert_jerk_st', 'magnitude_jerk_dy',
                        'magnitude_horz_jerk_dy',
                        'magnitude_vert_jerk_dy', 'ncv_st', 'ncv_dy', 'nca_st', 'nca_dy',
                        'on_surface_st',
                        'on_surface_dy', 'target']

    data = pd.DataFrame( raw )
    data.columns = features_headers

    data = data_cleaning( data )

    # salvare in \dataset\numerical_dataset
    data.to_csv( r'E:\Desktop\Parkinson_py\dataset\numerical_dataset\extracted_data.csv', index=False )
