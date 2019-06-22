import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import io
from datetime import datetime
import tempfile

from statistics import mean
from itertools import combinations
from pydub import AudioSegment
from pydub import silence
from PIL import Image

# import thinkdsp
# import thinkplot
import librosa as lr

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool
from functools import partial



# support functions
def delete_files_in_directory(dirs):  # delete all files in passed dirs (list)
    import os
    from glob import glob

    [os.remove(f) for each_dir in dirs for f in glob(each_dir+'/*.*')]


def copy_files(s_dirs):  # copies files from source to dest - defaults to 100,000 files
    from glob import glob

    files=[]
    for dir, n_files in s_dirs:
        files.extend([file for i, file in enumerate(glob(dir+'/*.wav')) if i < n_files])
    return files


def append_df_to_csv(df, csv_file_path, sep=","):  # saves (or appends) df to csv
    import os

    if not os.path.isfile(csv_file_path):
        df.to_csv(csv_file_path, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csv_file_path, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csv_file_path, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csv_file_path, mode='a', index=False, sep=sep, header=False)


def detect_leading_silence(sound, silence_threshold=-30.0, chunk_size=10):
    # returns leading part of array which meats the silence threshold..  for back of file, reversed array is sent...
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


# process all the files

def process_files_for_trim(thresh1, thresh2, thumb_dim, pct_to_ignore_front, pct_to_ignore_rear, file):
    log_cols = ["File_Name", "Spoken_Nbr", "dBFS_mean", "dBFS_std", "dBFS_thresh", "org_dur", "trim_dur"]
    log = pd.DataFrame(columns=log_cols)
    sound = AudioSegment.from_file(file, format="wav")
    seconds_to_skip_front = len(sound) * pct_to_ignore_front  # trim x% off front of file - virtually noone talkes that fast and there is often initial microphone noise
    seconds_to_skip_rear = len(sound) * pct_to_ignore_rear  # trim x% off front of file - virtually noone talkes that fast and there is often initial microphone noise
    sound = sound[seconds_to_skip_front:len(sound)-seconds_to_skip_rear]  # trim sound
    # sound = sound[:len(sound)-seconds_to_skip]  # trim sound
    all_dBFS = []
    for i in range(len(sound)):
        all_dBFS.append([sound[i:i + 1].dBFS])

    # get the mean, stdev and determine threah (threah determins how silence is determined)
    dBFS_mean = np.ma.masked_invalid(all_dBFS).mean()
    dBFS_std = np.ma.masked_invalid(all_dBFS).std()
    dBFS_thresh = dBFS_mean + (dBFS_std * thresh1)
    dBFS_thresh2 = dBFS_mean + (dBFS_std * thresh2)

    start_trim = detect_leading_silence(sound, silence_threshold=dBFS_thresh, chunk_size=20)  # get leading silence
    end_trim = detect_leading_silence(sound.reverse(), silence_threshold=dBFS_thresh2, chunk_size=20)  # get trailing silence

    duration = len(sound)  # length of sound
    trimmed_sound = sound[start_trim:duration - end_trim]  # trim the sound

    tmphandle, temppath = tempfile.mkstemp()
    trimmed_sound.export(temppath, format="wav")  # Exports to a wav file in trimmed directory

    audio_trim, sfreq_trim = lr.load(temppath)
    time_trim = np.arange(0, len(audio_trim)) / sfreq_trim
    e_time_trim = max(time_trim, default=1)

    plt.style.use('default')
    fig, ax = plt.subplots()

    # plot trimmed file
    plt.plot(time_trim, audio_trim, "k", alpha=.7)
    plt.xlim([0, .3])
    plt.ylim([-.6, .6])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, dpi=100)
    # plt.show()
    plt.close()

    im = Image.open(buf).convert('L')  # convert to grayscale
    im.thumbnail(thumb_dim)
    im = np.array(im)
    im = im.flatten(order='C')
    # dBFS_images.append(im)

    # im.close()
    buf.close()
    os.close(tmphandle)

    # sys.exit()

    # set the beginning and ending silence to match length of original audio clip
    # silence_start = AudioSegment.silent(duration=start_trim + seconds_to_skip)  # make initial blank sound
    # silence_end = AudioSegment.silent(duration=end_trim)  # make ending blank sound

    # # combine audio files
    # newAudio1 = trimmed_sound + silence_end
    # newAudio1 = silence_start + newAudio1  # no sound should be the same length as the original sound
    #
    # # export file
    # tmphandle, temppath = tempfile.mkstemp()
    # newAudio1.export(temppath, format="wav")  # Exports to a wav file in the special directory (for graphing only)


    # update the log with misc info
    log_entry = pd.DataFrame([[file, file[:2], dBFS_mean, dBFS_std, dBFS_thresh, len(sound), len(trimmed_sound)]],
                             columns=log_cols)
    label = os.path.basename(file)[:2]
    # print(label)
    # log = log.append(log_entry)
    # return log, dBFS_images
    return label, im


def classifyx(X, y, thresh1, thresh2, pct_to_ignore_front, pct_to_ignore_rear, results_log_cols, results_log):

    # y = log['Spoken_Nbr']     # y contains label

    classifiers = [
        # KNeighborsClassifier(2, n_jobs = -1),
        # DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100, n_jobs=-1)
        # AdaBoostClassifier(),
        # GradientBoostingClassifier()
    ]

    skf = StratifiedKFold(n_splits=5)

    for clf in classifiers:
        start_time_1 = datetime.now()
        scores = cross_val_score(clf, X, y, scoring='accuracy', cv=skf)
        acc = mean(scores)
        name = clf.__class__.__name__
        end_time_1 = datetime.now()
        elapsed_time_1 = round((end_time_1 - start_time_1).total_seconds(),2)
        print(f'{name+":":<28} Score: {acc:<5.2f}  Time:{elapsed_time_1:6.2f} seconds')
        results_log_entry = pd.DataFrame([[name, acc, thresh1, thresh2, pct_to_ignore_front, pct_to_ignore_rear, elapsed_time_1]], columns=results_log_cols)
        results_log = results_log.append(results_log_entry)
        append_df_to_csv(results_log, 'results_log.csv', sep=",")
        results_log = results_log.iloc[:-1]

def get_source_files():
    # source directories
    org_dir = 'counting'  # these are the original files I created
    numbers1_dir = 'numbers'  # numbers from Matt
    numbers2_dir = 'morenumbers'  # more numbers from Matt

    dirs_to_process = []  # initialize array ultimately containing directory and # files to process
    dirs_to_process.append(['numbers', 1000])  # comment out for zero, change to 100 for 100 files
    dirs_to_process.append(['morenumbers', 1000])

    files = copy_files(dirs_to_process)  # populate array with files to process

    return files


def main():
    plt.style.use('ggplot')
    # print(np.__version__)
    plt.rcParams.update({'figure.max_open_warning': 0})  # limit warnings for multiple graphs

    files = get_source_files() # populate with files to analyze

    # create pandas df for tracking info
    log_cols=["File_Name", "Spoken_Nbr", "dBFS_mean", "dBFS_std", "dBFS_thresh", "org_dur", "trim_dur"]
    log = pd.DataFrame(columns=log_cols)

    # create pandas df from tracking classifer results
    results_log_cols = ["Classifier", "Accuracy", "Thresh1", "Thresh2", "Front_Ignore", "Rear_Ignore", 'Elapsed_Time']
    results_log = pd.DataFrame(columns=results_log_cols)

    # values_1 = np.array(np.linspace(.2, 3, 15)).round(2)
    # comb = combinations(values_1, 2)
    comb = [(1.6, 2.8)]
    thumb_dim = (64,64)

    # values_1 = np.array(np.linspace(.1, 25, .25)).round(2)
    # pct_to_ignore = combinations(values_1, 2)
    pct_to_ignore = [(.2,.2)]

    for thresh1, thresh2 in comb:
        for pct_to_ignore_front, pct_to_ignore_rear in pct_to_ignore:
            pool = Pool()
            func = partial(process_files_for_trim, thresh1, thresh2, thumb_dim, pct_to_ignore_front, pct_to_ignore_rear)
            label, dBFS_images = zip(*pool.map(func, files))
            pool.close()
            pool.join()

            label = np.array(label) # convert list to np.array
            dBFS_images =np.array(dBFS_images) #convert list to np.array

            classifyx(dBFS_images, label, thresh1, thresh2, pct_to_ignore_front, pct_to_ignore_rear, results_log_cols, results_log)


if __name__ == "__main__":
    main()


