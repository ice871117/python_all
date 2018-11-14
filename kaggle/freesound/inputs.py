"""Input pipeline for kaggle free-sound audio tag"""

import os
import random
import subprocess

import numpy as np
import pandas as pd
import librosa

# All input clips use a 44.1 kHz sample rate.
SAMPLE_RATE = 22050


class Input:
    """
    handles the file preparation and the transformation from audio sample to batch log mel data
    """
    __audio_path = ""
    __train_file_path = ""
    __shuffle = False
    __loop = False
    __temp_file_path = ""
    __total_files = ""
    __rand_arr = []
    __file_name_to_label_map = dict()
    __label_to_index_map = dict()
    __category_num = 0
    __n_mfcc = 30
    __fixed_sample = 100
    __speed = None
    __current_file_index = 0
    __last_read_file_name = ""

    def __init__(self, audio_path="",
                 train_file_path="",
                 shuffle=False,
                 loop=False,
                 temp_file_path="",
                 n_mfcc=30,
                 fixed_sample=100):
        self.__audio_path = audio_path
        self.__train_file_path = train_file_path
        self.__shuffle = shuffle
        self.__loop = loop
        self.__temp_file_path = temp_file_path
        self.__n_mfcc = n_mfcc
        self.__fixed_sample = fixed_sample
        # init start
        self.do_init()

    def do_init(self):
        self.__total_files = len(os.listdir(self.__audio_path))
        if self.__shuffle:
            self.__rand_arr = self.__get_random_arr(self.__total_files)
        for root, dirs, files in os.walk(self.__audio_path):
            for fileName in files:
                if fileName.find("DS_Store") >= 0:
                    os.remove(os.path.join(root, fileName))
        if self.__train_file_path:
            self.__init_train_category(self.__train_file_path)

    def __init_train_category(self, category_csv):
        all_data = pd.read_csv(category_csv)
        row_num = all_data.shape[0]
        for i in range(0, row_num):
            file_name = all_data.loc[i, 'fname']
            label = all_data.loc[i, 'label'].strip()
            self.__file_name_to_label_map[file_name] = label
            if label not in self.__label_to_index_map:
                self.__label_to_index_map[label] = len(self.__label_to_index_map)
        self.__category_num = len(self.__label_to_index_map)
        print("init category : ", self.__category_num)
        print("all categories : ", self.__label_to_index_map)

    def get_total_files(self):
        return self.__total_files

    def get_category_num(self):
        return self.__category_num

    def get_last_read_file_name(self):
        return self.__last_read_file_name

    def find_label_by_index(self, index):
        """
        find the corresponding label for the given index
        :param index: the index according to the order of occurrence
        :return: the label if found, None otherwise
        """
        for k, v in self.__label_to_index_map.items():
            if v == index:
                return k
        return None

    def next(self, batch_size):
        """
        :param batch_size: how many rows of data you want to sample
        :return:
                (data, mark)
                data: the np.array in size [batchSize, dataSize] if left files are enough
                or return [leftFileNum, dataSize] or None if no file is left
                mark: the np.array in size [batchSize, categoryNum] one-hot form
        """
        for root, dirs, files in os.walk(self.__audio_path):
            data = np.zeros((batch_size, self.__n_mfcc, self.__fixed_sample), np.float32)
            mark = np.zeros((batch_size, self.__category_num), np.int32)
            j = 0
            try:
                for i in range(self.__current_file_index, self.__total_files):
                    if j >= batch_size:
                        break
                    self.__last_read_file_name = file_name = files[i if not self.__shuffle else self.__rand_arr[i]]
                    category_index = self.__label_to_index_map[self.__file_name_to_label_map[file_name]] \
                        if file_name in self.__file_name_to_label_map else 0
                    print("----> opening index=", i, " name=", file_name, " category=",
                          self.find_label_by_index(category_index))
                    self.__current_file_index = i + 1
                    if self.__loop and self.is_eof():
                        self.__current_file_index = 0
                    try:
                        file_name = os.path.join(root, file_name)
                        if self.__speed:
                            file_name = self.__changeWaveSpeed(self.__speed, file_name, self.__temp_file_path)
                        data[j] = self.get_log_mel(file_name)
                        mark[j][category_index] = 1
                        if np.any(np.isnan(data[j])):
                            raise Exception("Data contains NaN")
                        j += 1
                        if j >= batch_size:
                            break
                    except Exception as e:
                        print(file_name, e)
                        continue
            except BaseException as e:
                print("next() exception occurred! e: ", str(e))
            if j < batch_size:
                return data[0:j, :], mark[0:j, :]
            else:
                return data, mark

    def get_log_mel(self, file_path):
        # Load sound file
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        duration = y.shape[0] / sr
        print(file_path + " - duration=", duration)
        offset = 0
        if duration > 20:
            offset = 10
        elif duration > 14:
            offset = 7
        elif duration > 8:
            offset = 4
        elif duration > 4:
            offset = 2
        if offset > 0:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE, offset=offset)
            duration = y.shape[0] / sr
            print(file_path + " - duration trimmed=", duration)

        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=self.__n_mfcc)

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_db = librosa.amplitude_to_db(S, ref=np.max)

        print("log_db.shape=", log_db.shape)

        sec_dim = log_db.shape[1]
        fix_size = self.__fixed_sample
        if sec_dim > fix_size:
            log_db = log_db[:, :fix_size]
        elif sec_dim < fix_size:
            log_db = np.pad(log_db, ((0, 0), (0, fix_size - sec_dim)), 'constant', constant_values=(0, 0))
        return log_db

    def set_speed(self, percentage):
        """
        :param percentage: -95 ~ 5000 (%)
        :return:
        """
        self.__speed = percentage

    def __changeWaveSpeed(self, speedPercentage, inputPath, outputPath):
        """
        :param speedPercentage: -95 ~ 5000 (%)
        :param inputPath:
        :param outputPath:
        :return:
        """
        cmd = 'soundstretch ' + inputPath + ' ' + outputPath + ' -ratio=' + str(speedPercentage)
        print("-speed=", speedPercentage)
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            process.wait()
        except OSError as e:
            raise
        return outputPath

    def is_eof(self):
        return self.__total_files >= 0 and self.__current_file_index >= self.__total_files - 1

    # tool methods

    @staticmethod
    def __get_random_arr(max):
        result = np.arange(0, max)
        for i in range(0, max):
            temp = result[i]
            swap_index = random.randint(0, max - 1)
            result[i] = result[swap_index]
            result[swap_index] = temp
        return result