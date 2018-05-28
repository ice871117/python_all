import wave
import numpy as np
import os
from ffmpy import FFmpeg
import logging
from matplotlib import pyplot as plt
import pandas as pd
import random


class NotEnoughFrameException(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class DataSource:
    """
    providing fft result (in amplitude) of single channel pcm data stream in np.array 
    of size [batchSize, dataSize] in np.float32
    NOTE THAT the fft result is symmetric, so the [0, dataSize/2] and [dataSize/2, dataSize] are same for each row.
    """
    totalFiles = -1
    workingPath = ''
    currentFileIndex = 0
    category_num = 0
    __REPEAT = 0 # how many times we read a same file
    __NFFT = 256
    __OVERLAP = 128
    __file_name_to_label_map = dict()
    __label_to_index_map = dict()
    last_read_file_name = ""
    __rand_arr = []

    def __init__(self, working_path, train_file_path, start_index=0, repeat=1, traverse_random=False):
        """
        :param working_path: the directory path of all m4a files
        :param train_file_path: the path for the train data category file
        """
        self.workingPath = working_path
        self.currentFileIndex = start_index
        self.__REPEAT = repeat
        self.totalFiles = len(os.listdir(self.workingPath))
        print("total files:", self.totalFiles)
        if traverse_random:
            self.__rand_arr = self.__get_random_arr(self.totalFiles)
        for root, dirs, files in os.walk(working_path):
            for fileName in files:
                if fileName.find("DS_Store") >= 0:
                    os.remove(os.path.join(root, fileName))
        if train_file_path:
            self.__init_train_category(train_file_path)

    def __init_train_category(self, category_csv):
        all_data = pd.read_csv(category_csv)
        row_num = all_data.shape[0]
        for i in range(0, row_num):
            file_name = all_data.loc[i, 'fname']
            label = all_data.loc[i, 'label']
            self.__file_name_to_label_map[file_name] = label
            if label not in self.__label_to_index_map:
                self.__label_to_index_map[label] = len(self.__label_to_index_map)
        self.category_num = len(self.__label_to_index_map)
        print("init category : ", self.category_num)
        print("all categories : ", self.__label_to_index_map)

    def get_category_num(self):
        return self.category_num

    def get_last_read_file_name(self):
        return self.last_read_file_name

    def set_current_index(self, index):
        """
        Set the current file index in given directory
        :param index: 
        :return: 
        """
        if self.totalFiles == -1:
            self.totalFiles = len(os.listdir(self.workingPath))
        if index > 0:
            self.currentFileIndex = index
            if self.currentFileIndex >= self.totalFiles:
                self.currentFileIndex = self.totalFiles - 1
        elif index < 0:
            self.currentFileIndex = self.totalFiles + index
            if self.currentFileIndex < 0:
                self.currentFileIndex = 0

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

    def next(self, skipped_seconds, batch_size, data_size):
        """
        :param skipped_seconds: how many seconds to be skipped at the front of each file
        :param batch_size: how many rows of data you want to sample
        :param data_size: how many frames of pcm you can get from each file
                         this number should be 2^N such as 4096
        :return:
                (data, mark)
                data: the np.array in size [batchSize, dataSize] if left files are enough
                or return [leftFileNum, dataSize] or None if no file is left
                mark: the np.array in size [batchSize, categoryNum] one-hot form
        """
        for root, dirs, files in os.walk(self.workingPath):
            if self.totalFiles <= 0:
                print("the folder ", self.workingPath, " is empty no file could be read!")
                return None, None
            if data_size <= 0:
                print("reach end or dataSize is below 0, currIdx=", str(self.currentFileIndex), ";dataSize=", data_size)
                return None, None
            data = np.zeros((batch_size, data_size), np.float32)
            mark = np.zeros((batch_size, self.category_num), np.int32)
            j = 0
            try:
                for i in range(self.currentFileIndex, self.totalFiles):
                    if j >= batch_size:
                        break
                    self.last_read_file_name = file_name = files[i if len(self.__rand_arr) == 0 else self.__rand_arr[i]]
                    category_index = self.__label_to_index_map[self.__file_name_to_label_map[file_name]]\
                        if file_name in self.__file_name_to_label_map else 0
                    print("----> opening index=", i, " name=", file_name, " category=", category_index)
                    self.currentFileIndex = i
                    # already wave and no need to transform
                    # outPath = self.transform2Wave(os.path.join(root, fileName), self.tempFilePath)
                    try:
                        temp = self.read_wave_data(skipped_seconds, data_size + self.__OVERLAP, os.path.join(root, file_name))
                        for k in range(0, temp.shape[0]):
                            data[j] = self.get_specgram(temp[k])
                            mark[j][category_index] = 1
                            if np.any(np.isnan(data[j])):
                                raise Exception("Data contains NaN")
                            j += 1
                            if j >= batch_size:
                                break
                    except NotEnoughFrameException as e:
                        print(file_name, e)
                        continue
                    except Exception as e:
                        print(e)
                        continue
            except BaseException as e:
                print("next() exception occurred! e: ", str(e))
                logging.exception(e)
            if j < batch_size:
                return data[0:j, :], mark[0:j, :]
            else:
                return data, mark

    @staticmethod
    def __get_random_arr(max):
        result = np.arange(0, max)
        for i in range(0, max):
            temp = result[i]
            swap_index = random.randint(0, max - 1)
            result[i] = result[swap_index]
            result[swap_index] = temp
        return result

    @staticmethod
    def transform_to_wave(file_path, out_path):
        ffmpeg = FFmpeg(inputs={file_path: None}, outputs={out_path: "-f wav -y"})
        ffmpeg.run()
        return out_path

    def read_wave_data(self, skipped_seconds, read_frames, file_path):
        # open a wave file, and return a Wave_read object
        f = wave.open(file_path, "rb")
        # read the wave's format information,and return a tuple
        params = f.getparams()
        # get the info
        nchannels, sampwidth, framerate, nframes = params[:4]
        print("framerate:", framerate)
        print("channel:", nchannels)
        print("bytesPerSample:", sampwidth)
        print("totalFrames:", nframes)
        real_repeat = self.__REPEAT
        tmp_read_frames = int(real_repeat * read_frames)
        skipped_frames = skipped_seconds *framerate
        while nframes < skipped_frames + tmp_read_frames:
            real_repeat -= 1
            if real_repeat <= 0:
                raise NotEnoughFrameException("beyond total frame (%d) skip (%d) read (%d)" % (nframes, skipped_frames, read_frames))
            tmp_read_frames = int(real_repeat * read_frames)
        #first skip as required
        if skipped_frames > 0 and skipped_frames * nchannels < nframes:
            f.setpos(skipped_frames * nchannels)
        # Reads and returns nframes of audio, as a string of bytes.
        str_data = f.readframes(tmp_read_frames)
        # close the stream
        f.close()
        # turn the wave's data to array
        wave_data = np.fromstring(str_data, dtype=np.short)
        if nchannels == 2:
            length = int(len(wave_data) / 2)
            result = np.zeros((1, length), dtype=np.float32)
            # for the data is stereo,and format is LRLRLR...
            # shape the array to n*2(-1 means fit the y coordinate)
            wave_data.resize(length, 2)
            for i in range(0, wave_data.shape[0]):
                result[0][i] = (wave_data[i][0] + 32768) / 65535.0
            result.resize(real_repeat, tmp_read_frames // real_repeat)
            return result
        else:
            length = wave_data.shape[0]
            result = np.zeros((1, length), dtype=np.float32)
            for i in range(0, length):
                result[0][i] = (wave_data[i] + 32768) / 65535.0
            result.resize(real_repeat, tmp_read_frames // real_repeat)
            return result

    @staticmethod
    def normalize(arr):
        return (arr - arr.mean()) / arr.var()

    @staticmethod
    def get_mel_freq(arr):
        return 2595.0 * np.log10(1 + arr / 700)

    def get_specgram(self, data):
        out = plt.mlab.specgram(data, NFFT=self.__NFFT, Fs=2, detrend=plt.mlab.detrend_none,
                                window=plt.mlab.window_hanning, noverlap=self.__OVERLAP, mode='psd', pad_to=self.__NFFT-1)
        # print('out.shape=', out[0].shape)
        out = 10 * np.log10(out[0].flatten())
        return self.normalize(out)

    def is_EOF(self):
        return self.totalFiles >= 0 and self.currentFileIndex >= self.totalFiles - 1


if __name__ == "__main__":
    workingPath = "../../../input/audio_train"
    train_file = "train.csv"
    data_source = DataSource(workingPath, train_file, start_index=204, repeat=1)
    (data, mark) = data_source.next(0, 1, 16384)

    print('data.shape=', data.shape)
    data.resize(128, 128)
    im = plt.imshow(data, cmap=plt.cm.hot, origin='upper')
    plt.title(data_source.find_label_by_index(mark.argmax()))
    plt.show()
    print(data)