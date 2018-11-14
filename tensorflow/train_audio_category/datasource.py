import wave
import numpy as np
import os
from ffmpy import FFmpeg
import logging
from matplotlib import pyplot as plt

class NotEnoughFrameException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value

class DataSource:
    '''
    providing fft result (in amplitude) of single channel pcm data stream in np.array 
    of size [batchSize, dataSize] in np.float32
    NOTE THAT the fft result is symmetric, so the [0, dataSize/2] and [dataSize/2, dataSize] are same for each row.
    '''
    totalFiles = -1
    workingPath = ''
    tempFilePath = ''
    currentFileIndex = 0
    categoryNum = 0
    __REPEAT = 0 # how many times we read a same file
    __NFFT = 256
    __OVERLAP = 128

    def __init__(self, workingPath, tempFilePath, categoryNum, startIndex = 0, repeat = 1):
        '''
        :param workingPath: the directory path of all m4a files
        :param tempFilePath: the path for temporary file(one wave temp file is generated)
        '''
        self.workingPath = workingPath
        self.tempFilePath = tempFilePath
        self.categoryNum = categoryNum
        self.currentFileIndex = startIndex
        self.__REPEAT = repeat
        for root, dirs, files in os.walk(workingPath):
            for fileName in files:
                if fileName.find("DS_Store") >= 0:
                    os.remove(os.path.join(root, fileName))

    def setCurrentIndex(self, index):
        '''
        Set the current file index in given directory
        :param index: 
        :return: 
        '''
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

    def next(self, skippedFrames, batchSize, dataSize):
        '''
        :param batchSize: how many files you want to sample
        :param dataSize: how many frames of pcm you can get from each file
                         this number should be 2^N such as 4096
        :return:
                (data, mark)
                data: the np.array in size [batchSize, dataSize] if left files are enough
                or return [leftFileNum, dataSize] or None if no file is left
                mark: the np.array in size [batchSize, categoryNum] one-hot form
        '''
        batchSize = batchSize / self.__REPEAT
        for root, dirs, files in os.walk(self.workingPath):
            if self.totalFiles == -1:
                self.totalFiles = len(files)
                print("total files:", self.totalFiles)
            end = int(self.currentFileIndex + batchSize)\
                if self.totalFiles > batchSize + self.currentFileIndex\
                else self.totalFiles
            if end <= 0 or dataSize <= 0:
                print("reach end or dataSize is below 0, currIdx=", str(self.currentFileIndex), ";dataSize=", dataSize)
                return None
            realBatch = int((end - self.currentFileIndex) * self.__REPEAT)
            data = np.zeros((realBatch, dataSize), np.float32)
            mark = np.zeros((realBatch, self.categoryNum), np.int32)
            try:
                j = 0
                for i in range(self.currentFileIndex, end):
                    fileName = files[i]
                    categoryIndex = int(fileName.split("_")[-1])
                    print("----> opening file ", i)
                    outPath = self.transform2Wave(os.path.join(root, fileName), self.tempFilePath)
                    try:
                        temp = self.readWaveData(skippedFrames, dataSize + self.__OVERLAP, outPath)
                        for k in range(0, temp.shape[0]):
                            data[j] = self.getSpecgram(temp[k], dataSize)
                            mark[j][categoryIndex] = 1
                            if np.any(np.isnan(data[j])):
                                raise Exception("Data contains NaN")
                            j += 1
                    except NotEnoughFrameException as e:
                        print(fileName, e)
                        continue
                    except Exception as e:
                        print(e)
                        continue
            except BaseException as e:
                print("next() exception occurred! e: ", str(e))
                logging.exception(e)
            self.currentFileIndex = end
            return data, mark

    def transform2Wave(self, filePath, outPath):
        ffmpeg = FFmpeg(inputs={filePath: None}, outputs={outPath: "-f wav -y"})
        ffmpeg.run()
        return outPath

    def readWaveData(self, skippedFrames, readFrames, file_path):
        # open a wave file, and return a Wave_read object
        f = wave.open(file_path, "rb")
        # read the wave's format infomation,and return a tuple
        params = f.getparams()
        # get the info
        nchannels, sampwidth, framerate, nframes = params[:4]
        print("framerate:", framerate)
        print("channel:", nchannels)
        print("bytesPerSample:", sampwidth)
        print("totalFrames:", nframes)
        readFrames = int(self.__REPEAT * readFrames)
        if nframes < skippedFrames + readFrames:
            raise NotEnoughFrameException("beyond total frame (%d) skip (%d) read (%d)" % (nframes, skippedFrames, readFrames))
        #first skip as required
        if skippedFrames > 0 and skippedFrames * nchannels < nframes:
            f.setpos(skippedFrames * nchannels)
        # Reads and returns nframes of audio, as a string of bytes.
        str_data = f.readframes(readFrames)
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
            result.resize(self.__REPEAT, int(readFrames / self.__REPEAT))
            return result
        else:
            length = wave_data.shape[0]
            result = np.zeros((1, length), dtype=np.float32)
            for i in range(0, length):
                result[0][i] = (wave_data[i] + 32768) / 65535.0
            result.resize(self.__REPEAT, int(readFrames / self.__REPEAT))
            return result

    def close(self):
        os.remove(self.tempFilePath)

    def normalize2dArr(self, arr):
        size = arr.shape
        max = np.max(arr)
        min = np.min(arr)
        base = max - min
        if abs(base) > 1e-10:
            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    arr[i][j] = (arr[i][j] * 1.0 - min) / base
        else:
            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    arr[i][j] = 0.5

    def getSpecgram(self, data, dataSize):
        out = plt.mlab.specgram(data, NFFT=self.__NFFT, Fs=2, detrend=plt.mlab.detrend_none,
                                window=plt.mlab.window_hanning, noverlap=self.__OVERLAP, mode='psd', pad_to=self.__NFFT-1)
        # print('out.shape=', out[0].shape)
        self.normalize2dArr(out[0])
        return out[0].flatten()

    def isEOF(self):
        return self.totalFiles >= 0 and self.currentFileIndex >= self.totalFiles - 1

workingPath = "/Users/wiizhang/Documents/python_scripts/tensorflow/train_audio_category/downloaded_programs"
tempPath = "/Users/wiizhang/Documents/python_scripts/tensorflow/train_audio_category/temp.wav"
dataSource = DataSource(workingPath, tempPath, 10, 301, 1)
(data, mark) = dataSource.next(44100 * 5, 1, 16384)

dataSource.close()

print('data.shape=', data.shape)
data.resize(128, 128)
im = plt.imshow(data, cmap=plt.cm.hot, origin='upper')
plt.title('specgram of wav')
plt.show()
print(data)