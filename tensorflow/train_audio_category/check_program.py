import os
from ffmpy import FFmpeg
import logging
import sys

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

    def __init__(self, workingPath, tempFilePath, startIndex = 0):
        '''
        :param workingPath: the directory path of all m4a files
        :param tempFilePath: the path for temporary file(one wave temp file is generated)
        '''
        self.workingPath = workingPath
        self.tempFilePath = tempFilePath
        self.currentFileIndex = startIndex
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

    def startCheck(self):
        badFiles = []
        for root, dirs, files in os.walk(self.workingPath):
            if self.totalFiles == -1:
                self.totalFiles = len(files)
                print("start check now, total files:", self.totalFiles)
            try:
                for i in range(self.currentFileIndex, self.totalFiles):
                    fileName = files[i]
                    print("----> opening file ", i)
                    currFilePath = os.path.join(root, fileName)
                    os.remove(self.tempFilePath)
                    outPath = self.transform2Wave(currFilePath, self.tempFilePath)
                    if os.path.exists(outPath) and os.path.getsize(outPath) > 0:
                        continue
                    else:
                        badFiles.append(currFilePath)
            except BaseException as e:
                print("next() exception occurred! e: ", str(e))
                logging.exception(e)
        for path in badFiles:
            os.remove(path)

    def transform2Wave(self, filePath, outPath):
        ffmpeg = FFmpeg(inputs={filePath: None}, outputs={outPath: "-f wav -y"})
        ffmpeg.run()
        return outPath

    def close(self):
        os.remove(self.tempFilePath)


    def isEOF(self):
        return self.totalFiles >= 0 and self.currentFileIndex >= self.totalFiles - 1


if __name__ == '__main__':
    if len(sys.argv) > 2:
        workingDir = sys.argv[1]
        tempPath = sys.argv[2]
        data = DataSource(workingDir, tempPath)
        data.startCheck()
    else:
        print("bad command arguments")