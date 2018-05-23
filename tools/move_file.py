import os
import shutil
import logging
import numpy as np
import sys
import random

def __getRandomArr(max):
    result = np.arange(0, max)
    for i in range(0, max):
        temp = result[i]
        swapIndex = random.randint(0, max - 1)
        result[i] = result[swapIndex]
        result[swapIndex] = temp
    return result

def startMove(workingPath, dstFolder, maxMove):
    if maxMove <= 0:
        print("quit for maxMove is ", maxMove)
        return
    if not os.path.exists(dstFolder):
        os.makedirs(dstFolder)
    for root, dirs, files in os.walk(workingPath):
        totalFiles = len(files)
        print("start to move files, total files:", totalFiles, "; total move:", maxMove)
        randoms = __getRandomArr(totalFiles)
        maxMove = min(maxMove, totalFiles)
        for i in range(0, maxMove):
            try:
                fileName = files[randoms[i]]
                print("----> moving file ", fileName, " to ", dstFolder)
                currFilePath = os.path.join(root, fileName)
                dstFilePath = os.path.join(dstFolder, fileName)
                shutil.move(currFilePath, dstFilePath)
            except BaseException as e:
                print("startMove() exception occurred! e: ", str(e))
                logging.exception(e)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        workingDir = sys.argv[1]
        dstFolder = sys.argv[2]
        maxMove = sys.argv[3]
        startMove(workingDir, dstFolder, int(maxMove))
    else:
        workingDir = "F:\\downloaded_programs_2"
        dstFolder = "F:\\test_move"
        maxMove = 200
        startMove(workingDir, dstFolder, maxMove)