from urllib import request
import sys

urlPrefix = "http://ws.stream.fm.qq.com/"
urlPostfix = ".m4a?fromtag=36&vkey=EE1FEB009AA1F2263F781F1C2685628307D93058CB1EEBB1BCDFBC41933CBE766036F0AECCC383D5BF3C094E44241B7969AD9B45DBE92A23&guid=5113689308552605696&rand=1509181230718"
programs = []
categorys = dict()

class Program:
    """
    save information for each program
    """
    showId = ""
    showName = ""
    author = ""
    categoryName = ""
    fileName = ""

    def __init__(self):
        pass

    def getUrl(self):
        return urlPrefix + self.fileName + urlPostfix

    def getFileName(self):
        return self.showId + "_" + self.showName + "_" + self.author + "_" + self.categoryName

def processLine(line):
    if not line:
        return
    items = line.split("|")
    if len(items) >= 5:
        program = Program()
        program.showId = items[0].strip()
        program.showName = items[1].strip()
        program.fileName = items[2].strip()
        program.author = items[3].strip()
        program.categoryName = items[4].strip()
        # add to global array
        programs.append(program)
        categoryNum = categorys.get(program.categoryName)
        if not categoryNum:
            categorys[program.categoryName] = 1
        else:
            categorys[program.categoryName] = 1 + categoryNum

def downloadFile(url, localFilePath):
    try:
        response = request.urlopen(url)
        with open(localFilePath, "wb") as f:
            blockSize = 8192
            totalWrite = 0
            while True:
                buffer = response.read(blockSize)
                if not buffer:
                    break
                totalWrite += len(buffer)
                f.write(buffer)

    except Exception as e:
        print(e)
        print('download exception %s' % url)
        return -1



if __name__ == '__main__':
    if len(sys.argv) > 2:
        inputFile = sys.argv[1]
        localDirPath = sys.argv[2]
        with open(inputFile, "r") as file:
            while 1:
                line = file.readline()
                if not line:
                    break
                processLine(line)
        print("total program: %d " % len(programs))
        print("all categorys:")

        for key, value in categorys.items():
            print("category:%s, num:%d" % (key, value))
        if not localDirPath.endswith("/"):
            localDirPath += "/"
        print("start downloading to %s " % localDirPath)

        begin = 4460
        end = 5000
        for i in range(begin, end):            
            if i >= len(programs):
                break
            program = programs[i]
            print("%.2f%% %s" % ((i - begin) / float(end - begin) * 100, program.fileName))
            downloadFile(program.getUrl(), localDirPath + program.getFileName())
        print("download completed")
    else:
        print("arg[1]=dataFileName arg[2]=outputDirPath")

