import os

targetDir="downloaded_programs"

categorys = {}

def findCategory(name):
    for key, value in categorys.items():
        if name.find(key) >= 0:
            return value
    return 9999

for root, dirs, files in os.walk(targetDir):
    categoryIndex = 0
    for fileName in files:
        if fileName.find("DS_Store") == -1:
            category = fileName.split("_")[-1]
            if categorys.get(category) == None:
                categorys[category] = categoryIndex
                categoryIndex += 1

for root, dirs, files in os.walk(targetDir):
    for fileName in files:
    	if fileName.find("DS_Store") == -1:
            newName = fileName + "_" + findCategory(fileName)
            os.rename(os.path.join(root, fileName), os.path.join(root, newName))
