import xlrd
import xlwt
import os
import sys

if len(sys.argv) > 1:
    fileName = sys.argv[1]
    data = xlrd.open_workbook(fileName)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数

    brandMap = dict()
    for i in range(0, nrows):
        rowValues = table.row_values(i)  # 某一行数据
        numCols = len(rowValues)
        lastCol = str(rowValues[numCols - 1])
        try:
            brand = lastCol[lastCol.index("f=") + 2:]
            brand = brand[0 : brand.index("&amp;")]
            if brandMap.get(brand) == None:
                brandMap.setdefault(brand, 1)
            else:
                brandMap[brand] += 1
        except BaseException as e:
            pass

    print("generating result...")
    writebook = xlwt.Workbook()
    table2 = writebook.add_sheet("result")
    index = 0
    for key, value in brandMap.items():
        if len(key.strip()) == 0:
            continue
        table2.write(index, 0, key)
        table2.write(index, 1, value)
        index += 1
    destFile = None
    if fileName.startswith("/") or finleName.find("/") >= 0:
        destFile = '/Users/wiizhang/downloads/resultof_' + fileName[fileName.rindex("/") + 1:]
    else:
        destFile = '/Users/wiizhang/downloads/resultof_' + fileName
    if destFile.find(".") >= 0:
        destFile = destFile[0:destFile.rindex(".")] + ".xls"
    try:
        os.remove(destFile)
    except:
        pass
    writebook.save(destFile)
    print("saving to " + destFile)



