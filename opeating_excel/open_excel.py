import xlrd
import sys

if len(sys.argv) > 1:
    fileName = sys.argv[1]
    data = xlrd.open_workbook(fileName)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    for i in range(0, nrows):
        rowValues = table.row_values(i)  # 某一行数据
        for item in rowValues:
            print(item)
