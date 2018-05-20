#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string

def fetch_spend_time(inputStr):
    end = inputStr.index('&')
    if end > 0:
        numInStr = inputStr[0:end]
        return string.atoi(numInStr)
    else:
        return -1

def get_percentage(count, total):
    a = float(count) / total
    return str(round(a, 2))

try:
    f = open('/Users/wiizhang/Downloads/企鹅fm灯塔流水_明细提取_2015-06-02-2.txt', 'r')
    content = f.read()
    startIndex = 0
    length = len(content)
    sumCount = 0
    circleCount = 0
    jumpOver = len('spend_time=')
    distribute = [0, 0, 0, 0, 0, 0, 0]

    while 0 <= startIndex < length:
        startIndex = content.find('spend_time=', startIndex, length + 1)
        if startIndex < 0:
            break
        startIndex += jumpOver
        content = content[startIndex:]
        result = fetch_spend_time(content)

        if result >= 0:
            if result == 0:
                distribute[0] += 1
            elif result <= 10:
                distribute[1] += 1
            elif result <= 20:
                distribute[2] += 1
            elif result <= 30:
                distribute[3] += 1
            elif result <= 40:
                distribute[4] += 1
            elif result <= 50:
                distribute[5] += 1
            else:
                distribute[6] += 1

            sumCount += result
            circleCount += 1
    if circleCount > 0:
        print ('statistic result : sum is ' + str(sumCount)
               + ' \r\n; time = 0 : ' + get_percentage(distribute[0], circleCount)
               + ' \r\n; 0 < time <= 10 : ' + get_percentage(distribute[1], circleCount)
               + ' \r\n; 10 < time <= 20 : ' + get_percentage(distribute[2], circleCount)
               + ' \r\n; 20 < time <= 30 : ' + get_percentage(distribute[3], circleCount)
               + ' \r\n; 30 < time <= 40 : ' + get_percentage(distribute[4], circleCount)
               + ' \r\n; 40 < time <= 50 : ' + get_percentage(distribute[5], circleCount)
               + ' \r\n; 50 < time : ' + get_percentage(distribute[6], circleCount)
               )

finally:
    if f:
        f.close()



