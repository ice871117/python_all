#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string

def fetch_spend_time(line):
    begin = line.find('play_stop') + len('play_stop')
    if begin > 0:
        num = string.strip(line[begin:])
        try:
            return string.atoi(num)
        except:
            return -1
    return -1

def get_percentage(count, total):
    a = float(count) / total
    return str(round(a, 2))

try:
    f = open('/Users/wiizhang/Downloads/android_play_stop_20150531.txt', 'r')

    sumCount = 0
    circleCount = 0
    jumpOver = len('spend_time=')
    max = 0
    distribute = [0, 0, 0, 0, 0, 0, 0]

    for line in f:
        result = fetch_spend_time(line)

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
            if result > 3600 * 3:
                print 'invalid time = ' + str(result)
    if circleCount > 0:
        print ('statistic result : sum is ' + str(sumCount)
               + '\r\n; max = ' + str(max)
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



