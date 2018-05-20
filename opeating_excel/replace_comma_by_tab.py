#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys


try:
    f = None
    fout = None
    if len(sys.argv) >= 3:
        print "source file: ", sys.argv[1], "\ndest file: ", sys.argv[2]
        f = open(sys.argv[1], 'r')
        fout = open(sys.argv[2], 'w')

        allText = f.readlines()
        newText = []
        for eachLine in allText:
            newText.append(eachLine.replace(",", "\t"))
        fout.writelines(newText)

finally:
    print 'finished'
    if f:
        f.close()
    if fout:
        fout.close()



