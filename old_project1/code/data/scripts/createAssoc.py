#!/usr/bin/python

numlines = sum(1 for line in open('rgb.txt'))

rgbfile = open('rgb.txt', 'r')
depthfile = open('depth.txt', 'r')

concat = ""

for i in range(3):
	rgbfile.readline()[:-1]
	depthfile.readline()
for i in range(numlines):
	concat += rgbfile.readline()[:-1]  + " " + depthfile.readline()


depthfile.close()
rgbfile.close()

outfile = open('rgbd_assoc.txt', 'w')

outfile.write(concat)

outfile.close()
