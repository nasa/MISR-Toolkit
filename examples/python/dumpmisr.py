#!/usr/bin/env python

import MisrToolkit as Mtk
import os

filename = os.getenv('MTKHOME') + '/../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf'
gridname = 'RedBand'
fieldname = 'Red DN'

print "Reading product file..."
region = Mtk.MtkRegion(37,40,45)
d = Mtk.MtkFile(filename).grid(gridname).field(fieldname).read(region)

print "Writing raw file..."
n = d.data()
n.dump('Red_DN.raw')
nlines = n.shape[0]
nsamples = n.shape[1]

print "Writing header info file..."
m = d.mapinfo()
ulc = m.geo.ulc.lat, m.geo.ulc.lon
urc = m.geo.urc.lat, m.geo.urc.lon
ctr = m.geo.ctr.lat, m.geo.ctr.lon
lrc = m.geo.lrc.lat, m.geo.lrc.lon
llc = m.geo.llc.lat, m.geo.llc.lon

f = open('Red_DN.info', 'w')
f.write('File = ' + filename + '\n')
f.write('Grid = ' + gridname + '\n')
f.write('Field = ' + fieldname + '\n')
f.write('Header bytes = 128\n')
f.write('Number of lines = ' + str(nlines) + '\n')
f.write('Number of samples = ' + str(nsamples) + '\n')
f.write('Element size in bytes = ' + str(n.dtype.itemsize) + '\n')
f.write('Datatype = ' + n.dtype.name + '\n')
f.write('Numpy string (n.dtype.str) = ' + str(n.dtype.str) + '\n')
f.write('ULC lat/lon = ' + str(ulc) + '\n')
f.write('URC lat/lon = ' + str(urc) + '\n')
f.write('CTR lat/lon = ' + str(ctr) + '\n')
f.write('LRC lat/lon = ' + str(lrc) + '\n')
f.write('LLC lat/lon = ' + str(llc) + '\n')
f.close()

