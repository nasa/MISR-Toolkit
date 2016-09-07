#!/usr/bin/env python

import MisrToolkit as Mtk
import os

region = Mtk.MtkRegion(37,40,42)
filename = os.getenv('MTKHOME') + '/../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf'
gridname = 'RedBand'
fieldname = 'Red Brf'
d = Mtk.MtkFile(filename).grid(gridname).field(fieldname).read(region)
n = d.data()
n.shape
m = d.mapinfo()

type(d)
type(n)
type(m)

help(d)
help(n)
help(m)
