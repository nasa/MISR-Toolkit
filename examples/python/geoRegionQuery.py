#!/usr/bin/env python

import MisrToolkit as Mtk

def geoRegionQuery(datasetName, level, version, startDateTime, endDateTime, latMin,
                   latMax, lonMin, lonMax):

    region = Mtk.MtkRegion(latMax, lonMin, latMin, lonMax)

    for p in region.path_list:
        orbitlist = Mtk.path_time_range_to_orbit_list(p, startDateTime, endDateTime)
        if orbitlist != []:
            print
            print 'Path = ', p, 'Orbits = ', orbitlist
            mapinfo = region.snap_to_grid(p,275)
            print "mapinfo.som.path: ", mapinfo.som.path
            print " nlines/nsample : ", mapinfo.nline, mapinfo.nsample
            print "   ULC (som x/y): ", mapinfo.som.ulc.x, mapinfo.som.ulc.y
            print "   CTR (som x/y): ", mapinfo.som.ctr.x, mapinfo.som.ctr.y
            print "   LRC (som x/y): ", mapinfo.som.lrc.x, mapinfo.som.lrc.y
            print "   ULC (lat/lon): ", mapinfo.geo.ulc.lat, mapinfo.geo.ulc.lon
            print "   URC (lat/lon): ", mapinfo.geo.urc.lat, mapinfo.geo.urc.lon
            print "   CTR (lat/lon): ", mapinfo.geo.ctr.lat, mapinfo.geo.ctr.lon
            print "   LRC (lat/lon): ", mapinfo.geo.lrc.lat, mapinfo.geo.lrc.lon
            print "   LLC (lat/lon): ", mapinfo.geo.llc.lat, mapinfo.geo.llc.lon
            for o in orbitlist:
                filename = Mtk.make_filename('',level, None, p, o, version)
                print "Product filename: ", filename
                st, et = Mtk.orbit_to_time_range(o)
                # Adjust time for local coverage base on Julian time per meter in som x and map ulc/lrc x
                jst = Mtk.datetime_to_julian(st)
                jet = Mtk.datetime_to_julian(et)
                # Julian time per meter som x (orbit_end_time - orbit_start_time) / total_orbit_meters_in_somx
                jtime_per_meter_somx = (jet - jst) / 40263575
                njst = jst + jtime_per_meter_somx * mapinfo.som.ulc.x
                njet = jst + jtime_per_meter_somx * mapinfo.som.lrc.x
                starttime = Mtk.julian_to_datetime(njst)
                endtime = Mtk.julian_to_datetime(njet)
                print "      Start time: ", starttime
                print "        End time: ", endtime

if __name__ == '__main__':
    misr = geoRegionQuery('MISR',
                          'TC_STEREO',
                          'F06_0012',
                          '2003-01-01T00:00:00',
                          '2003-01-01T23:59:59',
                          -90, 90,
                          -180, 180)
    print
    misr = geoRegionQuery('MISR',
                          'AS_AEROSOL',
                          'F03_0024',
                          '2003-01-01T00:00:00',
                          '2003-03-01T23:59:59',
                          30, 40,
                          -120, -110)
