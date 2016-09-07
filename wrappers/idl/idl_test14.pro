pro idl_test14

resolution = 275
path = 38
sb = 48
eb = 70

status = mtk_setregion_by_path_blockrange(path, sb, eb, region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_snap_to_grid(path, resolution, region, map)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

WINDOW, /FREE, XSIZE = 360 * 4, YSIZE = 180*4, $
  title = "Path " + string(strtrim(path,2))
;MAP_SET, 0.0, 0.0, 0.0, /NOBORDER, XMARGIN = 0.0, YMARGIN = 0.0, /CONTINENTS, /GRID
MAP_SET,/GRID,/CONTINENT,LIMIT=[60,-130,20,-70], /NOBORDER, XMARGIN = 0.0, YMARGIN = 0.0  


file = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AGP_P038_F01_24.hdf"
status = mtk_readdata(file, 'Standard', 'GeoLatitude', region, latbuf, latmap)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_readdata(file, 'Standard', 'GeoLongitude', region, lonbuf, lonmap)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

plots,[lonbuf],[latbuf],psym=3,/data


rc = 255B
gc = 0B
bc = 0B

PLOTS,$
  [map.geo_llc_lon, map.geo_lrc_lon, map.geo_urc_lon, map.geo_ulc_lon, map.geo_llc_lon], $
  [map.geo_llc_lat, map.geo_lrc_lat, map.geo_urc_lat, map.geo_ulc_lat, map.geo_llc_lat], $
  /data, color=ishft(long(bc),16)+ishft(long(gc),8)+ishft(long(rc),0)

OPLOT,[map.geo_ctr_lon], [map.geo_ctr_lat], $
  color=ishft(long(bc),16)+ishft(long(gc),8)+ishft(long(rc),0), $
  psym = 2


status = mtk_path_blockrange_to_blockcorners(path, sb, eb, bcnr)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

rc = 0B
gc = 255B
bc1 = 255B
bc2 = 0B

for blk = sb, eb do begin

    PLOTS,$
      [bcnr.block[blk].llc.lon, bcnr.block[blk].lrc.lon, bcnr.block[blk].urc.lon, bcnr.block[blk].ulc.lon, bcnr.block[blk].llc.lon], $
      [bcnr.block[blk].llc.lat, bcnr.block[blk].lrc.lat, bcnr.block[blk].urc.lat, bcnr.block[blk].ulc.lat, bcnr.block[blk].llc.lat], $
      /data, color=ishft(long(bc1),16)+ishft(long(gc),8)+ishft(long(rc),0)

    OPLOT,[bcnr.block[blk].ctr.lon], [bcnr.block[blk].ctr.lat], $
      color=ishft(long(bc2),16)+ishft(long(gc),8)+ishft(long(rc),0), $
      psym = 2

endfor

print,"Press return to close all windows..."
done=""
read,done
while (!d.window ne -1) do wdelete,!d.window
end
