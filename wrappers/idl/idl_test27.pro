pro idl_test27

resolution = 275
path = [38, 1]
start_block = [1, 40]
end_block = [180, 130]

WINDOW, /FREE, XSIZE = 360 * 4, YSIZE = 180*4, $
  title = "Path " + string(strtrim(path[0],2)) + " & Path " + string(strtrim(path[1],2))
MAP_SET, 0.0, 0.0, 0.0, /NOBORDER, XMARGIN = 0.0, YMARGIN = 0.0, /CONTINENTS, /GRID

status = mtk_setregion_by_path_blockrange(path[0], start_block[0], end_block[0], region1)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

status = mtk_setregion_by_path_blockrange(path[1], start_block[1], end_block[1], region2)
if (status ne 0) then begin
    print,"Test Error: ", mtk_error_message(status)
    stop
end

region = [region1, region2]

for i = 0, 1 do begin

  status = mtk_snap_to_grid(path[i], resolution, region[i], map)
  if (status ne 0) then begin 
      print,"Test Error: ", mtk_error_message(status)
      stop
  end

  sb = map.start_block
  eb = map.end_block

  status = mtk_path_blockrange_to_blockcorners(path[i], sb, eb, bcnr)
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

  rc = 255B
  gc = 0B
  bc = 0B

  PLOTS,$
    [map.geo_llc_lon, map.geo_lrc_lon, map.geo_urc_lon, map.geo_ulc_lon, map.geo_llc_lon], $
    [map.geo_llc_lat, map.geo_lrc_lat, map.geo_urc_lat, map.geo_ulc_lat, map.geo_llc_lat], $
    /data, color=ishft(long(bc),16)+ishft(long(gc),8)+ishft(long(rc),0)

  OPLOT,[map.geo_ctr_lon], [map.geo_ctr_lat], $
    color=ishft(long(bc),16)+ishft(long(gc),8)+ishft(long(rc),0), $
    psym = 7, symsize=5

endfor

print,"The blocks are within the map grid."
print,"Press return to close all windows..."
done=""
read,done
while (!d.window ne -1) do wdelete,!d.window
end
