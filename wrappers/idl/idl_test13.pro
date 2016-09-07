pro idl_test13

resolution = 275

status = mtk_setregion_by_path_blockrange(39,50,70,region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_region_to_pathlist(region, npath, pathlist)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,"Mtk_region_to_pathlist = ", pathlist

for i = 0, npath-1 do begin

    WINDOW, /FREE, XSIZE = 360 * 2, YSIZE = 180 * 2, xpos = i, ypos = i,$
      title = "Path " + string(strtrim(pathlist[i],2))
;    MAP_SET, 0.0, 0.0, 0.0, /NOBORDER, XMARGIN = 0.0, YMARGIN = 0.0, /CONTINENTS, /GRID
    MAP_SET,/GRID,/CONTINENT,LIMIT=[60,-130,20,-70], /NOBORDER, XMARGIN = 0.0, YMARGIN = 0.0  

    status = mtk_snap_to_grid(pathlist[i], resolution, region, map)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
        stop
    end
    print, map.path, map.start_block, map.end_block

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

    path = map.path
    sb = map.start_block
    eb = map.end_block
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
endfor

print,"Press return to close all windows..."
done=""
read,done
while (!d.window ne -1) do begin
    wait, 0.25
    wdelete,!d.window
endwhile
end

