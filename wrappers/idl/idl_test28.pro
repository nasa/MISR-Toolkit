pro idl_test28

resolution = 275
sb = 1
eb = 180

WINDOW, /FREE, XSIZE = 360 * 4, YSIZE = 180*4
MAP_SET, 0.0, 0.0, 0.0, /NOBORDER, XMARGIN = 0.0, YMARGIN = 0.0, /CONTINENTS, /GRID

for path = 1, 233, 10 do begin 

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

;        OPLOT,[bcnr.block[blk].ctr.lon], [bcnr.block[blk].ctr.lat], $
;          color=ishft(long(bc2),16)+ishft(long(gc),8)+ishft(long(rc),0), $
;          psym = 2

    endfor
    wait,.05
endfor

print,"Press return to close all windows..."
done=""
read,done
while (!d.window ne -1) do wdelete,!d.window
end
