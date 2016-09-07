pro idl_test7

path = 37
sblock = 30
eblock = 35
status = mtk_setregion_by_path_blockrange(path, sblock, eblock, region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
help,region
help,/struct,region

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf"
gridname = "Standard"

fieldname = "AveSceneElev"
status = mtk_readdata(filename, gridname, fieldname, region, elevbuf, map)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_readblockrange(filename, gridname, fieldname, sblock, eblock, elevblk)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
fieldname = "GeoLatitude"
status = mtk_readdata(filename, gridname, fieldname, region, latbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_readblockrange(filename, gridname, fieldname, sblock, eblock, latblk)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

fieldname = "GeoLongitude"
status = mtk_readdata(filename, gridname, fieldname, region, lonbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_readblockrange(filename, gridname, fieldname, sblock, eblock, lonblk)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

window,0,xsize=map.nsample,ysize=map.nline,title="AveSceneElev"
tv,hist_equal(elevbuf),/order

print, "        ", max([map.geo_ulc_lat, map.geo_urc_lat])
print, min([map.geo_ulc_lon,map.geo_llc_lon]), max([map.geo_urc_lon,map.geo_lrc_lon])
print, "        ", min([map.geo_llc_lat, map.geo_lrc_lat])

print, 'Left-click to query coordinates'
print, 'Right-click to exit'

fmt = '(a,2f14.6)'
i = 0
repeat begin
    cursor,s,l,/device,/down
    xyouts,s,l,strtrim(i,2),/device
    l = map.nline - l
    print,"point: ", i
    print,"         (IDL l,s) = ", l, s

    status = mtk_ls_to_latlon(map, l, s, lat, lon)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"     (MTK lat,lon) = ", lat, lon,format=fmt

    status = mtk_dd_to_deg_min_sec(lat, deg, min, sec)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print," (MTK lat (d,m,s)) = ", deg, min, sec

    status = mtk_dd_to_deg_min_sec(lon, deg, min, sec)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
        stop
    end
    print," (MTK lon (d,m,s)) = ", deg, min, sec

    status = mtk_latlon_to_bls(map.path, map.resolution, lat, lon, bp, lp, sp)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"       (MTK b,l,s) = ", bp, lp, sp

    print,"(Plane data value) = ", elevbuf[s,l]
    if (bp gt 0) then begin 
        print,"(Stack data value) = ", elevblk[sp,lp,bp - sblock]
    end else begin
        print,"(Stack data value) = out of bounds"
    endelse 

    print," (Plane lat value) = ", latbuf[s,l]
    if (bp gt 0) then begin
        print," (Stack lat value) = ", latblk[sp,lp,bp - sblock]
    end else begin
        print,"(Stack lat value) = out of bounds"
    endelse 

    print," (Plane lon value) = ", lonbuf[s,l]
    if (bp gt 0) then begin
        print," (Stack lon value) = ", lonblk[sp,lp,bp - sblock]
    end else begin
        print,"(Stack lon value) = out of bounds"
    endelse

    i = i + 1
endrep until !err eq 4

wdelete,0
end
