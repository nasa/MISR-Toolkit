pro idl_test19

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf"

status = mtk_file_to_gridlist(filename, gridcnt, gridlist)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, 'Pick one'
for i = 0, gridcnt-1 do begin
    print, i, ') ', gridlist[i]
end
read,ans
gridname = gridlist[ans]

status = mtk_file_grid_to_fieldlist(filename, gridname, fieldcnt, fieldlist)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, 'Pick one'
for i = 0, fieldcnt-1 do begin
    print, i, ') ', fieldlist[i]
end
read,ans
fieldname = fieldlist[ans]

status = mtk_file_grid_field_to_dimlist(filename, gridname, fieldname, ndim, dimlist, dimsize);
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, "Enter dimension for ", fieldname
for i = 0, ndim-1 do begin
    print, dimlist[i], "(0-", strtrim(dimsize[i]-1,2),")"
    read, ans
    fieldname = fieldname + "[" + strtrim(fix(ans),2) + "]"
end
print, fieldname

print,gridname,'/',fieldname

lat = 33.2
lon = -112.5
lat_ext = 1000000
lon_ext = 500000
status = mtk_setregion_by_latlon_extent(lat, lon, lat_ext, lon_ext, "m", region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

status = mtk_readdata(filename, gridname, fieldname, region, buf, map)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
if (status ne 0) then stop

window,0,xsize=map.nsample,ysize=map.nline,title=fieldname
tv,hist_equal(buf),/order

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
    print,"        (IDL l,s) = ", l, s

    status = mtk_ls_to_latlon(map, l, s, lat, lon)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"    (MTK lat,lon) = ", lat, lon,format=fmt

    status = mtk_dd_to_deg_min_sec(lat, deg, min, sec)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"(MTK lat (d,m,s)) = ", deg, min, sec

    status = mtk_dd_to_deg_min_sec(lon, deg, min, sec)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"(MTK lon (d,m,s)) = ", deg, min, sec

    status = mtk_latlon_to_ls(map, lat, lon, l, s)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"        (MTK l,s) = ", l, s

    status = mtk_latlon_to_bls(map.path, map.resolution, lat, lon, bp, lp, sp)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"      (MTK b,l,s) = ", bp, lp, sp

    print,"      (Fieldtype) = ", fieldname
    print,"     (Data value) = ", buf[s,l]

    i = i + 1
endrep until !err eq 4

wdelete,0
end
