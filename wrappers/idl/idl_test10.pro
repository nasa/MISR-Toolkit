pro idl_test10

status = mtk_setregion_by_latlon_extent(77.68,-72.025, $
                                        600,1000,'275m',region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf"
gridname = "Standard"
fieldname = "GeoLatitude"
status = mtk_readdata(filename, gridname, fieldname, region, latbuf)
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

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf"
gridname = "BlueBand"
fieldname = "Blue Radiance/RDQI"
status = mtk_readdata(filename, gridname, fieldname, region, buf, mapinfo)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

status = mtk_time_meta_read(filename, timemetadata)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

window,0,xsize=mapinfo.nsample,ysize=mapinfo.nline,title=fieldname
tvscl,buf,/order

print, "        ", max([mapinfo.geo_ulc_lat, mapinfo.geo_urc_lat])
print, min([mapinfo.geo_ulc_lon,mapinfo.geo_llc_lon]), max([mapinfo.geo_urc_lon,mapinfo.geo_lrc_lon])
print, "        ", min([mapinfo.geo_llc_lat, mapinfo.geo_lrc_lat])

print, 'Left-click to query coordinates'
print, 'Right-click to exit'

fmt = '(a,2f14.6)'
i = 0
repeat begin
    cursor,s,l,/device,/down
    xyouts,s,l,strtrim(i,2),/device
    l = mapinfo.nline - l
    print,"point: ", i
    print,"     (IDL l,s) = ", l, s

    status = mtk_ls_to_latlon(mapinfo, l, s, lat, lon)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print," (MTK lat,lon) = ", lat, lon,format=fmt
    print," (AGP lat,lon) = ", latbuf[s,l], lonbuf[s,l],format=fmt

    status = mtk_latlon_to_ls(mapinfo, lat, lon, l, s)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"     (MTK l,s) = ", l, s

    status = mtk_ls_to_somxy(mapinfo, l, s, somx, somy)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"(MTK som x, y) = ", somx, somy

    status = mtk_pixel_time(timemetadata, somx, somy, pixeltime)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"  (Pixel time) = ", pixeltime

    print,"  (data value) = ",buf[s,l]

    i = i + 1
endrep until !err eq 4

wdelete,0
end
