pro idl_test11

lat = 33.2
lon = -112.5
lat_ext = 1000000
lon_ext = 500000
status = mtk_setregion_by_latlon_extent(lat, lon, lat_ext, lon_ext, "m", region)
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

gridname = "RedBand"
fieldname = "Red Radiance/RDQI"
status = mtk_readdata(filename, gridname, fieldname, region, red)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

gridname = "BlueBand"
fieldname = "Blue Radiance/RDQI"
status = mtk_readdata(filename, gridname, fieldname, region, blu, blumapinfo)

gridname = "GreenBand"
fieldname = "Green Radiance/RDQI"
status = mtk_readdata(filename, gridname, fieldname, region, grn)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

littlered = rebin(red, blumapinfo.nsample, blumapinfo.nline)

img = [[[bytscl(littlered)]],[[bytscl(grn)]],[[bytscl(blu)]]]

status = mtk_time_meta_read(filename, timemetadata)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

window,0,xsize=blumapinfo.nsample,ysize=blumapinfo.nline,title=fieldname
tvscl,hist_equal(img),/order,true=3

print, "        ", max([blumapinfo.geo_ulc_lat, blumapinfo.geo_urc_lat])
print, min([blumapinfo.geo_ulc_lon,blumapinfo.geo_llc_lon]), max([blumapinfo.geo_urc_lon,blumapinfo.geo_lrc_lon])
print, "        ", min([blumapinfo.geo_llc_lat, blumapinfo.geo_lrc_lat])

print, 'Left-click to query coordinates'
print, 'Right-click to exit'

fmt = '(a,2f14.6)'
i = 0
repeat begin
    cursor,s,l,/device,/down
    xyouts,s,l,strtrim(i,2),/device
    l = blumapinfo.nline - l
    print,"point: ", i
    print,"        (IDL l,s) = ", l, s

    status = mtk_ls_to_latlon(blumapinfo, l, s, lat, lon)
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

    print,"    (AGP lat,lon) = ", latbuf[s,l], lonbuf[s,l], format=fmt

    status = mtk_dd_to_deg_min_sec(latbuf[s,l], deg, min, sec)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"(AGP lat (d,m,s)) = ", deg, min, sec

    status = mtk_dd_to_deg_min_sec(lonbuf[s,l], deg, min, sec)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"(AGP lon (d,m,s)) = ", deg, min, sec

    status = mtk_latlon_to_ls(blumapinfo, lat, lon, l, s)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"        (MTK l,s) = ", l, s

    status = mtk_latlon_to_bls(blumapinfo.path, blumapinfo.resolution, lat, lon, bp, lp, sp)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"      (MTK b,l,s) = ", bp, lp, sp

    status = mtk_ls_to_somxy(blumapinfo, l, s, somx, somy)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"   (MTK som x, y) = ", somx, somy

    status = mtk_pixel_time(timemetadata, somx, somy, pixeltime)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
    end
    print,"     (Pixel time) = ", pixeltime

    print,"       (Data RGB) = ", littlered[s,l], grn[s,l], blu[s,l]

    i = i + 1
endrep until !err eq 4

wdelete,0
end
