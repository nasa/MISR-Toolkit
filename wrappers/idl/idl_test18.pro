pro idl_test18

print, 'Pick one'
print, '1) Radiance/RDQI'
print, '2) Scaled Radiance (DN)'
print, '3) Radiance'
print, '4) RDQI'
print, '5) Equivalent Reflectance'
print, '6) Brf'
ans = ''
read,ans

if ans eq 1 then field = "Radiance/RDQI"
if ans eq 2 then field = "DN"
if ans eq 3 then field = "Radiance"
if ans eq 4 then field = "RDQI"
if ans eq 5 then field = "Equivalent Reflectance"
if ans eq 6 then field = "Brf"

lat = 33.2
lon = -112.5
lat_ext = 1000
lon_ext = 500
status = mtk_setregion_by_latlon_extent(lat, lon, lat_ext, lon_ext, "km", region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf"

gridname = "RedBand"
fieldname = "Red " + field
status = mtk_readdata(filename, gridname, fieldname, region, red)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

gridname = "BlueBand"
fieldname = "Blue " + field
status = mtk_readdata(filename, gridname, fieldname, region, blu, blumapinfo)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

gridname = "GreenBand"
fieldname = "Green " + field
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

window,0,xsize=blumapinfo.nsample,ysize=blumapinfo.nline,title=field
tv,hist_equal(img),/order,true=3

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

    print,"      (Fieldtype) = ", field
    print,"       (Data RGB) = ", littlered[s,l], grn[s,l], blu[s,l]

    i = i + 1
endrep until !err eq 4

wdelete,0
end
