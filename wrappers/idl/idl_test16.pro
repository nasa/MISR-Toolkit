pro idl_test16

lat = 22.87
lon = 28.50
latext = 7-1
lonext = 5
status = mtk_setregion_by_latlon_extent(lat,lon,latext,lonext,"dd",region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

title = [ 'Conventional Product', 'StandardProduct' ]
file = [ 'MISR_AM1_AGP_P177_F01_24_conv.hdf', 'MISR_AM1_AGP_P177_F01_24.hdf' ]

for j = 0, 1 do begin

    print, 'Testing '+title[j]

    filename = getenv("MTKHOME")+'/../Mtk_testdata/in/'+file[j]
    print, filename

    gridname = "Standard"

    fieldname = "GeoLatitude"
    t = systime(1)
    status = mtk_readdata(filename, gridname, fieldname, region, latbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
    print, fieldname, ' read time: ', systime(1) - t, ' seconds'

    fieldname = "GeoLongitude"
    t = systime(1)
    status = mtk_readdata(filename, gridname, fieldname, region, lonbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
    print, fieldname, ' read time: ', systime(1) - t, ' seconds'


    fieldname = "AveSceneElev"
    t = systime(1)
    status = mtk_readdata(filename, gridname, fieldname, region, img, mapinfo)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
    print, fieldname, ' read time: ', systime(1) - t, ' seconds'

    window,0,xsize=mapinfo.nsample,ysize=mapinfo.nline,title=title[j]
    tvscl,bytscl(img,min=0),/order

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
        print,"        (IDL l,s) = ", l, s

        status = mtk_ls_to_latlon(mapinfo, l, s, lat, lon)
        if (status ne 0) then begin 
            print,"Test Error: ", mtk_error_message(status)
        end
        print,"    (MTK lat,lon) = ", lat, lon, format=fmt

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

        status = mtk_latlon_to_ls(mapinfo, lat, lon, l, s)
        if (status ne 0) then begin 
            print,"Test Error: ", mtk_error_message(status)
        end
        print,"        (MTK l,s) = ", l, s

        status = mtk_latlon_to_bls(mapinfo.path, mapinfo.resolution, $
                                   lat, lon, bp, lp, sp)
        if (status ne 0) then begin 
            print,"Test Error: ", mtk_error_message(status)
        end
        print,"      (MTK b,l,s) = ", bp, lp, sp

        i = i + 1
    endrep until !err eq 4

end

wdelete,0
end
