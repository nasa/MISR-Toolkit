;
; Cross reference a latitude and longitude with AGP
;
pro xref_latlon, lat, lon, ROOTDIR=agp_rootdir

if (keyword_set(agp_rootdir) eq 0) then agp_rootdir = '/data/bank/anc/AGP/database'

status = mtk_latlon_to_pathlist(lat, lon, npath, pathlist)
check_status, 'MTK_LATLON_TO_PATHLIST(): ', status

print, 'Path list = ',pathlist

status = mtk_setregion_by_latlon_extent(lat, lon, 2, 2, '1.1km', region)
check_status, 'MTK_SETREGION_BY_LATLON_EXTENT(): ', status

for i = 0, npath-1 do begin
    agp_dir = string(format='(%"%s/path%03d")',agp_rootdir,pathlist[i])

    status = mtk_make_filename(agp_dir, 'AGP', '', string(pathlist[i]), '', 'F01_24', filename)
    check_status, 'MTK_MAKE_FILENAME(): ', status

    status = mtk_readdata(filename, 'Standard', 'GeoLatitude', region, latbuf, latmap)
    check_status, 'MTK_READDATA() latitude: ', status

    status = mtk_latlon_to_ls(latmap, lat, lon, latline, latsample)
    check_status, 'MTK_LATLON_TO_LS(): ', status

    status = mtk_ls_to_latlon(latmap, latline, latsample, lat2, lon2)
    check_status, 'MTK_LS_TO_LATLON(): ', status

    status = mtk_readdata(filename, 'Standard', 'GeoLongitude', region, lonbuf, lonmap)
    check_status, 'MTK_READDATA() longitude: ', status

    status = mtk_latlon_to_ls(lonmap, lat, lon, lonline, lonsample)
    check_status, 'MTK_LATLON_TO_LS(): ', status

    status = mtk_ls_to_latlon(latmap, lonline, lonsample, lat3, lon3)
    check_status, 'MTK_LS_TO_LATLON(): ', status

    print, filename
    print
    print, format='("          Input Latitude/Longitude: ",f11.5,f11.5)', lat, lon
    print, format='("   MTK Latitude buffer Line/Sample: ",f11.5,f11.5)', latline, latsample
    print, format='("            MTK Latitude/Longitude: ",f11.5,f11.5)', lat2, lon2
    print, format='("  MTK Longitude buffer Line/Sample: ",f11.5,f11.5)', lonline, lonsample
    print, format='("            MTK Latitude/Longitude: ",f11.5,f11.5)', lat3, lon3

    print, '  AGP GeoLatidue/GeoLongitude data plane:'
    print, format='("    (",f9.5,",",f11.5,")   (",f9.5,",",f11.5,")")', $
           latbuf[floor(latsample),floor(latline)], lonbuf[floor(lonsample),floor(lonline)], $
           latbuf[ceil(latsample),floor(latline)], lonbuf[ceil(lonsample),floor(lonline)]
    print, format='("                (",f9.5,",",f11.5,")")', lat, lon 
    print, format='("    (",f9.5,",",f11.5,")   (",f9.5,",",f11.5,")")', $
           latbuf[floor(latsample),ceil(latline)], lonbuf[floor(lonsample),ceil(lonline)], $
           latbuf[ceil(latsample),ceil(latline)], lonbuf[ceil(lonsample),ceil(lonline)]

    status = mtk_latlon_to_bls(pathlist[i], 1100, lat, lon, b, l, s)
    check_status, 'MTK_LATLON_TO_BLS(): ', status

    status = mtk_bls_to_latlon(pathlist[i], 1100, b, l, s, lat4, lon4)
    check_status, 'MTK_BLS_TO_LATLON(): ', status

    status = mtk_readblockrange(filename,'Standard','GeoLatitude', b, b, latstack)
    check_status, 'MTK_READBLOCKRANGE() latitude: ', status

    status = mtk_readblockrange(filename,'Standard','GeoLongitude', b, b, lonstack)
    check_status, 'MTK_READBLOCKRANGE() longitude: ', status

    print
    print, format='("  Input Latitude/Longitude: ",f11.5,f11.5)', lat, lon
    print, format='("     MTK Block/Line/Sample: ",i4,f11.5,f11.5)', b, l, s
    print, format='("    MTK Latitude/Longitude: ",f11.5,f11.5)', lat3, lon3

    nline = float((size(latstack))(2))
    nsample = float((size(latstack))(1))

    print, '  AGP GeoLatidue/GeoLongitude stacked blocks:'
    if ((l gt 0.0 and l lt nline-1) and (s gt 0 and s lt nsample-1)) then begin
        print, format='("    (",f9.5,",",f11.5,")   (",f9.5,",",f11.5,")")', $
               latstack[floor(s),floor(l)], lonstack[floor(s),floor(l)], $
               latstack[ceil(s),floor(l)], lonstack[ceil(s),floor(l)]
        print, format='("                (",f9.5,",",f11.5,")")', lat, lon 
        print, format='("    (",f9.5,",",f11.5,")   (",f9.5,",",f11.5,")")', $
               latstack[floor(s),ceil(l)], lonstack[floor(s),ceil(l)], $
               latstack[ceil(s),ceil(l)], lonstack[ceil(s),ceil(l)]
    endif else begin
        print, "Line, sample (", strtrim(l,2), ", ", strtrim(s,2), $
               ") are off the edge of read block ", strtrim(b,2)
    endelse

    print

endfor

end
