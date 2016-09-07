pro idl_test25

path = 37
sblock = 40
eblock = 50
status = mtk_setregion_by_path_blockrange(path, sblock, eblock, region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

filename_aa = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf"
gridname = "GreenBand"
fieldname = "Green Radiance/RDQI"
status = mtk_readdata(filename_aa, gridname, fieldname, region, buf, mapinfo)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_time_meta_read(filename_aa, timeinfo_aa)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

filename_an = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AN_F03_0024.hdf"
status = mtk_time_meta_read(filename_an, timeinfo_an)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

; first line

l = 0
s = mapinfo.nsample / 2

status = mtk_ls_to_somxy(mapinfo, l, s, somx, somy)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_pixel_time(timeinfo_aa, somx, somy, datetime_aa)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_pixel_time(timeinfo_an, somx, somy, datetime_an)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, "First line"
print, "     line, sample = ", l, s
print, "       somx, somy = ", somx, somy
print, "AA pixel datetime = ", datetime_aa
print, "AN pixel datetime = ", datetime_an

; Center line

l = mapinfo.nline / 2
s = mapinfo.nsample / 2

status = mtk_ls_to_somxy(mapinfo, l, s, somx, somy)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_pixel_time(timeinfo_aa, somx, somy, datetime_aa)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_pixel_time(timeinfo_an, somx, somy, datetime_an)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, "Center line"
print, "     line, sample = ", l, s
print, "       somx, somy = ", somx, somy
print, "AA pixel datetime = ", datetime_aa
print, "AN pixel datetime = ", datetime_an

; Last line

l = mapinfo.nline - 1
s = mapinfo.nsample / 2

status = mtk_ls_to_somxy(mapinfo, l, s, somx, somy)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_pixel_time(timeinfo_aa, somx, somy, datetime_aa)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_pixel_time(timeinfo_an, somx, somy, datetime_an)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, "Last line"
print, "     line, sample = ", l, s
print, "       somx, somy = ", somx, somy
print, "AA pixel datetime = ", datetime_aa
print, "AN pixel datetime = ", datetime_an

; An array of varying samples

npts = 10
l = replicate(mapinfo.nline / 2, npts)
s = indgen(npts) + mapinfo.nsample / 2

status = mtk_ls_to_somxy(mapinfo, l, s, somx, somy)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_pixel_time(timeinfo_aa, somx, somy, datetime_aa)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_pixel_time(timeinfo_an, somx, somy, datetime_an)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

for i = 0, npts-1 do begin
    print, "Sample ", i
    print, "     line, sample = ", l[i], s[i]
    print, "       somx, somy = ", somx[i], somy[i]
    print, "AA pixel datetime = ", datetime_aa[i]
    print, "AN pixel datetime = ", datetime_an[i]
end

; An array of varying lines

npts = 10
l = indgen(npts) + mapinfo.nline / 2
s = replicate(mapinfo.nsample / 2, npts)

status = mtk_ls_to_somxy(mapinfo, l, s, somx, somy)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_pixel_time(timeinfo_aa, somx, somy, datetime_aa)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_pixel_time(timeinfo_an, somx, somy, datetime_an)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

for i = 0, npts-1 do begin
    print, "Line ", i
    print, "     line, sample = ", l[i], s[i]
    print, "       somx, somy = ", somx[i], somy[i]
    print, "AA pixel datetime = ", datetime_aa[i]
    print, "AN pixel datetime = ", datetime_an[i]
end

end
