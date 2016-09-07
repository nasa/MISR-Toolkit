pro idl_test23

filepath = getenv("MTKHOME")+"/../Mtk_testdata/in"
filename = dialog_pickfile(PATH=filepath)

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

status = mtk_file_to_path(filename, path)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_file_to_blockrange(filename, sb, eb)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print
print, 'Enter start block (', strtrim(sb,2), '):'
read, nsb
print, 'Enter end block (', strtrim(eb,2), '):'
read, neb

status = mtk_setregion_by_path_blockrange(path, nsb, neb, region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print,"Reading data..."
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, buf, map)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
if (status ne 0) then stop
print, systime(1) - t, ' seconds

print,"Writing data..."
envifilename = getenv("MTKHOME")+"/../Mtk_testdata/out/envidata"
t = systime(1)
status = mtk_write_envi_file(envifilename, buf, map, filename, gridname, fieldname)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds
print,"Wrote envi file to ", envifilename

print,"Creating lat/lon data..."
t = systime(1)
status = mtk_create_latlon(map, latbuf, lonbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds
print,"Done creating lat/lon data..."

print,"Writing latitude data..."
envilatfilename = getenv("MTKHOME")+"/../Mtk_testdata/out/envilat"
t = systime(1)
status = mtk_write_envi_file(envilatfilename, latbuf, map, filename, gridname, "Latitude")
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds
print,"Wrote envi latitude file to ", envilatfilename

print,"Writing longitude data..."
envilonfilename = getenv("MTKHOME")+"/../Mtk_testdata/out/envilon"
t = systime(1)
status = mtk_write_envi_file(envilonfilename, lonbuf, map, filename, gridname, "Longitude")
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds
print,"Wrote envi longitude file to ", envilonfilename

end
