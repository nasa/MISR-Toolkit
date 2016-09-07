pro idl_test29

path = 39
start_block = 50
end_block = 52

status = mtk_setregion_by_path_blockrange(path, start_block, end_block, region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

status = mtk_create_geogrid(51.0, -114.0, 46.0, -106.0, .01, 0.01, latbuf, lonbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
window,/free,xsize=(size(latbuf))[1],ysize=(size(latbuf))[2],title='Latitude'
tvscl, latbuf,/order
window,/free,xsize=(size(lonbuf))[1],ysize=(size(lonbuf))[2],title='Longitude'
tvscl, lonbuf,/order

; First scene

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf"
gridname = "Standard"
fieldname = "AveSceneElev"

status = mtk_readdata(filename, gridname, fieldname, region, srcbuf, mapinfo)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

status = mtk_transform_coordinates(mapinfo, latbuf, lonbuf, linebuf, samplebuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

status = mtk_resample_nearestneighbor(srcbuf, linebuf, samplebuf, destbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

window,/free,xsize=mapinfo.nsample,ysize=mapinfo.nline,title=file_basename(filename)
tvscl, srcbuf,/order
window,/free,xsize=(size(linebuf))[1],ysize=(size(linebuf))[2],title=file_basename(filename)
tvscl, linebuf,/order
window,/free,xsize=(size(samplebuf))[1],ysize=(size(samplebuf))[2],title=file_basename(filename)
tvscl, samplebuf,/order
window,/free,xsize=(size(destbuf))[1],ysize=(size(destbuf))[2],title=file_basename(filename)
tvscl, destbuf,/order

; Second scene

filename2 = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AGP_P038_F01_24.hdf"

status = mtk_readdata(filename2, gridname, fieldname, region, srcbuf2, mapinfo2)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

status = mtk_transform_coordinates(mapinfo2, latbuf, lonbuf, linebuf2, samplebuf2)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

status = mtk_resample_nearestneighbor(srcbuf2, linebuf2, samplebuf2, destbuf2)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

window,/free,xsize=mapinfo.nsample,ysize=mapinfo.nline,title=file_basename(filename2)
tvscl, srcbuf2,/order
window,/free,xsize=(size(linebuf2))[1],ysize=(size(linebuf2))[2],title=file_basename(filename2)
tvscl, linebuf2,/order
window,/free,xsize=(size(samplebuf2))[1],ysize=(size(samplebuf2))[2],title=file_basename(filename2)
tvscl, samplebuf2,/order
window,/free,xsize=(size(destbuf2))[1],ysize=(size(destbuf2))[2],title=file_basename(filename2)
tvscl, destbuf2,/order

; Third scene

filename3 = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf"

status = mtk_readdata(filename3, gridname, fieldname, region, srcbuf3, mapinfo3)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

status = mtk_transform_coordinates(mapinfo3, latbuf, lonbuf, linebuf3, samplebuf3)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

status = mtk_resample_nearestneighbor(srcbuf3, linebuf3, samplebuf3, destbuf3)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

window,/free,xsize=mapinfo.nsample,ysize=mapinfo.nline,title=file_basename(filename3)
tvscl, srcbuf3,/order
window,/free,xsize=(size(linebuf3))[1],ysize=(size(linebuf3))[2],title=file_basename(filename3)
tvscl, linebuf3,/order
window,/free,xsize=(size(samplebuf3))[1],ysize=(size(samplebuf3))[2],title=file_basename(filename3)
tvscl, samplebuf3,/order
window,/free,xsize=(size(destbuf3))[1],ysize=(size(destbuf3))[2],title=file_basename(filename3)
tvscl, destbuf3,/order

; Fourth scene

filename4 = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AGP_P036_F01_24.hdf"

status = mtk_readdata(filename4, gridname, fieldname, region, srcbuf4, mapinfo4)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

status = mtk_transform_coordinates(mapinfo4, latbuf, lonbuf, linebuf4, samplebuf4)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

status = mtk_resample_nearestneighbor(srcbuf4, linebuf4, samplebuf4, destbuf4)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

window,/free,xsize=mapinfo.nsample,ysize=mapinfo.nline,title=file_basename(filename4)
tvscl, srcbuf4,/order
window,/free,xsize=(size(linebuf4))[1],ysize=(size(linebuf4))[2],title=file_basename(filename4)
tvscl, linebuf4,/order
window,/free,xsize=(size(samplebuf4))[1],ysize=(size(samplebuf4))[2],title=file_basename(filename4)
tvscl, samplebuf4,/order
window,/free,xsize=(size(destbuf4))[1],ysize=(size(destbuf4))[2],title=file_basename(filename4)
tvscl, destbuf4,/order

; Animate scenes

destbufstack = intarr((size(destbuf4))[1],(size(destbuf4))[2], 4)
destbufstack[*,*,0] = destbuf
destbufstack[*,*,1] = destbuf2
destbufstack[*,*,2] = destbuf3
destbufstack[*,*,3] = destbuf4

window,/free,xsize=(size(destbuf4))[1],ysize=(size(destbuf4))[2],title='Nearest Neighbor Resampling'

print, 'Left-Click to advance through scenes'
print, 'Right-click to exit'
i = 0
tvscl, destbufstack[*,*,i mod 4],/order
repeat begin
    cursor,s,l,/device,/down
    lp = (size(latbuf))[2] - l
    tvscl, destbufstack[*,*,i mod 4],/order
    mesgstr = '(' + strtrim(latbuf[s,lp],2) + ', ' + strtrim(lonbuf[s,lp],2) + ') = ' + strtrim(destbufstack[s,lp,i mod 4],2)
    xyouts,s,l,mesgstr,/device
    i = i + 1
endrep until !err eq 4

while (!d.window ne -1) do wdelete,!d.window
end
