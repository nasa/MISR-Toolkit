pro idl_test2

wid = widget_base()

lat = 33.2
lon = -113.5
status = mtk_latlon_to_pathlist(lat, lon, pathcnt, pathlist)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,lat,lon
print,pathcnt
print,pathlist

lat_ext = 1100000
lon_ext = 1000000
status = mtk_setregion_by_latlon_extent(lat, lon, lat_ext, lon_ext, "m", region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
help,region
help,/struct,region

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AGP_P036_F01_24.hdf"
gridname = "Standard"
fieldname = "AveSceneElev"
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, elevbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds'
help,elevbuf
print, min(elevbuf), max(elevbuf)
t = systime(1)
slide_image,bytscl(elevbuf),/order,show_full=0, xvisible=1024,yvisible=512, $
 title=filename + " / " + gridname + " / " + fieldname, group=wid
print, systime(1) - t, ' seconds to display'

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf"
gridname = "Standard"
fieldname = "AveSceneElev"
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, elevbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds'
help,elevbuf
print, min(elevbuf), max(elevbuf)
t = systime(1)
slide_image,bytscl(elevbuf),/order,show_full=0, xvisible=1024,yvisible=512, $
 title=filename + " / " + gridname + " / " + fieldname, group=wid
print, systime(1) - t, ' seconds to display'

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AGP_P038_F01_24.hdf"
gridname = "Standard"
fieldname = "AveSceneElev"
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, elevbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds'
help,elevbuf
print, min(elevbuf), max(elevbuf)
t = systime(1)
slide_image,bytscl(elevbuf),/order,show_full=0, xvisible=1024,yvisible=512, $
 title=filename + " / " + gridname + " / " + fieldname, group=wid
print, systime(1) - t, ' seconds to display'

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf"
gridname = "Standard"
fieldname = "AveSceneElev"
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, elevbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds'
help,elevbuf
print, min(elevbuf), max(elevbuf)
t = systime(1)
slide_image,bytscl(elevbuf),/order,show_full=0, xvisible=1024,yvisible=512, $
 title=filename + " / " + gridname + " / " + fieldname, group=wid
print, systime(1) - t, ' seconds to display'

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_TC_STEREO_P037_O029058_F07_0013.hdf"
gridname = "SubregParams"
fieldname = "StereoHeight_BestWinds"
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, sthtbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds'
help,sthtbuf
print, min(sthtbuf), max(sthtbuf)
t = systime(1)
slide_image,bytscl(sthtbuf),/order,show_full=0, xvisible=1024,yvisible=512, $
 title=filename + " / " + gridname + " / " + fieldname, group=wid
print, systime(1) - t, ' seconds to display'

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf"
gridname = "RegParamsAer"
fieldname = "RegBestEstimateSpectralOptDepth[2]"
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, odbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds'
help,odbuf
print, min(odbuf), max(odbuf)
t = systime(1)
slide_image,bytscl(odbuf),/order,show_full=0, xvisible=1024,yvisible=512, $
 title=filename + " / " + gridname + " / " + fieldname, group=wid
print, systime(1) - t, ' seconds to display'


filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf"
gridname = "GeometricParameters"
fieldname = "SolarAzimuth"
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, solazmbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds'
help,solazmbuf
print, min(solazmbuf), max(solazmbuf)
t = systime(1)
slide_image,bytscl(solazmbuf),/order,show_full=0, xvisible=1024,yvisible=512, $
 title=filename + " / " + gridname + " / " + fieldname, group=wid
print, systime(1) - t, ' seconds to display'

gridname = "GeometricParameters"
fieldname = "SolarZenith"
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, solzenbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds'
help,solzenbuf
print, min(solzenbuf), max(solzenbuf)
t = systime(1)
slide_image,bytscl(solzenbuf),/order,show_full=0, xvisible=1024,yvisible=512, $
 title=filename + " / " + gridname + " / " + fieldname, group=wid
print, systime(1) - t, ' seconds to display'

gridname = "GreenBand"
fieldname = "Green Radiance/RDQI"
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, grnbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds'
help,grnbuf
print, min(grnbuf), max(grnbuf)
slide_image,bytscl(grnbuf),/order,show_full=0, xvisible=1024,yvisible=512, $
 title=filename + " / " + gridname + " / " + fieldname, group=wid
print, systime(1) - t, ' seconds to display'

gridname = "BlueBand"
fieldname = "Blue Radiance/RDQI"
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, blubuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds'
help,blubuf
print, min(blubuf), max(blubuf)
t = systime(1)
slide_image,bytscl(blubuf),/order,show_full=0, xvisible=1024,yvisible=512, $
 title=filename + " / " + gridname + " / " + fieldname, group=wid
print, systime(1) - t, ' seconds to display'

gridname = "NIRBand"
fieldname = "NIR Radiance/RDQI"
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, nirbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds'
help,nirbuf
print, min(nirbuf), max(nirbuf)
t = systime(1)
slide_image,bytscl(nirbuf),/order,show_full=0, xvisible=1024,yvisible=512, $
 title=filename + " / " + gridname + " / " + fieldname, group=wid
print, systime(1) - t, ' seconds to display'

gridname = "RedBand"
fieldname = "Red Radiance/RDQI"
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, redbuf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds'
help,redbuf
print, min(redbuf), max(redbuf)
t = systime(1)
slide_image,bytscl(redbuf),/order,show_full=0, xvisible=1024,yvisible=512, $
 title=filename + " / " + gridname + " / " + fieldname, group=wid
print, systime(1) - t, ' seconds to display'

print,"Press return to close all windows..."
done=""
read,done
widget_control,/destroy, wid
end
