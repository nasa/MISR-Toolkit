pro idl_test8

wid = widget_base()

lat = 33.2
lon = -113.5
resolution = '1100 meters'
lat_extent = 1000
lon_extent = 800
status = mtk_setregion_by_latlon_extent(lat, lon, $
                                        lat_extent, lon_extent, $
                                        resolution, region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
help,region
help,/struct,region

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf"
gridname = "RegParamsAer"
fieldname = "RegBestEstimateAngstromExponent"
print,fieldname
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, angbuf, mapinfo)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, systime(1) - t, ' seconds'
help,angbuf
print, min(angbuf), max(angbuf)
t = systime(1)
slide_image,bytscl(angbuf, min=-0.2, max=3.5),/order, $
  show_full=0, xvisible=1024,yvisible=512, $
  title=filename + " / " + gridname + " / " + fieldname, group=wid
print, systime(1) - t, ' seconds to display'

help,mapinfo
help,/struct,mapinfo

print, 'ULC'
linep = 0.0
samplep = 0.0;
status = mtk_ls_to_latlon(mapinfo, linep, samplep, latp, lonp)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, linep, samplep, format='(2f13.5)'
print, latp, lonp
status = mtk_latlon_to_ls(mapinfo, latp, lonp, linep, samplep)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, latp, lonp
print, linep, samplep, format='(2f13.5)'

linep = 0;
samplep = 0;
status = mtk_ls_to_somxy(mapinfo, linep, samplep, som_xp, som_yp)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, linep, samplep, format='(2f13.5)'
print, som_xp, som_yp
status = mtk_somxy_to_ls(mapinfo, som_xp, som_yp, linep, samplep)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, som_xp, som_yp
print, linep, samplep, format='(2f13.5)'


print, 'CTR'
linep = (mapinfo.nline-1)/2.0;
samplep = (mapinfo.nsample-1)/2.0;
status = mtk_ls_to_latlon(mapinfo, linep, samplep, latp, lonp)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, linep, samplep, format='(2f13.5)'
print, latp, lonp
status = mtk_latlon_to_ls(mapinfo, latp, lonp, linep, samplep)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, latp, lonp
print, linep, samplep, format='(2f13.5)'

linep = (mapinfo.nline-1)/2.0;
samplep = (mapinfo.nsample-1)/2.0;
status = mtk_ls_to_somxy(mapinfo, linep, samplep, som_xp, som_yp)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, linep, samplep, format='(2f13.5)'
print, som_xp, som_yp
status = mtk_somxy_to_ls(mapinfo, som_xp, som_yp, linep, samplep)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, som_xp, som_yp
print, linep, samplep, format='(2f13.5)'


print, 'LRC'
linep = mapinfo.nline-1;
samplep = mapinfo.nsample-1;
status = mtk_ls_to_latlon(mapinfo, linep, samplep, latp, lonp)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, linep, samplep, format='(2f13.5)'
print, latp, lonp
status = mtk_latlon_to_ls(mapinfo, latp, lonp, linep, samplep)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, latp, lonp
print, linep, samplep, format='(2f13.5)'

linep = mapinfo.nline-1;
samplep = mapinfo.nsample-1;
status = mtk_ls_to_latlon(mapinfo, linep, samplep, latp, lonp)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_ls_to_somxy(mapinfo, linep, samplep, som_xp, som_yp)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, linep, samplep, format='(2f13.5)'
print, som_xp, som_yp
status = mtk_somxy_to_ls(mapinfo, som_xp, som_yp, linep, samplep)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print, som_xp, som_yp
print, linep, samplep, format='(2f13.5)'

print,"Press return to close all windows..."
done=""
read,done
widget_control,/destroy, wid
end
