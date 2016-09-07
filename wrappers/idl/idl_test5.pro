pro idl_test5

wid = widget_base()

path = 37
sblock = 40
eblock = 45
status = mtk_setregion_by_path_blockrange(path, sblock, eblock, region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
help,region
help,/struct,region

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf"

;gridname = "DomParamsAer"
;fieldname = "DomMeanOptDepth"
;print,fieldname
;t = systime(1)
;status = mtk_readdata(filename, gridname, fieldname, region, modbuf)
;if (status ne 0) then begin 
;    print,"Test Error: ", mtk_error_message(status)
;    stop
;end
;print, systime(1) - t, ' seconds'
;help,modbuf
;print, min(modbuf), max(modbuf)
;t = systime(1)
;slide_image,bytscl(modbuf),/order,show_full=0, xvisible=1024,yvisible=512, $
;  title=filename + " / " + gridname + " / " + fieldname, group=wid
;print, systime(1) - t, ' seconds to display'

gridname = "RegParamsAer"
fieldname = "RegBestEstimateAngstromExponent"
print,fieldname
t = systime(1)
status = mtk_readdata(filename, gridname, fieldname, region, angbuf)
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

for band = 0, 3 do begin
    gridname = "RegParamsAer"
    fieldname = "RegBestEstimateSpectralSSA[" + strtrim(band,2) + "]"
    print,fieldname
    t = systime(1)
    status = mtk_readdata(filename, gridname, fieldname, region, ssabuf)
    if (status ne 0) then begin 
        print,"Test Error: ", mtk_error_message(status)
        stop
    end
    print, systime(1) - t, ' seconds'
    help,ssabuf
    print, min(ssabuf), max(ssabuf)
    t = systime(1)
    slide_image,bytscl(ssabuf, min=0.8, max=1.0),/order, $
      show_full=0, xvisible=1024,yvisible=512, $
      title=filename + " / " + gridname + " / " + fieldname, group=wid
    print, systime(1) - t, ' seconds to display'
endfor

for band = 0, 3 do begin
    gridname = "RegParamsAer"
    fieldname = "RegBestEstimateSpectralOptDepth[" + strtrim(band,2) + "]"
    print,fieldname
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
endfor

for cam = 2, 3 do begin
    for band = 1, 2 do begin
        gridname = "SubregParamsAer"
        fieldname = "RetrAppMask[" + strtrim(band,2) + "]["  + strtrim(cam,2) + "]"
        print,fieldname
        t = systime(1)
        status = mtk_readdata(filename, gridname, fieldname, region, maskbuf)
        if (status ne 0) then begin 
            print,"Test Error: ", mtk_error_message(status)
            stop
        end
        print, systime(1) - t, ' seconds'
        help,maskbuf
        print, min(maskbuf), max(maskbuf)
        t = systime(1)
        slide_image,bytscl(maskbuf, max=20),/order, $
          show_full=0, xvisible=1024,yvisible=512, $
          title=filename + " / " + gridname + " / " + fieldname, group=wid
        print, systime(1) - t, ' seconds to display'
    endfor
endfor

print,"Press return to close all windows..."
done=""
read,done
widget_control,/destroy, wid
end
