pro idl_test15

wid = widget_base()

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_LM_P177_O004194_BA_SITE_EGYPTDESERT_F02_0020_conv.hdf"

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
status = mtk_setregion_by_path_blockrange(path, sb, eb, region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end


gridname = "RedBand"
fieldname = "Red Radiance/RDQI"

status = mtk_readdata(filename, gridname, fieldname, region, buf, map)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

slide_image,bytscl(buf, min=0),/order, $
  show_full=0, xvisible=1024,yvisible=512, $
  title=filename + " / " + gridname + " / " + fieldname, group=wid


filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AGP_P177_F01_24_conv.hdf"
gridname = "Standard"
fieldname = "AveSceneElev"
status = mtk_readdata(filename, gridname, fieldname, region, buf, map)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

slide_image,bytscl(buf, min=0),/order, $
  show_full=0, xvisible=1024,yvisible=512, $
  title=filename + " / " + gridname + " / " + fieldname, group=wid

fieldname = "Block_number"
status = mtk_readdata(filename, gridname, fieldname, region, buf, map)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

slide_image,bytscl(buf, min=0),/order, $
  show_full=0, xvisible=1024,yvisible=512, $
  title=filename + " / " + gridname + " / " + fieldname, group=wid


filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P177_O004194_BA_F02_0020_conv.hdf"
gridname = "RedBand"
fieldname = "Red Radiance/RDQI"
status = mtk_readdata(filename, gridname, fieldname, region, buf, map)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

slide_image,bytscl(buf, min=0),/order, $
  show_full=0, xvisible=1024,yvisible=512, $
  title=filename + " / " + gridname + " / " + fieldname, group=wid

print,"Press return to close all windows..."
done=""
read,done
widget_control,/destroy, wid
end
