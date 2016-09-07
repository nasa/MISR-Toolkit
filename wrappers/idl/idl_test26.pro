pro idl_test26

status = mtk_setregion_by_path_blockrange(37, 45, 45, region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf"
gridname = "SubregParamsLnd"
fieldname = "LAIDelta1[1]"
status = mtk_readdata(filename, gridname, fieldname, region, data)
print, fieldname, " status: ", mtk_error_message(status)

fieldname = "LAIDelta1"
status = mtk_readdata(filename, gridname, fieldname, region, data)
print, fieldname, " status: ", mtk_error_message(status)

fieldname = "LAIDelta1[200]"
status = mtk_readdata(filename, gridname, fieldname, region, data)
print, fieldname, " status: ", mtk_error_message(status)

fieldname = "LAIDelta1[1][0]"
status = mtk_readdata(filename, gridname, fieldname, region, data)
print, fieldname, " status: ", mtk_error_message(status)

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf"
gridname = "RedBand"
fieldname = "Red Radiance/RDQI"
status = mtk_readdata(filename, gridname, fieldname, region, data)
print, fieldname, " status: ", mtk_error_message(status)

fieldname = "Red Radiance/RDQI[0]"
status = mtk_readdata(filename, gridname, fieldname, region, data)
print, fieldname, " status: ", mtk_error_message(status)

fieldname = "Red Brf"
status = mtk_readdata(filename, gridname, fieldname, region, data)
print, fieldname, " status: ", mtk_error_message(status)

fieldname = "Red Brf[0]"
status = mtk_readdata(filename, gridname, fieldname, region, data)
print, fieldname, " status: ", mtk_error_message(status)

fieldname = "Brf"
status = mtk_readdata(filename, gridname, fieldname, region, data)
print, fieldname, " status: ", mtk_error_message(status)

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/badfile.hdf"
gridname = "RedBand"
fieldname = "Red Radiance/RDQI"
status = mtk_readdata(filename, gridname, fieldname, region, data)
print, filename, " status: ", mtk_error_message(status)

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf"
gridname = "badgrid"
fieldname = "Red Radiance/RDQI"
status = mtk_readdata(filename, gridname, fieldname, region, data)
print, gridname, " status: ", mtk_error_message(status)

print,"List of all the Mtk error/status messages:"
for i = 0, 70 do begin
  print,i,') ',mtk_error_message(i)
endfor

end
