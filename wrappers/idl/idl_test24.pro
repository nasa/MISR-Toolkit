pro idl_test24

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf"

status = mtk_file_block_meta_list(filename, nblockmeta, blockmetalist)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, "filename = ", filename
print, "nblockmeta = ", nblockmeta
print, "blockmetalist = ", blockmetalist

i = 1
status = mtk_file_block_meta_field_list(filename, blockmetalist[i], nfield, fieldlist)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, "For ", blockmetalist[i], ":" 
print, "   nfield = ", nfield
print, "   fieldlist = ", fieldlist

j = 2
status = mtk_file_block_meta_field_read(filename, blockmetalist[i], fieldlist[j], buf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, "   For ", fieldlist[j]
print, "       buf = ", buf

j = 1
status = mtk_file_block_meta_field_read(filename, blockmetalist[i], fieldlist[j], buf)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, "   For ", fieldlist[j]
print, "       buf = ", string(buf)

end
