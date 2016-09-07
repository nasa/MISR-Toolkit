pro idl_test12

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf"
gridname = "RedBand"
fieldname = "Red Radiance/RDQI"

status = mtk_file_to_path(filename, path)
status = mtk_file_to_orbit(filename, orbit)
status = mtk_file_to_blockrange(filename, sb, eb)
status = mtk_file_grid_to_resolution(filename, gridname, res)
status = mtk_path_to_projparam(path, res, pp)
status = mtk_file_to_gridlist(filename, gridcnt, gridlist)
status = mtk_file_grid_to_fieldlist(filename, gridname, fieldcnt, fieldlist)
status = mtk_file_grid_to_native_fieldlist(filename, gridname, nativefieldcnt, nativefieldlist)
status = mtk_file_grid_field_to_dimlist(filename, gridname, fieldname, dimcnt, dimlist, dimsize)
status = mtk_file_grid_field_to_datatype(filename, gridname, fieldname, datatype)
field_valid = mtk_file_grid_field_check(filename, gridname, fieldname)
status = mtk_file_lgid(filename, lgid)
status = mtk_file_version(filename, version)
status = mtk_file_type(filename, filetype)
status = mtk_file_coremetadata_query(filename, ncoremeta, coremeta)
status = mtk_file_coremetadata_get(filename, "RANGEBEGINNINGDATE", begin_date)
status = mtk_file_coremetadata_get(filename, "RANGEENDINGDATE", end_date)
status = mtk_fileattr_list(filename, fileattrcnt, fileattrlist)
status = mtk_fileattr_get(filename, "Path_number" ,path_num)
status = mtk_gridattr_list(filename, gridname, gridattrcnt, gridattrlist)
status = mtk_gridattr_get(filename, gridname, "Scale factor", scale_fac)
status = mtk_fillvalue_get(filename, gridname, fieldname, fillvalue)

print,"file = ",filename
print,"lgid = ", lgid
print,"version = ", version
print,"filetype = ", filetype
print,"datatype = ", datatype
print,"field valid = ",  mtk_error_message(field_valid)
print,"grid = ",gridname
print,"field = ",fieldname
print,"path number = ",path_num
print,"path = ",path
print,"orbit = ",orbit
print,"res = ", res
print,"start block = ",sb
print,"end block = ",eb
print,"gridcnt = ", gridcnt
print,"gridlist = ", gridlist
print,"fieldcnt = ", fieldcnt
print,"fieldlist = ", fieldlist
print,"nativefieldlist = ", nativefieldlist
print,"dimcnt = ", dimcnt
if (dimcnt gt 0) then print,"dimlist = ", dimlist
if (dimcnt gt 0) then print,"dimsize = ", dimsize
print, "fillvalue = ", fillvalue
print, "File attribute count = ", fileattrcnt
print, "File attribute list = ", fileattrlist
print, "Grid attribute count = ", gridattrcnt
print, "Grid attribute list = ", gridattrlist
print,"number coremeta data entries = ", ncoremeta
print,"coremeta data entries = ", coremeta

print,"Beginning date = ", begin_date
print,"Ending date = ", end_date

print,"Grid scale factor = ", scale_fac

print,"pp.ulc_block1 = ",pp.ulc_block1
print,"pp.lrc_block1 = ",pp.lrc_block1
print,"pp.projparam = ", pp.projparam
print,"pp.reloffset = ", pp.reloffset

help,/struct,pp


filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AGP_P177_F01_24.hdf"
gridname = "Standard"
fieldname = "AveSceneElev"

status = mtk_file_to_path(filename, path)
status = mtk_file_to_blockrange(filename, sb, eb)
status = mtk_file_grid_to_resolution(filename, gridname, res)
status = mtk_path_to_projparam(path, res, pp)
status = mtk_file_to_gridlist(filename, gridcnt, gridlist)
status = mtk_file_grid_to_fieldlist(filename, gridname, fieldcnt, fieldlist)
status = mtk_file_grid_to_native_fieldlist(filename, gridname, nativefieldcnt, nativefieldlist)
status = mtk_file_grid_field_to_dimlist(filename, gridname, fieldname, dimcnt, dimlist, dimsize)
status = mtk_file_grid_field_to_datatype(filename, gridname, fieldname, datatype)
field_valid = mtk_file_grid_field_check(filename, gridname, fieldname)
status = mtk_file_lgid(filename, lgid)
status = mtk_file_version(filename, version)
status = mtk_file_type(filename, filetype)
status = mtk_file_coremetadata_query(filename, ncoremeta, coremeta)
status = mtk_file_coremetadata_get(filename, "RANGEBEGINNINGDATE", begin_date)
status = mtk_file_coremetadata_get(filename, "RANGEENDINGDATE", end_date)
status = mtk_fileattr_list(filename, fileattrcnt, fileattrlist)
status = mtk_fileattr_get(filename, "Path_number" ,path_num)
status = mtk_gridattr_list(filename, gridname, gridattrcnt, gridattrlist)
status = mtk_gridattr_get(filename, gridname, "Block_size.resolution_x", bs )

print,"file = ",filename
print,"lgid = ", lgid
print,"version = ", version
print,"filetype = ", filetype
print,"datatype = ", datatype
print,"field valid = ",  mtk_error_message(field_valid)
print,"grid = ",gridname
print,"field = ",fieldname
print,"path number = ",path_num
print,"path = ",path
print,"res = ", res
print,"start block = ",sb
print,"end block = ",eb
print,"gridcnt = ", gridcnt
print,"gridlist = ", gridlist
print,"fieldcnt = ", fieldcnt
print,"fieldlist = ", fieldlist
print,"nativefieldlist = ", nativefieldlist
print,"dimcnt = ", dimcnt
if (dimcnt gt 0) then print,"dimlist = ", dimlist
if (dimcnt gt 0) then print,"dimsize = ", dimsize
print, "File attribute count = ", fileattrcnt
print, "File attribute list = ", fileattrlist
print, "Grid attribute count = ", gridattrcnt
print, "Grid attribute list = ", gridattrlist
print,"number coremeta data entries = ", ncoremeta
print,"coremeta data entries = ", coremeta

print,"Beginning date = ", begin_date
print,"Ending date = ", end_date

print,"Grid block size = ", bs

print,"pp.ulc_block1 = ",pp.ulc_block1
print,"pp.lrc_block1 = ",pp.lrc_block1
print,"pp.projparam = ", pp.projparam
print,"pp.reloffset = ", pp.reloffset

help,/struct,pp


filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf"
gridname = "SubregParamsLnd"
fieldname = "LAIDelta1[1]"

status = mtk_file_to_path(filename, path)
status = mtk_file_to_orbit(filename, orbit)
status = mtk_file_to_blockrange(filename, sb, eb)
status = mtk_file_grid_to_resolution(filename, gridname, res)
status = mtk_path_to_projparam(path, res, pp)
status = mtk_file_to_gridlist(filename, gridcnt, gridlist)
status = mtk_file_grid_to_fieldlist(filename, gridname, fieldcnt, fieldlist)
status = mtk_file_grid_to_native_fieldlist(filename, gridname, nativefieldcnt, nativefieldlist)
status = mtk_file_grid_field_to_dimlist(filename, gridname, fieldname, dimcnt, dimlist, dimsize)
status = mtk_file_grid_field_to_datatype(filename, gridname, fieldname, datatype)
field_valid = mtk_file_grid_field_check(filename, gridname, fieldname)
status = mtk_file_lgid(filename, lgid)
status = mtk_file_version(filename, version)
status = mtk_file_type(filename, filetype)
status = mtk_file_coremetadata_query(filename, ncoremeta, coremeta)
status = mtk_file_coremetadata_get(filename, "RANGEBEGINNINGDATE", begin_date)
status = mtk_file_coremetadata_get(filename, "RANGEENDINGDATE", end_date)
status = mtk_fileattr_list(filename, fileattrcnt, fileattrlist)
status = mtk_fileattr_get(filename, "Path_number" ,path_num)
status = mtk_gridattr_list(filename, gridname, gridattrcnt, gridattrlist)
status = mtk_gridattr_get(filename, gridname, "Overflow LandHDRF", landhdrf )
status = mtk_fillvalue_get(filename, gridname, fieldname, fillvalue)

print,"file = ",filename
print,"lgid = ", lgid
print,"version = ", version
print,"filetype = ", filetype
print,"datatype = ", datatype
print,"field valid = ", mtk_error_message(field_valid)
print,"grid = ",gridname
print,"field = ",fieldname
print,"path number = ",path_num
print,"path = ",path
print,"orbit = ",orbit
print,"res = ", res
print,"start block = ",sb
print,"end block = ",eb
print,"gridcnt = ", gridcnt
print,"gridlist = ", gridlist
print,"fieldcnt = ", fieldcnt
print,"fieldlist = ", fieldlist
print,"nativefieldlist = ", nativefieldlist
print,"dimcnt = ", dimcnt
if (dimcnt gt 0) then print,"dimlist = ", dimlist
if (dimcnt gt 0) then print,"dimsize = ", dimsize
print, "fillvalue = ", fillvalue
print, "File attribute count = ", fileattrcnt
print, "File attribute list = ", fileattrlist
print, "Grid attribute count = ", gridattrcnt
print, "Grid attribute list = ", gridattrlist
print,"number coremeta data entries = ", ncoremeta
print,"coremeta data entries = ", coremeta

print,"Beginning date = ", begin_date
print,"Ending date = ", end_date

print,"Overflow Value for Land HDRF = ", landhdrf

print,"pp.ulc_block1 = ",pp.ulc_block1
print,"pp.lrc_block1 = ",pp.lrc_block1
print,"pp.projparam = ", pp.projparam
print,"pp.reloffset = ", pp.reloffset

help,/struct,pp


basedir = getenv("MTKHOME")+"/../Mtk_testdata"
product = 'AGP'
camera = ''
path = 177
orbit = 0
pathstr = '177'
orbitstr = ''
version = 'F01_24'

status = mtk_make_filename(basedir+'/in', product, camera, path, orbit, version, filename)
;status = mtk_find_filelist(basedir, product, camera, pathstr, orbitstr, version, filecnt, filelist)

print, filename
;print, filecnt
;print, filelist

basedir = getenv("MTKHOME")+"/../Mtk_testdata"
product = 'GRP_ELLIPSOID_GM'
camera = '.*'
path = 37
orbit = 29058
pathstr = '037'
orbitstr = '029058'
version = 'F03_0024'

status = mtk_make_filename(basedir+'/in', product, camera, path, orbit, version, filename)
;status = mtk_find_filelist(basedir, product, camera, pathstr, orbitstr, version, filecnt, filelist)

print, filename
;print, filecnt
;print, filelist

end
