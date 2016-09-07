pro idl_test17

dd = [-85.09283d0, 23.54d0, 23.55d0, 23.56d0]

status = mtk_dd_to_deg_min_sec(dd, deg, min, sec)
print, "mtk_dd_to_deg_min_sec status = ", status
status = mtk_dd_to_dms(dd, dms)
print, "mtk_dd_to_dms status = ", status
status = mtk_dd_to_rad(dd, rad)
print, "mtk_dd_to_rad status = ", status

print, "dd =  ", dd
print, "deg = ", deg
print, "min = ", min
print, "sec = ", sec
print, "dms = ", dms, format='(a,4f16.4)'
print, "rad = ", rad

status = mtk_deg_min_sec_to_dd(deg, min, sec, dd)
print, "mtk_deg_min_sec_to_dd status = ", status
status = mtk_deg_min_sec_to_dms(deg, min, sec, dms)
print, "mtk_deg_min_sec_to_dms status = ", status
status = mtk_deg_min_sec_to_rad(deg, min, sec, rad)
print, "mtk_deg_min_sec_to_rad status = ", status

print, "dd =  ", dd
print, "deg = ", deg
print, "min = ", min
print, "sec = ", sec
print, "dms = ", dms, format='(a,4f16.4)'
print, "rad = ", rad

status = mtk_dms_to_dd(dms, dd)
print, "mtk_dms_to_dd status = ", status
status = mtk_dms_to_deg_min_sec(dms, deg, min, sec)
print, "mtk_dms_to_deg_min_sec status = ", status
status = mtk_dms_to_rad(dms, rad)
print, "mtk_dms_to_rad status = ", status

print, "dd =  ", dd
print, "deg = ", deg
print, "min = ", min
print, "sec = ", sec
print, "dms = ", dms, format='(a,4f16.4)'
print, "rad = ", rad

status = mtk_rad_to_dd(rad, dd)
print, "mtk_rad_to_dd status = ", status
status = mtk_rad_to_deg_min_sec(rad, deg, min, sec)
print, "mtk_rad_to_deg_min_sec status = ", status
status = mtk_rad_to_dms(rad, dms)
print, "mtk_rad_to_dms status = ", status

print, "dd =  ", dd
print, "deg = ", deg
print, "min = ", min
print, "sec = ", sec
print, "dms = ", dms, format='(a,4f16.4)'
print, "rad = ", rad

end
