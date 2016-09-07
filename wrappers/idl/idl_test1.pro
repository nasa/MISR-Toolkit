pro idl_test1

print, "Testing mtk_latlon_to_pathlist..."
status = mtk_latlon_to_pathlist(66.25887d0, 109.9549d0, pathcnt, pathlist)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,pathcnt
print,pathlist

print, "Testing mtk_latlon_to_pathlist..."
lat = 23.4
lon = -87.3
status = mtk_latlon_to_pathlist(lat, lon, pathcnt, pathlist)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,lat,lon
print,pathcnt
print,pathlist

print, "Testing mtk_bls_to_latlon..."
status = mtk_bls_to_latlon(1, 1100, 1, 0, 0, lat, lon)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,lat,lon

print, "Testing mtk_latlon_to_bls..."
status = mtk_latlon_to_bls(1, 1100, lat, lon, b, l, s)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,b,l,s

print, "Testing mtk_bls_to_latlon..."
b = [1,2,3,4]
l = [10,10,10,10.5]
s = [20,20,20,20.5]
status = mtk_bls_to_latlon(1, 1100, b, l, s, lat, lon)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,b,l,s
print,lat,lon

print, "Testing mtk_latlon_to_bls..."
status = mtk_latlon_to_bls(1, 1100, lat, lon, b, l, s)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,lat,lon
print,b,l,s

print, "Testing mtk_latlon_to_bls..."
lat = 66.25887
lon = 109.9549
status = mtk_latlon_to_bls(1, 1100, lat, lon, b, l, s)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,lat,lon
print,b,l,s

print, "Testing mtk_bls_to_latlon..."
status = mtk_bls_to_latlon(1, 1100, b, l, s, lat, lon)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,b,l,s
print,lat,lon

print, "Testing mtk_bls_to_latlon..."
b = indgen(180)+1
l = replicate(256,180)
s = replicate(1024,180)
status = mtk_bls_to_latlon(19,275, b, l, s, lat, lon)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
for i = 0, 180-1 do begin
    print,b[i],l[i],s[i],lat[i],lon[i]
endfor

print, "Testing mtk_setregion_by_latlon_extent..."
lat = 16.25887
lon = 109.9549
status = mtk_setregion_by_latlon_extent(lat, lon, 2*2750000, 2*2750000, "m", region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
help,region
help,/struct,region

status = mtk_region_to_pathlist(region, pathcnt, pathlist)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,pathcnt
print,pathlist

lat = 66.25887
lon = 109.9549
status = mtk_setregion_by_latlon_extent(lat, lon, 275, 275, "km", region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
help,region
help,/struct,region

status = mtk_region_to_pathlist(region, pathcnt, pathlist)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,pathcnt
print,pathlist

lat = 66.25887
lon = -109.9549
status = mtk_setregion_by_latlon_extent(lat, lon, 275, 275, "kilometers", region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
help,region
help,/struct,region

status = mtk_region_to_pathlist(region, pathcnt, pathlist)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,pathcnt
print,pathlist

lat = -66.25887
lon = 109.9549
status = mtk_setregion_by_latlon_extent(lat, lon, 275000, 275000, "meters", region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
help,region
help,/struct,region

status = mtk_region_to_pathlist(region, pathcnt, pathlist)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,pathcnt
print,pathlist

path = 37
sblock = 40
eblock = 50
status = mtk_setregion_by_path_blockrange(path, sblock, eblock, region)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
help,region
help,/struct,region

status = mtk_region_to_pathlist(region, pathcnt, pathlist)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
print,pathcnt
print,pathlist

end
