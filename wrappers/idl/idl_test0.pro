pro idl_test0

path=37
res = 1100
block = 30

filename = getenv("MTKHOME")+"/../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf"
gridname = "Standard"

fieldname = "GeoLatitude"
status=mtk_readblock(filename, gridname, fieldname, block, latitude)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

fieldname = "GeoLongitude"
status=mtk_readblock(filename, gridname, fieldname, block, longitude)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end


;ULC of block

i = 0 ; line
j = 0 ; sample

print,"ULC:"

print, block,i,j 

print, latitude[j,i], longitude[j,i], format="(2f17.6)"

status=mtk_bls_to_latlon(path, res, block, i, j, lat, lon)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, lat, lon, format="(2f17.6)"

status=mtk_latlon_to_somxy(path, lat, lon, x, y)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, x, y

status=mtk_somxy_to_bls(path, res, x, y, b, l, s)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, b, l, s

status=mtk_latlon_to_bls(path, res, lat, lon, bp, lp, sp)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, bp, lp, sp

status=mtk_latlon_to_bls(path, res, latitude[j,i], longitude[j,i], bp2, lp2, sp2)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, bp2, lp2, sp2
print, bp2, round(lp2), round(sp2)

status=mtk_latlon_to_somxy(path, latitude[j,i], longitude[j,i], xp, yp)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, xp, yp


;CTR of block

i = 128/2 ; line
j = 512/2 ; sample

print,"CTR:"

print, block,i,j 

print, latitude[j,i], longitude[j,i], format="(2f17.6)"

status=mtk_bls_to_latlon(path, res, block, i, j, lat, lon)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, lat, lon, format="(2f17.6)"

status=mtk_latlon_to_somxy(path, lat, lon, x, y)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, x, y

status=mtk_somxy_to_bls(path, res, x, y, b, l, s)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, b, l, s

status=mtk_latlon_to_bls(path, res, lat, lon, bp, lp, sp)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, bp, lp, sp

status=mtk_latlon_to_bls(path, res, latitude[j,i], longitude[j,i], bp2, lp2, sp2)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, bp2, lp2, sp2
print, bp2, round(lp2), round(sp2)

status=mtk_latlon_to_somxy(path, latitude[j,i], longitude[j,i], xp, yp)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, xp, yp


;LRC of block

i = 128-1 ; line
j = 512-1 ; sample

print, "LRC:"

print, block,i,j 

print, latitude[j,i], longitude[j,i], format="(2f17.6)"

status=mtk_bls_to_latlon(path, res, block, i, j, lat, lon)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, lat, lon, format="(2f17.6)"

status=mtk_latlon_to_somxy(path, lat, lon, x, y)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, x, y

status=mtk_somxy_to_bls(path, res, x, y, b, l, s)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, b, l, s

status=mtk_latlon_to_bls(path, res, lat, lon, bp, lp, sp)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, bp, lp, sp

status=mtk_latlon_to_bls(path, res, latitude[j,i], longitude[j,i], $
                     bp2, lp2, sp2)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, bp2, lp2, sp2
print, bp2, round(lp2), round(sp2)

status=mtk_latlon_to_somxy(path, latitude[j,i], longitude[j,i], xp, yp)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, xp, yp

end
