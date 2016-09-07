pro idl_test20

orbit = 30000

status = mtk_orbit_to_timerange(orbit,st,et)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, "Orbit = ", orbit
print, "start time = ", st
print, "end time = ", et

orbit = 3000

status = mtk_orbit_to_timerange(orbit,st,et)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, "Orbit = ", orbit
print, "start time = ", st
print, "end time = ", et
end
