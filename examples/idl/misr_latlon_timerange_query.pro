pro misr_latlon_timerange_query, lat, lon, stime, etime

if (n_params() ne 4) then begin
    print, "Usage: misr_latlon_timerange_query, lat, lon, stime, etime"
    return
end

res = 275 ; MISR resolution in meters per pixel

status = mtk_latlon_to_pathlist( lat, lon, npath, pathlist )
if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

for i = 0, npath-1 do begin

    path = pathlist[i]

    status = mtk_latlon_to_somxy( path, lat, lon, somx, somy )
    if (status ne 0) then begin & print, mtk_error_message(status) & return & endif


    status = mtk_somxy_to_bls( path, res, somx, somy, block, line, sample )
    if (status ne 0) then begin & print, mtk_error_message(status) & return & endif


    status = mtk_path_timerange_to_orbitlist( path, stime, etime, norbit, orbitlist )
    if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

    print, format='(%"Path:%d   Lat/Lon <deg>: (%f,%f)   Som X/Y <meters>: (%f, %f)   Block/Line/Sample: (%d, %f, %f)")', $
      path, lat, lon, somx, somy, block, line, sample

    for j = 0, norbit-1 do begin

        orbit = orbitlist[j]

        status = mtk_orbit_to_timerange( orbit, orbit_stime, orbit_etime )
        if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

        status = mtk_datetime_to_julian( orbit_stime, orbit_stime_julian )
        if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

        status = mtk_datetime_to_julian( orbit_etime, orbit_etime_julian )
        if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

        total_orbit_meters_in_somx =  40263575.0 ; Approximate number of meters som x in a orbit
        jtime_per_meter_somx = (orbit_etime_julian - orbit_stime_julian) / total_orbit_meters_in_somx

        orbit_overpass_time_julian = orbit_stime_julian + jtime_per_meter_somx * somx

        status = mtk_julian_to_datetime( orbit_overpass_time_julian, orbit_overpass_time )
        if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

        print, format='(%"   Orbit: %6d    Overpass time: %s    Orbit start/end time (Asc. node): %s - %s")', $
          orbit, orbit_overpass_time, orbit_stime, orbit_etime

    endfor
    print
endfor

end
