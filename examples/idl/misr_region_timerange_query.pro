pro misr_region_timerange_query, ctr_lat_dd, ctr_lon_dd, geo_extent_km, utc_datetime, time_extent_hours

; --------------------------------------------------------------------------
; Argument check
; --------------------------------------------------------------------------

if (n_params() ne 5) then begin
    print, "Usage: misr_region_timerange_query, ctr_lat_dd, ctr_lon_dd, geo_extent_km, utc_datetime, time_extent_hours"
    return
end

; --------------------------------------------------------------------------
; Find paths that cross specified region defined by lat/lon and geo_extent
; --------------------------------------------------------------------------

res = 275 ; MISR resolution in meters per pixel

status = mtk_setregion_by_latlon_extent( ctr_lat_dd, ctr_lon_dd, geo_extent_km, geo_extent_km, "km", region )
if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

status = mtk_region_to_pathlist( region, npath, pathlist )
if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

; --------------------------------------------------------------------------
; Determine date/time range from specified time and time_extent
; --------------------------------------------------------------------------

status = mtk_datetime_to_julian( utc_datetime, julian_date )
if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

stime_julian_date = julian_date - ((time_extent_hours / 2.0) / 24.0)
etime_julian_date = julian_date + ((time_extent_hours / 2.0) / 24.0)

status = mtk_julian_to_datetime( stime_julian_date, stime )
if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

status = mtk_julian_to_datetime( etime_julian_date, etime )
if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

; --------------------------------------------------------------------------
; Print search area in terms of both geo-location and date/time
; --------------------------------------------------------------------------

print, "        Latitude: ", strtrim(ctr_lat_dd,2)
print, "       Longitude: ", strtrim(ctr_lon_dd,2)
print, "      Geo region: +/-", strtrim(geo_extent_km/2.0,2)," km" 
print, "Center date/time: ", utc_datetime
print, "     Time window: +/-", strtrim(time_extent_hours/2.0,2)," hours"
print, " Start data/time: ", stime
print, "   End data/time: ", etime
print

; --------------------------------------------------------------------------
; Print paths that cross geographic region
; --------------------------------------------------------------------------

print, "Paths that cross the geographic region: "
print, "    ", strtrim(pathlist,2)
print

; --------------------------------------------------------------------------
; Print orbits/paths that are within the date/time range
; --------------------------------------------------------------------------

status = mtk_timerange_to_orbitlist( stime, etime, norbit, orbitlist )
if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

orbitlist_str = ''
for i = 0, norbit-1 do begin
    orbit_number = orbitlist[i]
    status = mtk_orbit_to_path( orbit_number, path_number )
    if (status ne 0) then begin & print, mtk_error_message(status) & return & endif
    orbitlist_str = orbitlist_str + string(format='(%"%d(%d)  ")',strtrim(orbit_number,2),path_number)
endfor

print, "Orbits(Path) that are within the date/time range: "
print, "    ", orbitlist_str
print

; --------------------------------------------------------------------------
; Loop over all possible paths
; --------------------------------------------------------------------------

for i = 0, npath-1 do begin

    path = pathlist[i]

; --------------------------------------------------------------------------
; Find orbits for each path within the date/time range
; --------------------------------------------------------------------------

    status = mtk_path_timerange_to_orbitlist( path, stime, etime, norbit, orbitlist )
    if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

    if (norbit eq 0) then print, format='(%"Path:%3d   No overpass orbits")',path

    for j = 0, norbit-1 do begin

        orbit = orbitlist[j]

; --------------------------------------------------------------------------
; Snap given region to a MISR grid for the path to determine lat/lon
; of bounding box
; --------------------------------------------------------------------------

        status = mtk_snap_to_grid( path, res, region, mapinfo )
        if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

        status = mtk_latlon_to_somxy( path, mapinfo.geo_ctr_lat, mapinfo.geo_ctr_lat, somx, somy )
        if (status ne 0) then begin & print, mtk_error_message(status) & return & endif

; --------------------------------------------------------------------------
; Print bounding box coordinates
; --------------------------------------------------------------------------

        print, format='(%"Path:%3d   Number overpass orbits: %d")',path,norbit

        print, "   Bounding box of geographic region:"
        print, format='(%"      ULC Lat/Lon <deg>: (%f,%f)\n   Center Lat/Lon <deg>: (%f,%f)\n      LRC Lat/Lon <deg>: (%f,%f)")', $
          mapinfo.geo_ulc_lat, mapinfo.geo_ulc_lon, mapinfo.geo_ctr_lat, mapinfo.geo_ctr_lon, $
          mapinfo.geo_lrc_lat, mapinfo.geo_lrc_lon
        print

; --------------------------------------------------------------------------
; Find block range for given path and region
; --------------------------------------------------------------------------

        status = mtk_region_path_to_blockrange( region, path, start_block, end_block )
        if (status ne 0) then begin & print, mtk_error_message(status) & return & endif
        
; --------------------------------------------------------------------------
; Estimate the overpass time
; --------------------------------------------------------------------------

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

; --------------------------------------------------------------------------
; Print result
; --------------------------------------------------------------------------

        print, format='(%"    Path: %4d\n    Orbit: %6d\n    Blocks: %d-%d\n    Approx. Overpass date/time: %s\n    Orbit start/end date/time (Asc. node): %s - %s")', $
          path, orbit, start_block, end_block, orbit_overpass_time, orbit_stime, orbit_etime

    endfor
endfor

end
