pro misr_predict, filename

stemplate = {VERSION:1.0, $
             DATASTART:0L, $
             DELIMITER:44B, $
             MISSINGVALUE:!VALUES.F_NAN, $
             COMMENTSYMBOL: '#', $
             FIELDCOUNT:5L, $
             FIELDTYPES:[4,4,4,7,4], $
             FIELDNAMES:['lat', 'lon', 'geo_extent', 'datetime', 'time_extent'], $
             FIELDLOCATIONS:[0,7,19,30,60], $
             FIELDGROUPS:[0,1,2,3,4]}

data = read_ascii(filename, template=stemplate)

for i = 0,(size(data))[2]-1 do begin
    print, 'Point: ',strtrim(i+1,2)
    misr_region_timerange_query, data.lat[i], data.lon[i], $
      data.geo_extent[i],data.datetime[i],data.time_extent[i]
    print
    print
endfor

end
