pro filelist_by_timerange, starttime, endtime

camera = ['DA', 'CA', 'BA', 'AA', 'AN', 'AF', 'BF', 'CF', 'DF']

status = mtk_timerange_to_orbitlist(starttime,endtime,norbit,orbitlist)

print, '      Start time = ', starttime
print, '        End time = ', endtime
print, 'Number of orbits = ', norbit
print, '      Orbit list = ', orbitlist

for i = 0, norbit-1 do begin
    print,i
    orbit = orbitlist[i]
    status = mtk_orbit_to_path(orbit, path)
    print,'orbit = ', orbitlist[i], ' path = ', path
    for cam = 0, 8 do begin
        status = mtk_make_filename('','GRP_ELLIPSOID_GM',camera[cam],$
                                   path,orbit,'F03_0023',filename)
        print,filename
    endfor
    status = mtk_make_filename('','GP_GMP','',path,orbit,$
                               'F03_0013',filename)
    print,filename
    status = mtk_make_filename('','TC_STEREO','',path,orbit,$
                               'F07_0012',filename)
    print,filename
endfor

end
