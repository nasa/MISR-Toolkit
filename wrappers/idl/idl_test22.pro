pro idl_test22

indatetime = '2001-01-30T19:46:25Z'

status = mtk_datetime_to_julian(indatetime, outjulian)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_julian_to_datetime(outjulian, outdatetime)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, ' Input datetime: ', indatetime
print, '  Output julian: ', outjulian
print, 'Output datetime: ', outdatetime

print

injulian = 2453728.27313d

status = mtk_julian_to_datetime(injulian, datetime)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
status = mtk_datetime_to_julian(datetime, outjulian)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print, '   Input julian: ', injulian
print, 'Output datetime: ', datetime
print, '  Output julian: ', outjulian

end
