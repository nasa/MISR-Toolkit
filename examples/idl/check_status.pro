pro check_status, mesg, status
    if (status ne 0) then begin 
        print, mesg, mtk_error_message(status)
        stop
    endif
end
