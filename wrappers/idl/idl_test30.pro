pro idl_test30

;;
;; Set up input parameters
;; 
path = 168
res = 275
expected_proj_code = 22
status = MTK_PATH_TO_PROJPARAM(path, res, proj_params)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end
if (proj_params.projcode ne expected_proj_code) then begin
    print, "Test Error: Projection Code Invalid. Expected: ", expected_proj_code, ". Got: ", proj_params.projcode
    stop
end

end
