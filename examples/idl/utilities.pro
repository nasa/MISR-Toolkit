;===============================================================================
; vm_used
;===============================================================================
FUNCTION vm_used
        CATCH, error_status
        IF error_status NE 0 THEN BEGIN
                CATCH, /CANCEL
                RETURN,1
        ENDIF
        success = EXECUTE('a=1')
        RETURN, 0

END

; vm_used

;===============================================================================
; convert_pseudocolor2truecolor
;===============================================================================
FUNCTION convert_pseudocolor2truecolor, r, g, b
        return, ishft(long(b),16)+ishft(long(g),8)+ishft(long(r),0)
END
; convert_pseudocolor2truecolor

;===============================================================================
; is_valid_number
;===============================================================================
FUNCTION is_valid_number, num_str
   valid_flag                 = 1

        routine_name    = '========== is_valid_number =========='
        CATCH, error_status
        IF error_status NE 0 THEN BEGIN
                valid_flag      = 0
                RETURN, valid_flag
        ENDIF

   valid_flag                 = 1
   valid_strarr               = STRTRIM([ SINDGEN(10), '-', '.' ],2)
   num_str                    = STRTRIM(num_str,2)
   n_decimal_pts              = 0
   len                        = STRLEN(num_str)
   i                          = 0
   check_exp                  = 0
   
   exp_sep                    = STR_SEP( STRUPCASE(num_str), 'E' )
   IF N_ELEMENTS(exp_sep) GT 2 THEN valid_flag = 0
   
   IF N_ELEMENTS(exp_sep) EQ 2 THEN BEGIN
        check_exp       = 1
        num_str         = exp_sep[0]
        num_str2        = exp_sep[1]
   ENDIF
   
   WHILE valid_flag AND i LT len DO BEGIN
      current_char = STRMID(num_str,i,1)
      idx          = WHERE(current_char EQ valid_strarr, cnt)
      CASE 1 OF
         cnt LE 0: valid_flag = 0
         ELSE: BEGIN
         
            IF current_char EQ '.' THEN n_decimal_pts = n_decimal_pts + 1
            
            IF n_decimal_pts GT 1 THEN valid_flag = 0
            
            IF current_char EQ '-' AND i GT 0 THEN valid_flag = 0
            
            END
      ENDCASE
      i = i + 1
   ENDWHILE
   
   IF valid_flag AND check_exp THEN BEGIN
        pos_neg         = STRMID(num_str2, 0, 1)
        tmp_num_str     = STRMID(num_str2, 1)
        IF (pos_neg EQ '+' OR pos_neg EQ '-') AND tmp_num_str NE '' THEN BEGIN
                valid_flag      = is_valid_number( tmp_num_str )
        ENDIF ELSE BEGIN
                valid_flag      = 0
        ENDELSE
   ENDIF
   
   RETURN, valid_flag
END

;===============================================================================
; get_file
;===============================================================================
FUNCTION get_file, use_gui
	CD, CURRENT = cdir
	cdir	= STRTRIM(cdir,2)
	IF STRMID(cdir,STRLEN(cdir)-1,1) NE PATH_SEP() THEN			$
		cdir = cdir + PATH_SEP()
	pos	= STRPOS(cdir, PATH_SEP(), /REVERSE_SEARCH)
	pos	= STRPOS(cdir, PATH_SEP(), pos-1, /REVERSE_SEARCH)
	pos	= STRPOS(cdir, PATH_SEP(), pos-1, /REVERSE_SEARCH)
	pos	= STRPOS(cdir, PATH_SEP(), pos-1, /REVERSE_SEARCH)
	cdir	= STRMID(cdir,0,pos)+PATH_SEP()+'Mtk_testdata'+PATH_SEP()+'in'+	$
		PATH_SEP()
	IF use_gui THEN BEGIN
		RETURN, DIALOG_PICKFILE(PATH = cdir, TITLE = 'Select a file')
	ENDIF ELSE BEGIN
		files	= FILE_SEARCH(cdir+'*', COUNT = cnt)
		qual_files	= files
		FOR i = 0, cnt - 1 DO BEGIN
			sep		= STR_SEP(files[i],PATH_SEP())
			files[i]	= sep[N_ELEMENTS(sep)-1]
		ENDFOR
		IF cnt LE 0 THEN BEGIN
			PRINT, 'No files found in '+cdir
			RETURN, ''
		ENDIF
		list	= STRTRIM(LINDGEN(cnt)+1L,2)+' - '+files
		FOR i = 0L, cnt-1 DO PRINT, list[i]
		str	= 'bad'
		PRINT, ''
		WHILE NOT is_valid_number(str) DO				$
			READ, 'Enter the number of the file of interest --> ', str
		num	= FIX(str)
		IF num GT cnt THEN BEGIN
			PRINT, 'Number entered is too high, using '+STRTRIM(cnt,2)
			num	= cnt
		ENDIF
		IF num LT 1 THEN BEGIN
			PRINT, 'Number entered is too low, using 1'
			num	= 1
		ENDIF
		
		RETURN, qual_files[num-1]
		
	ENDELSE
END
; get_file

;1234567890123456789012345678901234567890123456789012345678901234567890123456789
;===============================================================================
; display_msg
;===============================================================================
PRO display_msg, msg, use_gui

	IF use_gui THEN BEGIN
		res	= DIALOG_MESSAGE(msg, /INFORMATION)
	ENDIF ELSE BEGIN
		top	= '**************************************************'
		top	= top + '******************************'
		bottom	= top
		n	= N_ELEMENTS(msg)
		PRINT, top
		FOR i = 0L, n-1 DO BEGIN
			line	= '* '+STRTRIM(msg[i],2)
			len	= STRLEN(line)
			pad	= 80-len
			FOR j=0,pad-2 DO line = line + ' '
			line	= line + '*'
			PRINT, line
		ENDFOR
		PRINT, bottom
		str	= ''
		READ, 'Press <RETURN> key to continue ', str
		PRINT, ''
	ENDELSE
END
; display_msg


;1234567890123456789012345678901234567890123456789012345678901234567890123456789
;===============================================================================
; convert_systime2mtktime
;===============================================================================
FUNCTION convert_systime2mtktime
  months = [                                                                   $
            'JAN',                                                             $
            'FEB',                                                             $
            'MAR',                                                             $
            'APR',                                                             $
            'MAY',                                                             $
            'JUN',                                                             $
            'JUL',                                                             $
            'AUG',                                                             $
            'SEP',                                                             $
            'OCT',                                                             $
            'NOV',                                                             $
            'DEC']

  s      = SYSTIME()
  sep    = STR_SEP(STRTRIM(STRCOMPRESS(s),2),' ')
  yr     = sep[N_ELEMENTS(sep)-1]
  month  = STRTRIM((WHERE(months EQ STRUPCASE(sep[1])))[0]+1,2)
  IF STRLEN(month) LT 2 THEN month = '0'+month
  day    = sep[2]
  IF STRLEN(day) LT 2 THEN day = '0'+day
  hour   = (STR_SEP(sep[3],':'))[0]
  IF STRLEN(hour) LT 2 THEN hour = '0'+hour
  minute = (STR_SEP(sep[3],':'))[1]
  IF STRLEN(minute) LT 2 THEN minute = '0'+minute
  sec    = (STR_SEP(sep[3],':'))[2]
  IF STRLEN(sec) LT 2 THEN sec = '0'+sec
  RETURN, yr+month+day+hour+minute+sec
END
; convert_systeime2mtktime
