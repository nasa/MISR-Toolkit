@utilities.pro

;===============================================================================
; get_help_base
;===============================================================================
FUNCTION get_help_base, parent_base
  b        = WIDGET_BASE(                                                      $
               GROUP_LEADER = parent_base,                                     $
               /COLUMN,                                                        $
               TITLE = 'Help',                                                 $
               /BASE_ALIGN_LEFT )

  txt      = [                                                                 $
               'This example allows a user to specify a latitude and ',        $
               'longitude as a center point, along with a region specified',   $
               'in degrees, meters, or pixels, and a starting and ending',     $
               'time.  From this information, various MTK routines are',       $
               'utilized to calculate the paths and corresponding orbits',     $
               'which either intersect the center latitude/longitude point',   $
               'or the region defined by the center point and the extent.',    $
               '',                                                             $
               'Manipulate the center point and extent controls on the main',  $
               'interface, then use the time droplists to select a starting',  $
               'and ending time to consider.  Press "Get Information" to',     $
               'display an interface showing all relevant paths and orbits.',  $
               '',                                                             $
               'MTK Routines Used:',                                           $
               '   MTK_PATH_TIMERANGE_TO_ORBITLIST( ',                         $
               '                          path,',                              $
               '                          start_time,',                        $
               '                          end_time,',                          $
               '                          n_orbits,',                          $
               '                          orbits)',                            $
               '   MTK_LATLON_TO_PATHLIST( ',                                  $
               '                          lat,',                               $
               '                          lon,',                               $
               '                          nlatlon_paths,',                     $
               '                          latlon_path )',                      $
               '   MTK_SETREGION_BY_LATLON_EXTENT( ',                          $
               '                          lat,',                               $
               '                          lon,',                               $
               '                          yextent,',                           $
               '                          xextent,',                           $
               '                          extent_units,',                      $
               '                          regioninfo)',                        $
               '   MTK_REGION_TO_PATHLIST( ',                                  $
               '                          regioninfo,',                        $
               '                          nregion_paths,',                     $
               '                          region_pathlist )' ]



  FOR j = 0, N_ELEMENTS(txt)-1 DO                                              $
    lbl = WIDGET_LABEL(b,VALUE = txt[j],/DYNAMIC_RESIZE)

  WIDGET_CONTROL, b, /REALIZE
  RETURN, b
END
; get_help_base

;===============================================================================
; example2_getinfo_eh
;===============================================================================
PRO example2_getinfo_eh, event
  WIDGET_CONTROL, event.top, GET_UVALUE = ptr
  WIDGET_CONTROL, event.id, GET_UVALUE = widget_id
  widget_id   = STRTRIM(STRUPCASE(widget_id),2)

  CASE widget_id OF
    'WD1': BEGIN
      (*ptr).wd1_idx   = event.index
      CASE event.index OF
        0: BEGIN
          WIDGET_CONTROL,                                                      $
            (*ptr).list1,                                                      $
            SET_VALUE = STRTRIM((*ptr).latlon_pathlist,2), SET_LIST_SELECT = 0
          END
        1: BEGIN
          WIDGET_CONTROL,                                                      $
            (*ptr).list1,                                                      $
            SET_VALUE = STRTRIM((*ptr).region_pathlist,2), SET_LIST_SELECT = 0
          END
        ELSE:
      ENDCASE
      END
    'LIST1': BEGIN
      IF (*ptr).wd1_idx EQ 0 THEN                                              $
        path2use = ((*ptr).latlon_pathlist)[event.index]                       $
      ELSE                                                                     $
        path2use = ((*ptr).region_pathlist)[event.index]
        routine   = 'MTK_PATH_TIMERANGE_TO_ORBITLIST'
      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_PATH_TIMERANGE_TO_ORBITLIST( path,           $
      ;                                            start_time,     $
      ;                                            end_time,       $
      ;                                            n_orbits,       $
      ;                                            orbits)
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
      status    = MTK_PATH_TIMERANGE_TO_ORBITLIST(                             $
                    LONG(path2use),                                            $
                    (*ptr).misr_start_time,                                    $
                    (*ptr).misr_end_time,                                      $
                    n_orbits,                                                  $
                    orbit_list )
      IF status THEN BEGIN
        msg     = ['Problem with routine '+routine+'... exiting...']
        res     = DIALOG_MESSAGE(msg, /ERROR)
        RETURN
      ENDIF

      WIDGET_CONTROL, (*ptr).list2, SET_VALUE = STRTRIM(orbit_list,2)
      WIDGET_CONTROL,                                                          $
        (*ptr).list2lbl,                                                       $
        SET_VALUE = 'Orbit List For Path '+STRTRIM(path2use,2)


      routine   = 'MTK_REGION_PATH_TO_BLOCKRANGE'
      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_REGION_PATH_TO_BLOCKRANGE( region,           $
      ;                                          path,             $
      ;                                          start_block,      $
      ;                                          end_block)
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
      status    = MTK_REGION_PATH_TO_BLOCKRANGE(                               $
                    (*ptr).regioninfo,                                         $
                    LONG(path2use),                                            $
                    start_block,                                               $
                    end_block )
      IF status THEN BEGIN
        msg     = ['Problem with routine '+routine+'... exiting...']
        res     = DIALOG_MESSAGE(msg, /ERROR)
        RETURN
      ENDIF
      WIDGET_CONTROL, (*ptr).blockrangelbl, SET_VALUE = 'Block Range: '+       $
                                                        STRTRIM(start_block,2)+$
                                                        ' through '+           $
                                                        STRTRIM(end_block,2)
      
      END
    'LIST2': BEGIN
      END
    'EXIT': WIDGET_CONTROL, event.top, /DESTROY
    ELSE:
  ENDCASE
  
END
; example2_getinfo_eh

;1234567890123456789012345678901234567890123456789012345678901234567890123456789
;===============================================================================
; example2_getinfo
;===============================================================================
PRO example2_getinfo, lat, lon, yextent, xextent, stime, etime, tlb, UNITS=units
  routine   = 'MTK_LATLON_TO_PATHLIST'
  ;=============================================================================
  ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
  ;
  ; MTK CALL: MTK_LATLON_TO_PATHLIST( lat,lon,                                $
  ;                                   nlatlon_paths,latlon_pathlist)
  ;
  ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
  ;=============================================================================
  status    = MTK_LATLON_TO_PATHLIST(                                          $
                lat,                                                           $
                lon,                                                           $
                nlatlon_paths,                                                 $
                latlon_pathlist )
  IF status THEN BEGIN
    msg     = ['Problem with routine MTK_LATLON_TO_PATHLIST... exiting...']
    res     = DIALOG_MESSAGE(msg, /ERROR)
    RETURN
  ENDIF

  extent_type = 'LATLON'
  IF KEYWORD_SET(units) THEN extent_type = STRTRIM(STRUPCASE(units),2)

  CASE extent_type OF
    'LATLON': BEGIN
      routine   = 'MTK_SETREGION_BY_LATLON_EXTENT'
      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_SETREGION_BY_LATLON_EXTENT(                              $
      ;                                                  lat,                  $
      ;                                                  lon,                  $
      ;                                                  yextent,              $
      ;                                                  xextent,              $
      ;                                                  extent_units,         $
      ;                                                  regioninfo)
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
      status    = MTK_SETREGION_BY_LATLON_EXTENT(                              $
                    lat,                                                       $
                    lon,                                                       $
                    yextent,                                                   $
                    xextent,                                                   $
                    'degrees',                                                 $
                    regioninfo)
      END
    'METERS': BEGIN
      routine   = 'MTK_SETREGION_BY_LATLON_EXTENT'
      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_SETREGION_BY_LATLON_EXTENT(                              $
      ;                                                  lat,                  $
      ;                                                  lon,                  $
      ;                                                  yextent,              $
      ;                                                  xextent,              $
      ;                                                  extent_units,         $
      ;                                                  regioninfo)
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
      status    = MTK_SETREGION_BY_LATLON_EXTENT(                              $
                    lat,                                                       $
                    lon,                                                       $
                    yextent,                                                   $
                    xextent,                                                   $
                    'meters',                                                  $
                    regioninfo)
      END
    'PIXELS': BEGIN
      routine   = 'MTK_SETREGION_BY_LATLON_EXTENT'
      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_SETREGION_BY_LATLON_EXTENT(                              $
      ;                                                  lat,                  $
      ;                                                  lon,                  $
      ;                                                  yextent,              $
      ;                                                  xextent,              $
      ;                                                  extent_units,         $
      ;                                                  regioninfo)
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
      status    = MTK_SETREGION_BY_LATLON_EXTENT(                              $
                    lat,                                                       $
                    lon,                                                       $
                    yextent,                                                   $
                    xextent,                                                   $
                    '275m',                                                    $
                    regioninfo)
      END
    ELSE:
  ENDCASE

  IF status THEN BEGIN
    msg     = ['Problem with routine '+routine+'... exiting...']
    res     = DIALOG_MESSAGE(msg, /ERROR)
    RETURN
  ENDIF

  routine   = 'MTK_REGION_TO_PATHLIST'
  ;=============================================================================
  ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
  ;
  ; MTK CALL: MTK_REGION_TO_PATHLIST( regioninfo,                             $
  ;                                   nregion_paths,                          $
  ;                                   region_pathlist )
  ;
  ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
  ;=============================================================================
  status    = MTK_REGION_TO_PATHLIST(                                          $
                regioninfo,                                                    $
                nregion_paths,                                                 $
                region_pathlist )

  IF status THEN BEGIN
    msg     = ['Problem with routine '+routine+'... exiting...']
    res     = DIALOG_MESSAGE(msg, /ERROR)
    RETURN
  ENDIF

  info_tlb  = WIDGET_BASE(                                                     $
                GROUP_LEADER = tlb,                                            $
                /COLUMN,                                                       $
                /MODAL,                                                        $
                /BASE_ALIGN_LEFT )
  lbl1txt   = 'Latitude: '+STRTRIM(lat,2)+', Longitude: '+STRTRIM(lon,2)
  lbl1      = WIDGET_LABEL(                                                    $
                info_tlb,                                                      $
                VALUE = lbl1txt,                                               $
                /DYNAMIC_RESIZE )
  lbl2txt   = 'Start Time (YYYY-MM-DDThh:mm:ssZ):'+stime
  lbl2      = WIDGET_LABEL(                                                    $
                info_tlb,                                                      $
                VALUE = lbl2txt,                                               $
                /DYNAMIC_RESIZE )
  lbl3txt   = 'Start Time (YYYY-MM-DDThh:mm:ssZ):'+etime
  lbl3      = WIDGET_LABEL(                                                    $
                info_tlb,                                                      $
                VALUE = lbl3txt,                                               $
                /DYNAMIC_RESIZE )
  wd1txt    = ['Show paths intersecting lat/lon',                              $
               'Show paths intersecting region']
  wd1       = WIDGET_DROPLIST(                                                 $
                info_tlb,                                                      $
                TITLE = '',                                                    $
                VALUE = wd1txt,                                                $
                UVALUE = 'wd1',                                                $
                /DYNAMIC_RESIZE )
  wd1_idx   = 0
  sub1      = WIDGET_BASE(                                                     $
                info_tlb,                                                      $
                /COLUMN,                                                       $
                /FRAME )
  list1lbl  = WIDGET_LABEL(                                                    $
                sub1,                                                          $
                /DYNAMIC_RESIZE,                                               $
                VALUE = 'Path List' )
  list1     = WIDGET_LIST(                                                     $
                sub1,                                                          $
                VALUE = STRTRIM(latlon_pathlist,2),                            $
                /FRAME,                                                        $
                UVALUE = 'list1',                                              $
                SCR_XSIZE = 200,                                               $
                SCR_YSIZE = 200 )

  misr_start_time                                                              $
            = stime
  misr_end_time                                                                $
            = etime
  routine   = 'MTK_PATH_TIMERANGE_TO_ORBITLIST'
  ;=============================================================================
  ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
  ;
  ; MTK CALL: MTK_PATH_TIMERANGE_TO_ORBITLIST( path,           $
  ;                                            start_time,     $
  ;                                            end_time,       $
  ;                                            n_orbits,       $
  ;                                            orbits)
  ;
  ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
  ;=============================================================================
  status    = MTK_PATH_TIMERANGE_TO_ORBITLIST(                                 $
                latlon_pathlist[0],                                            $
                misr_start_time,                                               $
                misr_end_time,                                                 $
                n_orbits,                                                      $
                orbit_list )
  IF status THEN BEGIN
    msg     = ['Problem with routine '+routine+'... exiting...']
    res     = DIALOG_MESSAGE(msg, /ERROR)
    RETURN
  ENDIF

  sub2      = WIDGET_BASE(                                                     $
                info_tlb,                                                      $
                /COLUMN,                                                       $
                /FRAME )
  list2lbl  = WIDGET_LABEL(                                                    $
                sub2,                                                          $
                /DYNAMIC_RESIZE,                                               $
                VALUE = 'Orbit List For Path '+STRTRIM(latlon_pathlist[0],2) )
  list2     = WIDGET_LIST(                                                     $
                sub2,                                                          $
                VALUE = STRTRIM(orbit_list,2),                                 $
                /FRAME,                                                        $
                UVALUE = 'list2',                                              $
                SCR_XSIZE = 200,                                               $
                SCR_YSIZE = 200 )
  sub3      = WIDGET_BASE(                                                     $
                info_tlb,                                                      $
                /COLUMN,                                                       $
                /FRAME )
  routine   = 'MTK_REGION_PATH_TO_BLOCKRANGE'
  ;=============================================================================
  ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
  ;
  ; MTK CALL: MTK_REGION_PATH_TO_BLOCKRANGE( region,           $
  ;                                          path,             $
  ;                                          start_block,      $
  ;                                          end_block)
  ;
  ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
  ;=============================================================================
  status    = MTK_REGION_PATH_TO_BLOCKRANGE(                                   $
                regioninfo,                                                    $
                latlon_pathlist[0],                                            $
                start_block,                                                   $
                end_block )
  IF status THEN BEGIN
    msg     = ['Problem with routine '+routine+'... exiting...']
    res     = DIALOG_MESSAGE(msg, /ERROR)
    RETURN
  ENDIF
  lbl3      = WIDGET_LABEL(                                                    $
                sub3,                                                          $
                VALUE = 'Block Range: '+                                       $
                        STRTRIM(start_block,2)+                                $
                        ' through '+                                           $
                        STRTRIM(end_block,2))
  exitb     = WIDGET_BUTTON(                                                   $
                info_tlb,                                                      $
                VALUE = 'E X I T',                                             $
                UVALUE = 'exit' )
  WIDGET_CONTROL, info_tlb, /REALIZE
  WIDGET_CONTROL, list1, SET_LIST_SELECT = 0
  ptr       = PTR_NEW( {                                                       $
                        wd1_idx          : wd1_idx,                            $
                        latlon_pathlist  : latlon_pathlist,                    $
                        region_pathlist  : region_pathlist,                    $
                        wd1              : wd1,                                $
                        list1            : list1,                              $
                        list2            : list2,                              $
                        list2lbl         : list2lbl,                           $
                        blockrangelbl    : lbl3,                               $
                        regioninfo       : regioninfo,                         $
                        misr_start_time  : misr_start_time,                    $
                        misr_end_time    : misr_end_time } )
  WIDGET_CONTROL, info_tlb, SET_UVALUE = ptr
  XMANAGER, 'example2_getinfo', info_tlb, EVENT_HANDLER = 'example2_getinfo_eh'
  PTR_FREE, ptr
END
; example2_getinfo

;1234567890123456789012345678901234567890123456789012345678901234567890123456789
;===============================================================================
; example2_eh
;===============================================================================
PRO example2_eh, event
  WIDGET_CONTROL, event.top, GET_UVALUE = ptr
  WIDGET_CONTROL, event.id, GET_UVALUE = widget_name

  CASE STRTRIM(STRUPCASE(widget_name),2) OF
    'LAT_CWF':
    'LON_CWF':
    'LATEXTENT_CWF':
    'LONEXTENT_CWF':
    'GET_INFO': BEGIN
      start_yr    = (LINDGEN(10)+2000)[(*ptr).time1_idx[0]]
      start_month = (LINDGEN(12)+1)[(*ptr).time1_idx[1]]
      start_day   = (LINDGEN(31)+1)[(*ptr).time1_idx[2]]
      start_hr    = (LINDGEN(24))[(*ptr).time1_idx[3]]
      start_min   = (LINDGEN(60))[(*ptr).time1_idx[4]]
      start_sec   = (LINDGEN(60))[(*ptr).time1_idx[5]]
      end_yr      = (LINDGEN(10)+2000)[(*ptr).time2_idx[0]]
      end_month   = (LINDGEN(12)+1)[(*ptr).time2_idx[1]]
      end_day     = (LINDGEN(31)+1)[(*ptr).time2_idx[2]]
      end_hr      = (LINDGEN(24))[(*ptr).time2_idx[3]]
      end_min     = (LINDGEN(60))[(*ptr).time2_idx[4]]
      end_sec     = (LINDGEN(60))[(*ptr).time2_idx[5]]


      CASE start_month OF
        2: BEGIN
          IF (start_yr-2000) MOD 4 EQ 0 AND start_day GT 29 OR                $
             (start_yr-2000) MOD 4 NE 0 AND start_day GT 28 THEN BEGIN
            msg  = ['Bad value for starting day value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
            RETURN
          ENDIF
          END
        4: BEGIN
          IF start_day GT 30 THEN BEGIN
            msg  = ['Bad value for starting day value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
          ENDIF
          END
        6: BEGIN
          IF start_day GT 30 THEN BEGIN
            msg  = ['Bad value for starting day value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
          ENDIF
          END
        9: BEGIN
          IF start_day GT 30 THEN BEGIN
            msg  = ['Bad value for starting day value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
          ENDIF
          END
        11: BEGIN
          IF start_day GT 30 THEN BEGIN
            msg  = ['Bad value for starting day value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
          ENDIF
          END
        ELSE:
      ENDCASE
      CASE end_month OF
        2: BEGIN
          IF (end_yr-2000) MOD 4 EQ 0 AND end_day GT 29 OR                     $
             (end_yr-2000) MOD 4 NE 0 AND end_day GT 28 THEN BEGIN
            msg  = ['Bad value for ending day value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
            RETURN
          ENDIF
          END
        4: BEGIN
          IF end_day GT 30 THEN BEGIN
            msg  = ['Bad value for ending day value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
          ENDIF
          END
        6: BEGIN
          IF end_day GT 30 THEN BEGIN
            msg  = ['Bad value for ending day value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
          ENDIF
          END
        9: BEGIN
          IF end_day GT 30 THEN BEGIN
            msg  = ['Bad value for ending day value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
          ENDIF
          END
        11: BEGIN
          IF end_day GT 30 THEN BEGIN
            msg  = ['Bad value for ending day value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
          ENDIF
          END
        ELSE:
      ENDCASE

      start_jday  = JULDAY(                                                    $
                      start_month,                                             $
                      start_day,                                               $
                      start_yr,                                                $
                      start_hr,                                                $
                      start_min,                                               $
                      start_sec )
      end_jday    = JULDAY(                                                    $
                      end_month,                                               $
                      end_day,                                                 $
                      end_yr,                                                  $
                      end_hr,                                                  $
                      end_min,                                                 $
                      end_sec )
      IF start_jday GT end_jday THEN BEGIN
        msg  = ['Starting time must be less than or equal to ending time!']
        res  = DIALOG_MESSAGE(msg, /ERROR)
        RETURN
      ENDIF

      ssyr       = STRTRIM(start_yr,2)
      ssmonth    = '0'+STRTRIM(start_month,2)
      ssmonth    = STRMID(ssmonth,STRLEN(ssmonth)-2,2)
      ssday      = '0'+STRTRIM(start_day,2)
      ssday      = STRMID(ssday,STRLEN(ssday)-2,2)
      sshr       = '0'+STRTRIM(start_hr,2)
      sshr       = STRMID(sshr,STRLEN(sshr)-2,2)
      ssmin      = '0'+STRTRIM(start_min,2)
      ssmin      = STRMID(ssmin,STRLEN(ssmin)-2,2)
      sssec      = '0'+STRTRIM(start_sec,2)
      sssec      = STRMID(sssec,STRLEN(sssec)-2,2)

      esyr       = STRTRIM(end_yr,2)
      esmonth    = '0'+STRTRIM(end_month,2)
      esmonth    = STRMID(esmonth,STRLEN(esmonth)-2,2)
      esday      = '0'+STRTRIM(end_day,2)
      esday      = STRMID(esday,STRLEN(esday)-2,2)
      eshr       = '0'+STRTRIM(end_hr,2)
      eshr       = STRMID(eshr,STRLEN(eshr)-2,2)
      esmin      = '0'+STRTRIM(end_min,2)
      esmin      = STRMID(esmin,STRLEN(esmin)-2,2)
      essec      = '0'+STRTRIM(end_sec,2)
      essec      = STRMID(essec,STRLEN(essec)-2,2)

      start_time  = ssyr+'-'+ssmonth+'-'+ssday+'T'+sshr+':'+ssmin+':'+sssec+'Z'
      end_time    = esyr+'-'+esmonth+'-'+esday+'T'+eshr+':'+esmin+':'+essec+'Z'

      WIDGET_CONTROL, (*ptr).lat_cwf, GET_VALUE = lat
      lat  = lat[0]
      IF lat LT -90.0 OR lat GT 90.0 THEN BEGIN
        msg  = ['Invalid center latitude value!']
        res  = DIALOG_MESSAGE(msg, /ERROR)
        RETURN
      ENDIF
      WIDGET_CONTROL, (*ptr).lon_cwf, GET_VALUE = lon
      lon  = lon[0]
      IF lon LT -180.0 OR lon GT 180.0 THEN BEGIN
        msg  = ['Invalid center longitude value!']
        res  = DIALOG_MESSAGE(msg, /ERROR)
        RETURN
      ENDIF

      CASE (*ptr).ext_idx OF
        0: BEGIN
          WIDGET_CONTROL, (*ptr).latext_cwf, GET_VALUE = latext
          latext  = latext[0]
          IF latext LE 0.0 OR latext GT 180.0 THEN BEGIN
            msg  = ['Invalid latitudinal extent value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
            RETURN
          ENDIF
          WIDGET_CONTROL, (*ptr).lonext_cwf, GET_VALUE = lonext
          lonext  = lonext[0]
          IF lonext LE 0.0 OR lonext GT 360.0 THEN BEGIN
            msg  = ['Invalid longitudinal extent value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
            RETURN
          ENDIF
          example2_getinfo, lat, lon, latext, lonext, start_time, end_time,                  $
                            event.top,UNITS='LATLON'
          END
        1: BEGIN
          WIDGET_CONTROL, (*ptr).xmeter_cwf, GET_VALUE = xmeter
          xmeter  = xmeter[0]
          IF xmeter LE 0.0 THEN BEGIN
            msg  = ['Invalid horizontal extent value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
            RETURN
          ENDIF
          WIDGET_CONTROL, (*ptr).ymeter_cwf, GET_VALUE = ymeter
          ymeter  = ymeter[0]
          IF ymeter LE 0.0 THEN BEGIN
            msg  = ['Invalid vertical extent value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
            RETURN
          ENDIF
          example2_getinfo, lat, lon, ymeter, xmeter,  start_time, end_time,                 $
                            event.top,UNITS='METERS'
          END
        2: BEGIN
          WIDGET_CONTROL, (*ptr).xpixel_cwf, GET_VALUE = xpixel
          xpixel  = xpixel[0]
          IF xpixel LE 0.0 THEN BEGIN
            msg  = ['Invalid horizontal extent value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
            RETURN
          ENDIF
          WIDGET_CONTROL, (*ptr).ypixel_cwf, GET_VALUE = ypixel
          ypixel  = ypixel[0]
          IF ypixel LE 0.0 THEN BEGIN
            msg  = ['Invalid vertical extent value!']
            res  = DIALOG_MESSAGE(msg, /ERROR)
            RETURN
          ENDIF
          example2_getinfo, lat, lon, ypixel, xpixel,   start_time, end_time,                $
                            event.top, UNITS='PIXELS'
          END
        ELSE:
      ENDCASE
      
      END
    'EXT_OPTIONS_WD': BEGIN
      (*ptr).ext_idx = event.index
      FOR i = 0, N_ELEMENTS((*ptr).ext_bases)-1 DO                             $
        WIDGET_CONTROL, ((*ptr).ext_bases)[i], MAP = (i EQ (*ptr).ext_idx)
      END
    'YR_WD1': BEGIN
      (*ptr).time1_idx[0] = event.index
      END
    'MONTH_WD1': BEGIN
      (*ptr).time1_idx[1] = event.index
      END
    'DAY_WD1': BEGIN
      (*ptr).time1_idx[2] = event.index
      END
    'HOUR_WD1': BEGIN
      (*ptr).time1_idx[3] = event.index
      END
    'MINUTE_WD1': BEGIN
      (*ptr).time1_idx[4] = event.index
      END
    'SECOND_WD1': BEGIN
      (*ptr).time1_idx[5] = event.index
      END
    'YR_WD2': BEGIN
      (*ptr).time2_idx[0] = event.index
      END
    'MONTH_WD2': BEGIN
      (*ptr).time2_idx[1] = event.index
      END
    'DAY_WD2': BEGIN
      (*ptr).time2_idx[2] = event.index
      END
    'HOUR_WD2': BEGIN
      (*ptr).time2_idx[3] = event.index
      END
    'MINUTE_WD2': BEGIN
      (*ptr).time2_idx[4] = event.index
      END
    'SECOND_WD2': BEGIN
      (*ptr).time2_idx[5] = event.index
      END
    'QUIT': BEGIN
      WIDGET_CONTROL, event.top, /DESTROY
      END
    'HELP': BEGIN
      IF WIDGET_INFO((*ptr).help_base, /VALID_ID) THEN                          $
        WIDGET_CONTROL, (*ptr).help_base, /SHOW                                $
      ELSE                                                                     $
        (*ptr).help_base  = get_help_base(event.top)
        
      END
    ELSE:
  ENDCASE
END
; example2_eh

;1234567890123456789012345678901234567890123456789012345678901234567890123456789
;===============================================================================
; example2
;===============================================================================
PRO example2
  tlb       = WIDGET_BASE(                                                     $
                /COLUMN,                                                       $
                TITLE = 'example2',                                            $
                /BASE_ALIGN_LEFT,                                              $
                EVENT_PRO = 'example2_eh' )
  lat_cwf   = CW_FIELD(                                                        $
                tlb,                                                           $
                /ROW,                                                          $
                /FLOAT,                                                        $
                VALUE = 0.0,                                                   $
                TITLE = 'Center Latitude (-90.0 <= lat <= 90.0):       ',      $
                UVALUE = 'lat_cwf' )
  lon_cwf   = CW_FIELD(                                                        $
                tlb,                                                           $
                /ROW,                                                          $
                /FLOAT,                                                        $
                VALUE = 0.0,                                                   $
                TITLE = 'Center Longitude (-180.0 <= lon <= 180.0):    ',      $
                UVALUE = 'lon_cwf' )
  ext_options                                                                  $
            = [ 'Define extent in degrees',                                    $
                'Define extent in meters',                                     $
                'Define extent in pixels' ]
  ext_wd    = WIDGET_DROPLIST(                                                 $
                tlb,                                                           $
                /DYNAMIC_RESIZE,                                               $
                VALUE = ext_options,                                           $
                UVALUE = 'ext_options_wd' )
  ext_base  = WIDGET_BASE(                                                     $
                tlb,                                                           $
                /FRAME )
  latlonext_base                                                               $
            = WIDGET_BASE(                                                     $
                ext_base,                                                      $
                /COLUMN,                                                       $
                /BASE_ALIGN_LEFT )
  latext_cwf= CW_FIELD(                                                        $
                latlonext_base,                                                $
                /ROW,                                                          $
                /FLOAT,                                                        $
                VALUE = 1.0,                                                   $
                TITLE = 'Latitudinal Extent (0.0 < extent <= 180.0):  ',       $
                UVALUE = 'latext_cwf' )
  lonext_cwf= CW_FIELD(                                                        $
                latlonext_base,                                                $
                /ROW,                                                          $
                /FLOAT,                                                        $
                VALUE = 1.0,                                                   $
                TITLE = 'Longitudinal Extent (0.0 < extent <= 360.0): ',       $
                UVALUE = 'lonext_cwf' )
  metersext_base                                                               $
            = WIDGET_BASE(                                                     $
                ext_base,                                                      $
                MAP = 0,                                                       $
                /COLUMN,                                                       $
                /BASE_ALIGN_LEFT )
  xmeter_cwf= CW_FIELD(                                                        $
                metersext_base,                                                $
                /ROW,                                                          $
                /FLOAT,                                                        $
                VALUE = 111000.0,                                              $
                TITLE = 'Horizontal Extent (meters): ',                        $
                UVALUE = 'xmeterext_cwf' )
  ymeter_cwf= CW_FIELD(                                                        $
                metersext_base,                                                $
                /ROW,                                                          $
                /FLOAT,                                                        $
                VALUE = 111000.0,                                              $
                TITLE = 'Vertical Extent (meters):   ',                        $
                UVALUE = 'lonext_cwf' )
  pixelext_base                                                                $
            = WIDGET_BASE(                                                     $
                ext_base,                                                      $
                MAP = 0,                                                       $
                /COLUMN,                                                       $
                /BASE_ALIGN_LEFT )
  xpixel_cwf= CW_FIELD(                                                        $
                pixelext_base,                                                 $
                /ROW,                                                          $
                /FLOAT,                                                        $
                VALUE = 1.0,                                                   $
                TITLE = 'Horizontal Extent (275m/pixels): ',                   $
                UVALUE = 'xmeterext_cwf' )
  ypixel_cwf= CW_FIELD(                                                        $
                pixelext_base,                                                 $
                /ROW,                                                          $
                /FLOAT,                                                        $
                VALUE = 1.0,                                                   $
                TITLE = 'Vertical Extent (275m/pixels):   ',                   $
                UVALUE = 'lonext_cwf' )
  timebase1a= WIDGET_BASE(                                                     $
                tlb,                                                           $
                /FRAME,                                                        $
                /COLUMN )
  time1lbl  = WIDGET_LABEL(                                                    $
                timebase1a,                                                    $
                VALUE = 'Starting Time' )
  timebase1b= WIDGET_BASE(                                                     $
                timebase1a,                                                    $
                /ROW )
  misr_start_time                                                              $
            = '2000-03-03T00:00:00Z'
  yr_wd1    = WIDGET_DROPLIST(                                                 $
                timebase1b,                                                    $
                VALUE = STRTRIM([LINDGEN(10)+2000L],2),                        $
                UVALUE = 'yr_wd1',                                             $
                TITLE = 'Year:',                                               $
                /DYNAMIC_RESIZE )
  month_wd1 = WIDGET_DROPLIST(                                                 $
                timebase1b,                                                    $
                VALUE = STRTRIM([LINDGEN(12)+1L],2),                           $
                UVALUE = 'month_wd1',                                          $
                TITLE = 'Month:',                                              $
                /DYNAMIC_RESIZE )
  day_wd1   = WIDGET_DROPLIST(                                                 $
                timebase1b,                                                    $
                VALUE = STRTRIM([LINDGEN(31)+1L],2),                           $
                UVALUE = 'day_wd1',                                            $
                TITLE = 'Day:',                                                $
                /DYNAMIC_RESIZE )
  hour_wd1  = WIDGET_DROPLIST(                                                 $
                timebase1b,                                                    $
                VALUE = STRTRIM([LINDGEN(24)],2),                              $
                TITLE = 'Hour:',                                               $
                UVALUE = 'hour_wd1',                                           $
                /DYNAMIC_RESIZE )
  minute_wd1= WIDGET_DROPLIST(                                                 $
                timebase1b,                                                    $
                VALUE = STRTRIM([LINDGEN(60)],2),                              $
                UVALUE = 'minute_wd1',                                         $
                TITLE = 'Minute:',                                             $
                /DYNAMIC_RESIZE )
  second_wd1= WIDGET_DROPLIST(                                                 $
                timebase1b,                                                    $
                VALUE = STRTRIM([LINDGEN(60)],2),                              $
                TITLE = 'Second:',                                             $
                UVALUE = 'second_wd1',                                         $
                /DYNAMIC_RESIZE )
  time1_idx = LONARR(6)
  timebase2a= WIDGET_BASE(                                                     $
                tlb,                                                           $
                /FRAME,                                                        $
                /COLUMN )
  time2lbl  = WIDGET_LABEL(                                                    $
                timebase2a,                                                    $
                VALUE = 'Ending Time' )
  timebase2b= WIDGET_BASE(                                                     $
                timebase2a,                                                    $
                /ROW )
  misr_start_time                                                              $
            = '2000-03-03T00:00:00Z'
  yr_wd2    = WIDGET_DROPLIST(                                                 $
                timebase2b,                                                    $
                VALUE = STRTRIM([LINDGEN(10)+2000L],2),                        $
                UVALUE = 'yr_wd2',                                             $
                TITLE = 'Year:',                                               $
                /DYNAMIC_RESIZE )
  month_wd2 = WIDGET_DROPLIST(                                                 $
                timebase2b,                                                    $
                VALUE = STRTRIM([LINDGEN(12)+1L],2),                           $
                UVALUE = 'month_wd2',                                          $
                TITLE = 'Month:',                                              $
                /DYNAMIC_RESIZE )
  day_wd2   = WIDGET_DROPLIST(                                                 $
                timebase2b,                                                    $
                VALUE = STRTRIM([LINDGEN(31)+1L],2),                           $
                UVALUE = 'day_wd2',                                            $
                TITLE = 'Day:',                                                $
                /DYNAMIC_RESIZE )
  hour_wd2  = WIDGET_DROPLIST(                                                 $
                timebase2b,                                                    $
                VALUE = STRTRIM([LINDGEN(24)],2),                              $
                TITLE = 'Hour:',                                               $
                UVALUE = 'hour_wd2',                                           $
                /DYNAMIC_RESIZE )
  minute_wd2= WIDGET_DROPLIST(                                                 $
                timebase2b,                                                    $
                VALUE = STRTRIM([LINDGEN(60)],2),                              $
                UVALUE = 'minute_wd2',                                         $
                TITLE = 'Minute:',                                             $
                /DYNAMIC_RESIZE )
  second_wd2= WIDGET_DROPLIST(                                                 $
                timebase2b,                                                    $
                VALUE = STRTRIM([LINDGEN(60)],2),                              $
                TITLE = 'Second:',                                             $
                UVALUE = 'second_wd2',                                         $
                /DYNAMIC_RESIZE )
  time2_idx = LONARR(6)

  btn_base  = WIDGET_BASE(                                                     $
                tlb,                                                           $
                /ROW,                                                          $
                /FRAME,                                                        $
                /ALIGN_CENTER,                                                 $
                /BASE_ALIGN_CENTER)

  info_btn  = WIDGET_BUTTON(                                                   $
                btn_base,                                                      $
                VALUE = 'Get Information',                                     $
                UVALUE = 'get_info' )
  quit_btn  = WIDGET_BUTTON(                                                   $
                btn_base,                                                      $
                VALUE = 'Quit',                                                $
                UVALUE = 'quit' )
  help_btn  = WIDGET_BUTTON(                                                   $
                btn_base,                                                      $
                VALUE = 'Help',                                                $
                UVALUE = 'help' )

  WIDGET_CONTROL, tlb, /REALIZE

  ext_bases = [latlonext_base,metersext_base,pixelext_base]

  ptr       = PTR_NEW( {                                                       $
                        lat_cwf      : lat_cwf,                                $
                        lon_cwf      : lon_cwf,                                $
                        latext_cwf   : latext_cwf,                             $
                        lonext_cwf   : lonext_cwf,                             $
                        xmeter_cwf   : xmeter_cwf,                             $
                        ymeter_cwf   : ymeter_cwf,                             $
                        xpixel_cwf   : xpixel_cwf,                             $
                        ypixel_cwf   : ypixel_cwf,                             $
                        info_btn     : info_btn,                               $
                        quit_btn     : quit_btn,                               $
                        ext_idx      : 0L,                                     $
                        ext_bases    : ext_bases,                              $
                        time1_idx    : time1_idx,                              $
                        time2_idx    : time2_idx,                              $
                        help_base    : (-1L) } )
  WIDGET_CONTROL, tlb, SET_UVALUE = ptr

  XMANAGER, 'example2', tlb, EVENT_HANDLER = 'example2_eh'

  PTR_FREE, ptr
  
END
; example2
