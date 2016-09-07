@utilities.pro

;1234567890123456789012345678901234567890123456789012345678901234567890123456789
;===============================================================================
; show_coords
;===============================================================================
;;;ckt,2005nov29 PRO show_coords, x, y, tlb, info_base, mapinfo
PRO show_coords, x, y, tlb, info_base, sStatePtr

   IF x LT 0 OR x GE (*sStatePtr).XSIZE OR                                     $
      y LT 0 OR y GE (*sStatePtr).YSIZE THEN RETURN

   y = (*sStatePtr).YSIZE - (y+1)

   IF (*sStatePtr).NDIMS EQ 3 THEN BEGIN
      dvr  = STRTRIM(((*sStatePtr).image)[0,x,y],2)
      dvg  = STRTRIM(((*sStatePtr).image)[1,x,y],2)
      dvb  = STRTRIM(((*sStatePtr).image)[2,x,y],2)
   ENDIF ELSE BEGIN
      dvr  = STRTRIM(((*sStatePtr).image)[x,y],2)
   ENDELSE
;;;ckt,2006feb05 help,dvr
   IF WIDGET_INFO( info_base, /VALID_ID) THEN BEGIN
      IF WIDGET_INFO( info_base, /VALID_ID) THEN                               $
         WIDGET_CONTROL, info_base, GET_UVALUE = struct
      IF WIDGET_INFO( struct.xy_lbl, /VALID_ID) THEN                           $
         WIDGET_CONTROL, struct.xy_lbl,                                        $
         SET_VALUE = 'LINE: '+STRTRIM(y,2)+'   SAMPLE: '+STRTRIM(x,2)
      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_LS_TO_SOMXY( mapinfo,                                    $
      ;                            x,                                          $
      ;                            y,                                          $
      ;                            som_x                                       $
      ;                            som_y )
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
        routine = 'MTK_LS_TO_SOMXY'
        status  = MTK_LS_TO_SOMXY(                                             $
                    (*sStatePtr).mapinfo,                                      $
                    y,                                                         $
                    x,                                                         $
                    som_x,                                                     $
                    som_y )
        IF status THEN BEGIN
;;;ckt,2005nov29          msg     = ['Problem with routine '+routine+'... exiting...']
;;;ckt,2005nov29          res     = DIALOG_MESSAGE(msg, /ERROR)
          RETURN
        ENDIF
      IF WIDGET_INFO( struct.somxy_lbl, /VALID_ID) THEN                        $
         WIDGET_CONTROL, struct.somxy_lbl,                                     $
         SET_VALUE = 'SOM_X: '+STRTRIM(som_x,2)+                               $
                     '   SOM_Y: '+STRTRIM(som_y,2)
      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_SOMXY_TO_BLS( path,                                      $
      ;                             resolution,                                $
      ;                             som_x,                                     $
      ;                             som_y,                                     $
      ;                             block,                                     $
      ;                             line,                                      $
      ;                             sample )
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
        routine = 'MTK_SOMXY_TO_BLS'
        status  = MTK_SOMXY_TO_BLS( (*sStatePtr).mapinfo.path,                 $
                                    (*sStatePtr).mapinfo.resolution,           $
                                    som_x,                                     $
                                    som_y,                                     $
                                    block,                                     $
                                    line,                                      $
                                    sample )
        IF status THEN BEGIN
;;;ckt,2005nov29          msg     = ['Problem with routine '+routine+'... exiting...']
;;;ckt,2005nov29          res     = DIALOG_MESSAGE(msg, /ERROR)
          RETURN
        ENDIF

      IF WIDGET_INFO( struct.bls_lbl, /VALID_ID) THEN                          $
      WIDGET_CONTROL, struct.bls_lbl,                                          $
         SET_VALUE = 'BLOCK: '+STRTRIM(block,2)+                               $
                     '   LINE: '+STRTRIM(line,2)+                              $
                     '   SAMPLE: '+STRTRIM(sample,2)


      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_SOMXY_TO_LATLON( path,                                   $
      ;                                som_x,                                  $
      ;                                som_y,                                  $
      ;                                lat,                                    $
      ;                                lon )
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
      routine = 'MTK_SOMXY_TO_LATLON'
      status  = MTK_SOMXY_TO_LATLON( (*sStatePtr).mapinfo.path,                $
                                       som_x,                                  $
                                       som_y,                                  $
                                       lat,                                    $
                                       lon )
      IF status THEN BEGIN
;;;ckt,2005nov29        msg     = ['Problem with routine '+routine+'... exiting...']
;;;ckt,2005nov29        res     = DIALOG_MESSAGE(msg, /ERROR)
        RETURN
      ENDIF
      IF WIDGET_INFO( struct.ll_lbl, /VALID_ID) THEN                           $
         WIDGET_CONTROL, struct.ll_lbl,                                        $
         SET_VALUE = 'Latitude: '+STRTRIM(lat,2)+                              $
                     '   Longitude: '+STRTRIM(lon,2)

      IF WIDGET_INFO( struct.data_lbl_r, /VALID_ID) THEN                       $
         WIDGET_CONTROL, struct.data_lbl_r,                                    $
         SET_VALUE = 'DATA VALUE (RED PLANE):   '+dvr
      IF (*sStatePtr).NDIMS EQ 3 THEN BEGIN
         IF WIDGET_INFO( struct.data_lbl_g, /VALID_ID) THEN                    $
            WIDGET_CONTROL, struct.data_lbl_g,                                 $
            SET_VALUE = 'DATA VALUE (GREEN PLANE): '+dvg
         IF WIDGET_INFO( struct.data_lbl_b, /VALID_ID) THEN                    $
            WIDGET_CONTROL, struct.data_lbl_b,                                 $
            SET_VALUE = 'DATA VALUE (BLUE PLANE):  '+dvb
      ENDIF


      IF WIDGET_INFO( info_base, /VALID_ID) THEN                               $
         WIDGET_CONTROL, info_base, SET_UVALUE = struct
   ENDIF ELSE BEGIN
      b    = WIDGET_BASE(GROUP_LEADER = tlb,/FLOATING,/COLUMN,TLB_FRAME_ATTR=9)
      lbl1 = WIDGET_LABEL(b,VALUE = 'Image Coordinates')
      lbl2 = WIDGET_LABEL(                                                     $
                b,                                                             $
                VALUE = 'LINE: '+STRTRIM(y,2)+'   SAMPLE: '+STRTRIM(x,2),      $
                /DYNAMIC_RESIZE)
      dum  = WIDGET_LABEL(b,VALUE='______________________')
      lbl3 = WIDGET_LABEL(b,VALUE = 'SOM Coordinates')
      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_LS_TO_SOMXY( mapinfo,                                    $
      ;                            x,                                          $
      ;                            y,                                          $
      ;                            som_x                                       $
      ;                            som_y )
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
        routine = 'MTK_LS_TO_SOMXY'
        status  = MTK_LS_TO_SOMXY(                                             $
                    (*sStatePtr).mapinfo,                                      $
                    y,                                                         $
                    x,                                                         $
                    som_x,                                                     $
                    som_y )
        IF status THEN BEGIN
;;;ckt,2005nov29          msg     = ['Problem with routine '+routine+'... exiting...']
;;;ckt,2005nov29          res     = DIALOG_MESSAGE(msg, /ERROR)
          RETURN
        ENDIF

      lbl4 = WIDGET_LABEL(                                                     $
                b,                                                             $
                VALUE = 'SOM_X: '+STRTRIM(som_x,2)+                            $
                     '   SOM_Y: '+STRTRIM(som_y,2),                            $
                /DYNAMIC_RESIZE)

      dum  = WIDGET_LABEL(b,VALUE='______________________')
      lbl5 = WIDGET_LABEL(b,VALUE = 'Block/Line/Sample Coordinates')
      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_SOMXY_TO_BLS( path,                                      $
      ;                             resolution,                                $
      ;                             som_x,                                     $
      ;                             som_y,                                     $
      ;                             block,                                     $
      ;                             line,                                      $
      ;                             sample )
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
        routine = 'MTK_SOMXY_TO_BLS'
        status  = MTK_SOMXY_TO_BLS( (*sStatePtr).mapinfo.path,                 $
                                    (*sStatePtr).mapinfo.resolution,           $
                                    som_x,                                     $
                                    som_y,                                     $
                                    block,                                     $
                                    line,                                      $
                                    sample )
        IF status THEN BEGIN
;;;ckt,2005nov29          msg     = ['Problem with routine '+routine+'... exiting...']
;;;ckt,2005nov29          res     = DIALOG_MESSAGE(msg, /ERROR)
          RETURN
        ENDIF
      lbl6 = WIDGET_LABEL(                                                     $
                b,                                                             $
                VALUE = 'BLOCK: '+STRTRIM(block,2)+                            $
                     '   LINE: '+STRTRIM(line,2)+                              $
                     '   SAMPLE: '+STRTRIM(sample,2),                          $
                /DYNAMIC_RESIZE)

      dum  = WIDGET_LABEL(b,VALUE='______________________')
      lbl7 = WIDGET_LABEL(b,VALUE = 'Latitude/Longitude')
      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_SOMXY_TO_LATLON( path,                                   $
      ;                                som_x,                                  $
      ;                                som_y,                                  $
      ;                                lat,                                    $
      ;                                lon )
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
        routine = 'MTK_SOMXY_TO_LATLON'
        status  = MTK_SOMXY_TO_LATLON( (*sStatePtr).mapinfo.path,              $
                                       som_x,                                  $
                                       som_y,                                  $
                                       lat,                                    $
                                       lon )
        IF status THEN BEGIN
;;;ckt,2005nov29          msg     = ['Problem with routine '+routine+'... exiting...']
;;;ckt,2005nov29          res     = DIALOG_MESSAGE(msg, /ERROR)
          RETURN
        ENDIF
      lbl8 = WIDGET_LABEL(                                                     $
                b,                                                             $
                VALUE = 'Latitude: '+STRTRIM(lat,2)+                           $
                     '   Longitude: '+STRTRIM(lon,2),                          $
                /DYNAMIC_RESIZE)

      dum  = WIDGET_LABEL(b,VALUE='______________________')
      IF (*sStatePtr).NDIMS EQ 3 THEN BEGIN
         lbl9r = WIDGET_LABEL(                                                 $
                   b,                                                          $
                   VALUE = 'DATA VALUE (RED PLANE):   '+dvr,                   $
                   /DYNAMIC_RESIZE)
         lbl9g = WIDGET_LABEL(                                                 $
                   b,                                                          $
                   VALUE = 'DATA VALUE (GREEN PLANE): '+dvg,                   $
                   /DYNAMIC_RESIZE)
         lbl9b = WIDGET_LABEL(                                                 $
                   b,                                                          $
                   VALUE = 'DATA VALUE (BLUE PLANE):  '+dvb,                   $
                   /DYNAMIC_RESIZE)
      ENDIF ELSE BEGIN
         lbl9r = WIDGET_LABEL(                                                 $
                   b,                                                          $
                   VALUE = 'DATA VALUE:               '+dvr,                   $
                   /DYNAMIC_RESIZE)
         lbl9g = WIDGET_LABEL(                                                 $
                   b,                                                          $
                   VALUE = '',                                                 $
                   /DYNAMIC_RESIZE)
         lbl9b = WIDGET_LABEL(                                                 $
                   b,                                                          $
                   VALUE = '',                                                 $
                   /DYNAMIC_RESIZE)
      ENDELSE

      WIDGET_CONTROL, b, /REALIZE
      WIDGET_CONTROL, b, SET_UVALUE = {                                        $
                                       xy_lbl    : lbl2,                       $
                                       somxy_lbl : lbl4,                       $
                                       bls_lbl   : lbl6,                       $
                                       ll_lbl    : lbl8,                       $
                                       data_lbl_r: lbl9r,                      $
                                       data_lbl_g: lbl9g,                      $
                                       data_lbl_b: lbl9b }

      WIDGET_CONTROL, WIDGET_INFO(tlb,/CHILD), SET_UVALUE = b
   ENDELSE
END
; show_coords

;1234567890123456789012345678901234567890123456789012345678901234567890123456789
;===============================================================================
; SLIDE_IMG_MTK_EVENT
;===============================================================================
PRO SLIDE_IMG_MTK_EVENT, ev
  COMPILE_OPT hidden

  ; Check for kill of top level base.
  if (TAG_NAMES(ev, /STRUCTURE_NAME) EQ 'WIDGET_KILL_REQUEST') then begin
    WIDGET_CONTROL, ev.top, /DESTROY
    RETURN
  endif

  WIDGET_CONTROL, ev.top, GET_UVALUE=sStatePtr
  WIDGET_CONTROL, WIDGET_INFO(ev.top,/CHILD), GET_UVALUE = info_base


  WIDGET_CONTROL, ev.id, GET_UVALUE=uval


  CASE uval OF
     'FULL_IMAGE': BEGIN
        CASE ev.type OF
           0: BEGIN
              xxx = FLOOR(ev.x*(*sStatePtr).XSIZE_OVER_XVISIBLE)
              yyy = FLOOR(ev.y*(*sStatePtr).YSIZE_OVER_YVISIBLE)
              END
           2: BEGIN
              xxx = FLOOR(ev.x*(*sStatePtr).XSIZE_OVER_XVISIBLE)
              yyy = FLOOR(ev.y*(*sStatePtr).YSIZE_OVER_YVISIBLE)
;;;print,TAG_NAMES(*sStatePtr)
;;;ckt,2005nov29              show_coords, xxx, yyy, ev.top, info_base, (*sStatePtr).mapinfo
              show_coords, xxx, yyy, ev.top, info_base, sStatePtr
              END
           4: BEGIN ; Expose event
              WSET, (*sStatePtr).FULL_WINDOW
              IF ((*sStatePtr).USE_CONGRID) THEN BEGIN
                 IF (*sStatePtr).NDIMS EQ 3 THEN $
                    TV,                                                        $
                       CONGRID(HIST_EQUAL(BYTSCL((*sStatePtr).image)),         $
                       3,                                                      $
                       (*sStatePtr).XVISIBLE,                                  $
                       (*sStatePtr).YVISIBLE),                                 $
                       ORDER=(*sStatePtr).ORDER,                               $
                       /TRUE $
                 ELSE $
                    TV,                                                        $
                       CONGRID(HIST_EQUAL(BYTSCL((*sStatePtr).image)),         $
                       (*sStatePtr).XVISIBLE,                                  $
                       (*sStatePtr).YVISIBLE),                                 $
                       ORDER=(*sStatePtr).ORDER
              ENDIF ELSE BEGIN
                 TV, HIST_EQUAL(BYTSCL((*sStatePtr).image)),                   $
                     ORDER=(*sStatePtr).ORDER,                                 $
                     TRUE = (*sStatePtr).NDIMS EQ 3
              ENDELSE
              END
           ELSE:
        ENDCASE
        END
     'SLIDE_IMAGE': BEGIN
        CASE ev.type OF
           0: BEGIN
              END
           2: BEGIN
              xxx = ev.x
              yyy = ev.y
;;;ckt,2005nov29              show_coords, xxx, yyy, ev.top, info_base, (*sStatePtr).mapinfo
              show_coords, xxx, yyy, ev.top, info_base, sStatePtr
              END
           3: BEGIN  ; Scroll event
              WIDGET_CONTROL, ev.id, GET_DRAW_VIEW = viewport_lower_xy
              WSET, (*sStatePtr).SLIDE_WINDOW
              TV,                                                              $
                 HIST_EQUAL(BYTSCL((*sStatePtr).image)),                       $
                 ORDER=(*sStatePtr).ORDER,                                     $
                 TRUE = (*sStatePtr).NDIMS EQ 3

              WSET, (*sStatePtr).FULL_WINDOW
              IF ((*sStatePtr).USE_CONGRID) THEN BEGIN
                 IF (*sStatePtr).NDIMS EQ 3 THEN $
                    TV,                                                        $
                       CONGRID(HIST_EQUAL(BYTSCL((*sStatePtr).image)),         $
                       3,                                                      $
                       (*sStatePtr).XVISIBLE,                                  $
                       (*sStatePtr).YVISIBLE),                                 $
                       ORDER=(*sStatePtr).ORDER,                               $
                       /TRUE $
                 ELSE $
                    TV,                                                        $
                       CONGRID(HIST_EQUAL(BYTSCL((*sStatePtr).image)),         $
                       (*sStatePtr).XVISIBLE,                                  $
                       (*sStatePtr).YVISIBLE),                                 $
                       ORDER=(*sStatePtr).ORDER
              ENDIF ELSE BEGIN
                 TV,                                                           $
                    HIST_EQUAL(BYTSCL((*sStatePtr).image)),                    $
                    ORDER=(*sStatePtr).ORDER,                                  $
                    TRUE = (*sStatePtr).NDIMS EQ 3
              ENDELSE

              x1 = viewport_lower_xy[0]*(1.0/(*sStatePtr).XSIZE_OVER_XVISIBLE)
              x2 = x1+                                                         $
                 (((*sStatePtr).XVISIBLE-1)*                                   $
                 (1.0/(*sStatePtr).XSIZE_OVER_XVISIBLE))
              x3 = x2
              x4 = x1
              y1 = viewport_lower_xy[1]*(1.0/(*sStatePtr).YSIZE_OVER_YVISIBLE)
              y2 = y1
              y3 = y2+                                                         $
                 (((*sStatePtr).YVISIBLE-1)*                                   $
                 (1.0/(*sStatePtr).YSIZE_OVER_YVISIBLE))
              y4 = y3

              PLOTS, ROUND([x1,x2,x3,x4,x1]),ROUND([y1,y2,y3,y4,y1]), /DEVICE


              END

           4: BEGIN ; Expose event
              WSET, (*sStatePtr).SLIDE_WINDOW
              TV,                                                              $
                 HIST_EQUAL(BYTSCL((*sStatePtr).image)),                       $
                 ORDER=(*sStatePtr).ORDER,                                     $
                 TRUE = (*sStatePtr).NDIMS EQ 3
              END
           ELSE:
        ENDCASE
        END

     'DONE': WIDGET_CONTROL, ev.top, /DESTROY

     ELSE:
  ENDCASE

END
; SLIDE_IMG_MTK_EVENT

;1234567890123456789012345678901234567890123456789012345678901234567890123456789
;===============================================================================
; create_slide_image
;===============================================================================
PRO create_slide_image, e

   SWIN = !D.WINDOW

   XSIZE       = e.XSIZE
   YSIZE       = e.YSIZE
   XVISIBLE    = e.XVISIBLE
   YVISIBLE    = e.YVISIBLE
   SHOW_FULL   = e.SHOW_FULL
   ORDER       = e.ORDER
   USE_CONGRID = e.USE_CONGRID
   RETAIN      = e.RETAIN
   TITLE       = e.TITLE
   REGISTER    = e.REGISTER
   BLOCK       = e.BLOCK

   IF (REGISTER OR BLOCK) THEN BEGIN
      base = WIDGET_BASE(                                                      $
                TITLE = TITLE,                                                 $
                GROUP = GROUP,                                                 $
                /COLUMN )
      junk = WIDGET_BUTTON(                                                    $
                WIDGET_BASE(base),                                             $
                VALUE='Done',                                                  $
                UVALUE='DONE')
      ibase= WIDGET_BASE(                                                      $
                base,                                                          $
                /ROW)
   ENDIF ELSE BEGIN
      base = WIDGET_BASE(                                                      $
                TITLE=title,                                                   $
                GROUP = GROUP,                                                 $
                /ROW)
      ibase= base
   ENDELSE
   ; Setting the managed attribute indicates our intention to put this app
   ; under the control of XMANAGER, and prevents our draw widgets from
   ; becoming candidates for becoming the default window on WSET, -1. XMANAGER
   ; sets this, but doing it here prevents our own WSETs at startup from
   ; having that problem.
   WIDGET_CONTROL, /MANAGED, base

   ; Expose and viewport events need not be reported if RETAIN=2, nor
   ; do they need to be reported if no image is present.  Otherwise,
   ; report these events.
;   doEvents = (RETAIN eq 2 ? 0 : 1)
   doEvents = 1

   IF (SHOW_FULL) THEN BEGIN
      fbase = WIDGET_BASE(                                                     $
                 ibase,                                                        $
                 /COLUMN,                                                      $
                 /FRAME)
      junk  = WIDGET_LABEL(                                                    $
                 fbase,                                                        $
                 VALUE='Full Image')
      all   = WIDGET_DRAW(                                                     $
                 fbase,                                                        $
                 RETAIN=RETAIN,                                                $
                 xsize=xvisible,                                               $
                 /MOTION_EVENTS,                                               $
                 /BUTTON_EVENTS,                                               $
                 ysize=yvisible,                                               $
                 uvalue='FULL_IMAGE',                                          $
                 EXPOSE_EVENTS=doEvents,                                       $
		 EVENT_PRO='SLIDE_IMG_MTK_EVENT')
      sbase = WIDGET_BASE(                                                     $
                 ibase,                                                        $
                 /COLUMN,                                                      $
                 /FRAME)
      junk  = WIDGET_LABEL(                                                    $
                 sbase,                                                        $
                 value='Full Resolution')
      scroll= WIDGET_DRAW(                                                     $
                sbase,                                                         $
                RETAIN=RETAIN,                                                 $
                XSIZE = XSIZE,                                                 $
                YSIZE = YSIZE,                                                 $
		/SCROLL,                                                       $
                 /MOTION_EVENTS,                                               $
                 /BUTTON_EVENTS,                                               $
		X_SCROLL_SIZE=XVISIBLE,                                        $
		Y_SCROLL_SIZE=YVISIBLE,                                        $
                UVALUE='SLIDE_IMAGE',                                          $
                EXPOSE_EVENTS=doEvents,                                        $
		VIEWPORT_EVENTS=doEvents,                                      $
		EVENT_PRO='SLIDE_IMG_MTK_EVENT')
      WIDGET_CONTROL, /REAL, base
      WIDGET_CONTROL, all, GET_VALUE = FULL_WINDOW
      e.FULL_WINDOW = FULL_WINDOW
  ENDIF ELSE BEGIN
      scroll= WIDGET_DRAW(                                                     $
                ibase,                                                         $
                RETAIN=RETAIN,                                                 $
                XSIZE = XSIZE,                                                 $
                YSIZE = YSIZE,                                                 $
		/SCROLL,                                                       $
		/FRAME,                                                        $
		X_SCROLL_SIZE=XVISIBLE,                                        $
                 /BUTTON_EVENTS,                                               $
                 /MOTION_EVENTS,                                               $
		Y_SCROLL_SIZE=YVISIBLE,                                        $
                UVALUE='SLIDE_IMAGE',                                          $
                EXPOSE_EVENTS=doEvents,                                        $
		VIEWPORT_EVENTS=doEvents,                                      $
		EVENT_PRO='SLIDE_IMG_MTK_EVENT')
    WIDGET_CONTROL, /REAL, base
    FULL_WINDOW = (-1L)
  ENDELSE

  WIDGET_CONTROL, scroll, GET_VALUE=SLIDE_WINDOW

  e.SLIDE_WINDOW = SLIDE_WINDOW

  IF (doEvents NE 0) THEN BEGIN
     sStatePtr = PTR_NEW(e,/NO_COPY)
;     {                                                                $
;               image      : e.image, $
;               useCongrid : USE_CONGRID, $
;               fullWin    : FULL_WINDOW, $
;               slideWin   : SLIDE_WINDOW, $
;               xvisible   : xvisible, $
;               yvisible   : yvisible, $
;               order      : order }
     WIDGET_CONTROL, base, SET_UVALUE=sStatePtr
     WIDGET_CONTROL, WIDGET_INFO(base,/CHILD), SET_UVALUE = (-1L)
  ENDIF

  ; Show the image(s) if one is present
  IF (SHOW_FULL) THEN BEGIN
     WSET, FULL_WINDOW
     IF (USE_CONGRID) THEN BEGIN
        IF (*sStatePtr).NDIMS EQ 3 THEN                               $
            TV, CONGRID(HIST_EQUAL(BYTSCL((*sStatePtr).image)),       $
                3, XVISIBLE, YVISIBLE), ORDER=ORDER, /TRUE                                     $
         ELSE $
            TV, CONGRID(HIST_EQUAL(BYTSCL((*sStatePtr).image)),       $
                XVISIBLE, YVISIBLE), ORDER=ORDER
     ENDIF ELSE BEGIN
        TV, HIST_EQUAL(BYTSCL((*sStatePtr).image)),                   $
            ORDER=ORDER, TRUE = (*sStatePtr).NDIMS EQ 3
     ENDELSE
  ENDIF
  WSET, SLIDE_WINDOW
  TV, HIST_EQUAL(BYTSCL((*sStatePtr).image)), ORDER=ORDER,            $
     TRUE = (*sStatePtr).NDIMS EQ 3

  IF (N_ELEMENTS(group) EQ 0) THEN GROUP=base
  WSET, SWIN


  WSET, SLIDE_WINDOW
  WIDGET_CONTROL, scroll, GET_DRAW_VIEW = viewport_lower_xy
  x1 = viewport_lower_xy[0]*(1.0/(*sStatePtr).XSIZE_OVER_XVISIBLE)
  x2 = x1+(((*sStatePtr).XVISIBLE-1)*(1.0/(*sStatePtr).XSIZE_OVER_XVISIBLE))
  x3 = x2
  x4 = x1
  y1 = viewport_lower_xy[1]*(1.0/(*sStatePtr).YSIZE_OVER_YVISIBLE)
  y2 = y1
  y3 = y2+(((*sStatePtr).YVISIBLE-1)*(1.0/(*sStatePtr).YSIZE_OVER_YVISIBLE))
  y4 = y3
  WSET, FULL_WINDOW
  PLOTS, ROUND([x1,x2,x3,x4,x1]),ROUND([y1,y2,y3,y4,y1]), /DEVICE


  IF (REGISTER OR BLOCK) THEN $
    XMANAGER, 'SLIDE_IMAGE', base, EVENT='SLIDE_IMG_MTK_EVENT', $
	NO_BLOCK=(NOT(FLOAT(block)))
END
; create_slide_image

; $Id: slide_image.pro,v 1.18 2004/01/21 15:55:02 scottm Exp $
;
; Copyright (c) 1991-2004, Research Systems, Inc.  All rights reserved.
;	Unauthorized reproduction prohibited.
;+
; NAME:
;	SLIDE_IMAGE
;
; PURPOSE:
;	Create a scrolling graphics window for examining large images.
;	By default, 2 draw widgets are used.  The left draw widget shows
;	a reduced version of the complete image, while the draw widget on
;	the right displays the actual image with scrollbars that allow sliding
;	the visible window.
;
; CALLING SEQUENCE:
;	SLIDE_IMAGE [, Image]
;
; INPUTS:
;	Image:	The 2-dimensional image array to be displayed.  If this
;		argument is not specified, no image is displayed. The
;		FULL_WINDOW and SCROLL_WINDOW keywords can be used to obtain
;		the window numbers of the 2 draw widgets so they can be drawn
;		into at a later time.
;
; KEYWORDS:
;      CONGRID:	Normally, the image is processed with the CONGRID
;		procedure before it is written to the fully visible
;		window on the left. Specifying CONGIRD=0 will force
;		the image to be drawn as is.
;
;  FULL_WINDOW:	A named variable in which to store the IDL window number of \
;		the non-sliding window.  This window number can be used with
;		the WSET procedure to draw to the scrolling window at a later
;		point.
;
;	GROUP:	The widget ID of the widget that calls SLIDE_IMAGE.  If this
;		keyword is specified, the death of the caller results in the
;		death of SLIDE_IMAGE.
;
;	BLOCK:  Set this keyword to have XMANAGER block when this
;		application is registered.  By default the Xmanager
;               keyword NO_BLOCK is set to 1 to provide access to the
;               command line if active command 	line processing is available.
;               Note that setting BLOCK for this application will cause
;		all widget applications to block, not only this
;		application.  For more information see the NO_BLOCK keyword
;		to XMANAGER.
;
;	ORDER:	This keyword is passed directly to the TV procedure
;		to control the order in which the images are drawn. Usually,
;		images are drawn from the bottom up.  Set this keyword to a
;		non-zero value to draw images from the top down.
;
;     REGISTER:	Set this keyword to create a "Done" button for SLIDE_IMAGE
;		and register the widgets with the XMANAGER procedure.
;
;		The basic widgets used in this procedure do not generate
;		widget events, so it is not necessary to process events
;		in an event loop.  The default is therefore to simply create
;		the widgets and return.  Hence, when register is not set,
;		SLIDE_IMAGE can be displayed and the user can still type
;		commands at the "IDL>" prompt that use the widgets.
;
;	RETAIN:	This keyword is passed directly to the WIDGET_DRAW
;		function, and controls the type of backing store
;		used for the draw windows.  If not present, a value of
;		2 is used to make IDL handle backing store.  It is
;		recommended that if RETAIN is set to zero, then the
;		REGISTER keyword should be set so that expose and scroll
;		events are handled.
;
; SLIDE_WINDOW:	A named variable in which to store the IDL window number of
;		the sliding window.  This window number can be used with the
;		WSET procedure to draw to the scrolling window at a later
;		time.
;
;	TITLE:	The title to be used for the SLIDE_IMAGE widget.  If this
;		keyword is not specified, "Slide Image" is used.
;
;	TOP_ID:	A named variable in which to store the top widget ID of the
;		SLIDE_IMAGE hierarchy.  This ID can be used to kill the
;		hierarchy as shown below:
;
;			SLIDE_IMAGE, TOP_ID=base, ...
;			.
;			.
;			.
;			WIDGET_CONTROL, /DESTROY, base
;
;	XSIZE:	The maximum width of the image that can be displayed by
;		the scrolling window.  This keyword should not be confused
;		with the visible size of the image, controlled by the XVISIBLE
;		keyword.  If XSIZE is not specified, the width of Image is
;		used.  If Image is not specified, 256 is used.
;
;     XVISIBLE:	The width of the viewport on the scrolling window.  If this
;		keyword is not specified, 256 is used.
;
;	YSIZE:	The maximum height of the image that can be displayed by
;		the scrolling window.  This keyword should not be confused
;		with the visible size of the image, controlled by the YVISIBLE
;		keyword.  If YSIZE is not present the height of Image is used.
;		If Image is not specified, 256 is used.
;
;     YVISIBLE:	The height of the viewport on the scrolling window. If
;		this keyword is not present, 256 is used.
;
; OUTPUTS:
;	None.
;
; COMMON BLOCKS:
;	None.
;
; SIDE EFFECTS:
;	Widgets for displaying a very large image are created.
;	The user typically uses the window manager to destroy
;	the window, although the TOP_ID keyword can also be used to
;	obtain the widget ID to use in destroying it via WIDGET_CONTROL.
;
; RESTRICTIONS:
;	Scrolling windows don't work correctly if backing store is not
;	provided.  They work best with window-system-provided backing store
;	(RETAIN=1), but are also usable with IDL provided backing store
;	(RETAIN=2).
;
;	Various machines place different restrictions on the size of the
;	actual image that can be handled.
;
; MODIFICATION HISTORY:
;	7 August, 1991, Written by AB, RSI.
;	10 March, 1993, ACY, Change default RETAIN=2
;	23 Sept., 1994  KDB, Fixed Typo in comments. Fixed error in
;			Congrid call. xvisible was used instead of yvisible.
;	20 March, 2001  DLD, Add event handling for expose and scroll events
;			when RETAIN=0.
;-
;pro slide_image_mtk, image, CONGRID=USE_CONGRID, ORDER=ORDER, $
;       REGISTER=REGISTER, $
;	RETAIN=RETAIN, SHOW_FULL=SHOW_FULL, SLIDE_WINDOW=SLIDE_WINDOW, $
;	XSIZE=XSIZE, XVISIBLE=XVISIBLE, YSIZE=YSIZE, YVISIBLE=YVISIBLE, $
;	TITLE=TITLE, TOP_ID=BASE, FULL_WINDOW=FULL_WINDOW, GROUP = GROUP, $
;	BLOCK=block

;1234567890123456789012345678901234567890123456789012345678901234567890123456789
;===============================================================================
; slide_image_mtk
;===============================================================================
PRO slide_image_mtk, image, mapinfo, _Extra = e


  IF N_TAGS(e) LE 0 THEN BEGIN
     e             = CREATE_STRUCT('USE_CONGRID',1)
     tag_names2add = [                                                         $
                      'ORDER',                                                 $
                      'REGISTER',                                              $
                      'RETAIN',                                                $
                      'SHOW_FULL',                                             $
                      'SLIDE_WINDOW',                                          $
                      'XSIZE',                                                 $
                      'YSIZE',                                                 $
                      'XVISIBLE',                                              $
                      'YVISIBLE',                                              $
                      'TITLE',                                                 $
                      'BASE',                                                  $
                      'FULL_WINDOW',                                           $
                      'GROUP',                                                 $
                      'BLOCK' ]
     tag_vals2add  = [                                                         $
                      PTR_NEW(1),                                              $
                      PTR_NEW(0),                                              $
                      PTR_NEW(2),                                              $
                      PTR_NEW(1),                                              $
                      PTR_NEW(-1L),                                            $
                      PTR_NEW(256),                                            $
                      PTR_NEW(256),                                            $
                      PTR_NEW(256),                                            $
                      PTR_NEW(256),                                            $
                      PTR_NEW('MTK Slide Image'),                              $
                      PTR_NEW(-1L),                                            $
                      PTR_NEW(-1L),                                            $
                      PTR_NEW(-1L),                                            $
                      PTR_NEW(0) ]
     FOR j = 0, N_ELEMENTS(tag_names2add)-1 DO                                 $
        e = CREATE_STRUCT(e,tag_names2add[j],*(tag_vals2add[j]))
  ENDIF ELSE BEGIN
     tags = STRTRIM(STRUPCASE(TAG_NAMES(e)),2)
     IF (WHERE(tags EQ 'CONGRID'))[0] LT 0 THEN                                $
        e = CREATE_STRUCT(e,'USE_CONGRID',1)
     IF (WHERE(tags EQ 'ORDER'))[0] LT 0 THEN                                  $
        e = CREATE_STRUCT(e,'ORDER',1)
     IF (WHERE(tags EQ 'REGISTER'))[0] LT 0 THEN                               $
        e = CREATE_STRUCT(e,'REGISTER',0)
     IF (WHERE(tags EQ 'RETAIN'))[0] LT 0 THEN                                 $
        e = CREATE_STRUCT(e,'RETAIN',2)
     IF (WHERE(tags EQ 'SHOW_FULL'))[0] LT 0 THEN                              $
        e = CREATE_STRUCT(e,'SHOW_FULL',1)
     IF (WHERE(tags EQ 'SLIDE_WINDOW'))[0] LT 0 THEN                           $
        e = CREATE_STRUCT(e,'SLIDE_WINDOW',(-1L))
     IF (WHERE(tags EQ 'XSIZE'))[0] LT 0 THEN                                  $
        e = CREATE_STRUCT(e,'XSIZE',256)
     IF (WHERE(tags EQ 'YSIZE'))[0] LT 0 THEN                                  $
        e = CREATE_STRUCT(e,'YSIZE',256)
     IF (WHERE(tags EQ 'XVISIBLE'))[0] LT 0 THEN                               $
        e = CREATE_STRUCT(e,'XVISIBLE',256)
     IF (WHERE(tags EQ 'YVISIBLE'))[0] LT 0 THEN                               $
        e = CREATE_STRUCT(e,'YVISIBLE',256)
     IF (WHERE(tags EQ 'TITLE'))[0] LT 0 THEN                                  $
        e = CREATE_STRUCT(e,'TITLE','MTK Slide Image')
     IF (WHERE(tags EQ 'BASE'))[0] LT 0 THEN                                   $
        e = CREATE_STRUCT(e,'BASE',(-1L))
     IF (WHERE(tags EQ 'FULL_WINDOW'))[0] LT 0 THEN                            $
        e = CREATE_STRUCT(e,'FULL_WINDOW',(-1L))
     IF (WHERE(tags EQ 'GROUP'))[0] LT 0 THEN                                  $
        e = CREATE_STRUCT(e,'GROUP',(-1L))
     IF (WHERE(tags EQ 'BLOCK'))[0] LT 0 THEN                                  $
        e = CREATE_STRUCT(e,'BLOCK',0)
  ENDELSE

  SWIN = !D.WINDOW

  IF N_PARAMS() LE 0 THEN BEGIN
     txt = ['No image specified... ']
     res = DIALOG_MESSAGE(txt,/INFORMATION)
     RETURN
  ENDIF

  image_size = SIZE(image,/DIMENSIONS)
  IF                                                                           $
     (N_ELEMENTS(image_size) NE 2 AND N_ELEMENTS(image_size) NE 3) OR          $
     (N_ELEMENTS(image_size) EQ 3 AND (WHERE(image_size EQ 3))[0] LT 0)        $
  THEN BEGIN
     txt = [                                                                   $
            'Problem with image dimensions...',                                $
            'Image must be 2D array or 3D array with one dimension equal to 3']
     res = DIALOG_MESSAGE(txt,/ERROR)
     RETURN
  ENDIF ELSE BEGIN
     idx = WHERE(image_size EQ 3,cnt)
     IF cnt GT 1 AND N_ELEMENTS(image_size) EQ 3 THEN BEGIN
        txt = [                                                                $
               'More than one dimension equals 3',                             $
               'cannot determine "band" dimension']
        res = DIALOG_MESSAGE(txt,/INFORMATION)
        RETURN
     ENDIF
     IF cnt EQ 1 AND N_ELEMENTS(image_size) EQ 3 THEN BEGIN
     	idx2  = WHERE(image_size NE 3)
        image = TRANSPOSE(image,[idx[0],idx2[0],idx2[1]])
     ENDIF
     image_size = SIZE(image,/DIMENSIONS)
     e.XSIZE = image_size[N_ELEMENTS(image_size)-2]
     e.YSIZE = image_size[N_ELEMENTS(image_size)-1]
  ENDELSE

  e           = CREATE_STRUCT(e,'NDIMS',N_ELEMENTS(image_size))
  e           = CREATE_STRUCT(e,'IMAGE',image)
  screen_dims = GET_SCREEN_SIZE()
  ratio       = FLOAT(e.XSIZE)/FLOAT(e.YSIZE)
  IF e.XSIZE GT e.YSIZE THEN BEGIN
     e.XVISIBLE = screen_dims[0]*0.90/2.0
     e.YVISIBLE = e.XVISIBLE*(1.0/ratio)
  ENDIF ELSE BEGIN
     e.YVISIBLE = screen_dims[1]*0.90/2.0
     e.XVISIBLE = e.YVISIBLE*ratio
  ENDELSE

  IF e.XVISIBLE GT e.XSIZE AND e.YVISIBLE GT e.YSIZE THEN BEGIN
     e.YVISIBLE = e.YSIZE
     e.XVISIBLE = e.XSIZE
  ENDIF

  e           = CREATE_STRUCT(e,'XSIZE_OVER_XVISIBLE',FLOAT(e.XSIZE)/e.XVISIBLE)
  e           = CREATE_STRUCT(e,'YSIZE_OVER_YVISIBLE',FLOAT(e.YSIZE)/e.YVISIBLE)
  e           = CREATE_STRUCT(e,'MAPINFO', mapinfo)

  create_slide_image, e

END
; slide_image_mtk

;1234567890123456789012345678901234567890123456789012345678901234567890123456789
;===============================================================================
; example3_eh
;===============================================================================
PRO example3_eh, event
  WIDGET_CONTROL, event.id, GET_UVALUE = widget_name
  widget_name  = STRTRIM(STRUPCASE(widget_name),2)
  WIDGET_CONTROL, event.top, GET_UVALUE = ptr

  CASE widget_name OF
    'CREATE_VIEWER': BEGIN
      WIDGET_CONTROL, (*ptr).sb_cwf, GET_VALUE = sb
      WIDGET_CONTROL, (*ptr).eb_cwf, GET_VALUE = eb
      IF sb GT eb THEN BEGIN
        txt  = [                                                               $
          'Starting block value cannot be greater than ending block value' ]
        res  = DIALOG_MESSAGE( txt, /ERROR )
        RETURN
      ENDIF
      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_FILE_TO_BLOCKRANGE( file,                                $
      ;                                   start_block,                         $
      ;                                   end_block )
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
      routine = 'MTK_FILE_TO_BLOCKRANGE'
      status  = MTK_FILE_TO_BLOCKRANGE(                                        $
                  (*ptr).current_file,                                         $
                  start_block,                                                 $
                  end_block )
      IF status THEN BEGIN
        msg     = ['Problem with routine '+routine+'... exiting...']
        res     = DIALOG_MESSAGE(msg, /ERROR)
        RETURN
      ENDIF
      IF sb LT start_block OR eb GT end_block THEN BEGIN
        txt  = [                                                               $
          'Starting and/or ending block values outside of valid range' ]
        res  = DIALOG_MESSAGE( txt, /ERROR )
        RETURN
      ENDIF
;;;ckt,2005dec15      IF eb-sb+1 GT 3 THEN BEGIN
;;;ckt,2005dec15        txt  = [                                                               $
;;;ckt,2005dec15          'This example program is best used with 3 or fewer blocks.',         $
;;;ckt,2005dec15          'Do you want to continue?' ]
;;;ckt,2005dec15        res  = DIALOG_MESSAGE( txt, /QUESTION )
;;;ckt,2005dec15        IF STRTRIM(STRUPCASE(res),2) EQ 'NO' THEN RETURN
;;;ckt,2005dec15      ENDIF

      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_FILE_TO_PATH( file,                                      $
      ;                             path )
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
      routine = 'MTK_FILE_TO_PATH'
      status  = MTK_FILE_TO_PATH(                                              $
                  (*ptr).current_file,                                         $
                  path )
      IF status THEN BEGIN
        msg     = ['Problem with routine '+routine+'... exiting...']
        res     = DIALOG_MESSAGE(msg, /ERROR)
        RETURN
      ENDIF

      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_FILE_TO_PATH( file,                                      $
      ;                             path )
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
      routine = 'MTK_FILE_TO_PATH'
      status  = MTK_FILE_TO_PATH(                                              $
                  (*ptr).current_file,                                         $
                  path )
      IF status THEN BEGIN
        msg     = ['Problem with routine '+routine+'... exiting...']
        res     = DIALOG_MESSAGE(msg, /ERROR)
        RETURN
      ENDIF

      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_SETREGION_BY_PATH_BLOCKRANGE( path,                      $
      ;                                              start_block,              $
      ;                                              end_block,                $
      ;                                              region )
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
      routine = 'MTK_SETREGION_BY_PATH_BLOCKRANGE'
      status  = MTK_SETREGION_BY_PATH_BLOCKRANGE(                              $
                  path,                                                        $
                  sb,                                                          $
                  eb,                                                          $
                  regioninfo )
      IF status THEN BEGIN
        msg     = ['Problem with routine '+routine+'... exiting...']
        res     = DIALOG_MESSAGE(msg, /ERROR)
        RETURN
      ENDIF

      rgb_exists= INTARR(3)

      WIDGET_CONTROL, (*ptr).red_plane_lbl, GET_VALUE = grid_field_dim
      chars2strip = 'RED PLANE:'
      pos         = STRPOS(grid_field_dim,chars2strip)
      IF pos GE 0 THEN                                                         $
        grid_field_dim = STRTRIM(STRMID(grid_field_dim,pos+STRLEN(chars2strip)),2)
      IF STRPOS(grid_field_dim,'::') GE 0 THEN BEGIN
        sep       = STR_SEP(grid_field_dim,'::')
        gridname  = sep[0]
        fieldname = sep[1]
;ckt,feb2006 print,'red fieldname = ',fieldname
        ;=======================================================================
        ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MT
        ;
        ; MTK CALL: MTK_READDATA( filename,                                    $
        ;                         gridname,                                    $
        ;                         fieldname,                                   $
        ;                         region                                       $
        ;                         databuf,                                     $
        ;                         mapinfo )
        ;
        ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MT
        ;=======================================================================
        routine = 'MTK_READDATA'
;print,'(*ptr).current_file=',(*ptr).current_file
;print,'gridname=',gridname
;print,'fieldname=',fieldname
;help,regioninfo
       status  = MTK_READDATA(                                                 $
                    (*ptr).current_file,                                       $
                    gridname,                                                  $
                    fieldname,                                                 $
                    regioninfo,                                                $
                    rdbuf,                                                     $
                    rmapinfo )
        IF status THEN BEGIN
          msg     = ['Problem with routine '+routine+'... exiting...']
          res     = DIALOG_MESSAGE(msg, /ERROR)
          RETURN
        ENDIF
        rgb_exists[0] = 1
      ENDIF

      WIDGET_CONTROL, (*ptr).green_plane_lbl, GET_VALUE = grid_field_dim
      chars2strip = 'GREEN PLANE:'
      pos         = STRPOS(grid_field_dim,chars2strip)
      IF pos GE 0 THEN                                                         $
        grid_field_dim = STRTRIM(STRMID(grid_field_dim,pos+STRLEN(chars2strip)),2)
      IF STRPOS(grid_field_dim,'::') GE 0 THEN BEGIN
        sep       = STR_SEP(grid_field_dim,'::')
        gridname  = sep[0]
        fieldname = sep[1]
;ckt,feb2006 print,'green fieldname = ',fieldname
        ;=======================================================================
        ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MT
        ;
        ; MTK CALL: MTK_READDATA( filename,                                    $
        ;                         gridname,                                    $
        ;                         fieldname,                                   $
        ;                         region                                       $
        ;                         databuf,                                     $
        ;                         mapinfo )
        ;
        ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MT
        ;=======================================================================
        routine = 'MTK_READDATA'
        status  = MTK_READDATA(                                                $
                    (*ptr).current_file,                                       $
                    gridname,                                                  $
                    fieldname,                                                 $
                    regioninfo,                                                $
                    gdbuf,                                                     $
                    gmapinfo )
        IF status THEN BEGIN
          msg     = ['Problem with routine '+routine+'... exiting...']
          res     = DIALOG_MESSAGE(msg, /ERROR)
          RETURN
        ENDIF
        rgb_exists[1] = 1
      ENDIF

      WIDGET_CONTROL, (*ptr).blue_plane_lbl, GET_VALUE = grid_field_dim
      chars2strip = 'BLUE PLANE:'
      pos         = STRPOS(grid_field_dim,chars2strip)
      IF pos GE 0 THEN                                                         $
        grid_field_dim = STRTRIM(STRMID(grid_field_dim,pos+STRLEN(chars2strip)),2)
      IF STRPOS(grid_field_dim,'::') GE 0 THEN BEGIN
        sep       = STR_SEP(grid_field_dim,'::')
        gridname  = sep[0]
        fieldname = sep[1]
;ckt,feb2006 print,'blue fieldname = ',fieldname
        ;=======================================================================
        ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MT
        ;
        ; MTK CALL: MTK_READDATA( filename,                                    $
        ;                         gridname,                                    $
        ;                         fieldname,                                   $
        ;                         region                                       $
        ;                         databuf,                                     $
        ;                         mapinfo )
        ;
        ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MT
        ;=======================================================================
        routine = 'MTK_READDATA'
        status  = MTK_READDATA(                                                $
                    (*ptr).current_file,                                       $
                    gridname,                                                  $
                    fieldname,                                                 $
                    regioninfo,                                                $
                    bdbuf,                                                     $
                    bmapinfo )
        IF status THEN BEGIN
          msg     = ['Problem with routine '+routine+'... exiting...']
          res     = DIALOG_MESSAGE(msg, /ERROR)
          RETURN
        ENDIF
        rgb_exists[2] = 1
      ENDIF

      idx    = WHERE(rgb_exists GT 0, cnt)
      xdim   = 0L
      ydim   = 0L


;help,rdbuf
;help,gdbuf
;help,bdbuf

      IF cnt GT 1 THEN BEGIN
        nt   = 0B
        FOR p = 0, cnt-1 DO BEGIN
           CASE idx[p] OF
             0: BEGIN & dims  = SIZE(rdbuf,/DIMENSIONS) & nt = nt*rdbuf[0] & END
             1: BEGIN & dims  = SIZE(gdbuf,/DIMENSIONS) & nt = nt*gdbuf[0] & END
             2: BEGIN & dims  = SIZE(bdbuf,/DIMENSIONS) & nt = nt*bdbuf[0] & END
             ELSE:
           ENDCASE

           xdim  = MAX([xdim,dims[0]])
           ydim  = MAX([ydim,dims[1]])

           IF xdim EQ dims[0] AND ydim EQ dims[1] THEN BEGIN
             CASE idx[p] OF
               0: mapinfo2use = rmapinfo
               1: mapinfo2use = gmapinfo
               2: mapinfo2use = bmapinfo
             ENDCASE
           ENDIF
        ENDFOR
        IF SIZE(nt,/TYPE) EQ 1 THEN nt = 0
;;;ckt,feb2006 help,nt
        dbuf     = BYTARR(3,xdim,ydim)*nt
        FOR p = 0, cnt-1 DO BEGIN
           CASE idx[p] OF
             0: dbuf[0,*,*] = CONGRID(rdbuf,xdim,ydim,CUBIC = (-0.5))
             1: dbuf[1,*,*] = CONGRID(gdbuf,xdim,ydim,CUBIC = (-0.5))
             2: dbuf[2,*,*] = CONGRID(bdbuf,xdim,ydim,CUBIC = (-0.5))
             ELSE:
           ENDCASE
        ENDFOR
      ENDIF ELSE BEGIN
        CASE idx[0] OF
             0: BEGIN
               dbuf = rdbuf
               mapinfo2use = rmapinfo
               END
             1: BEGIN
               dbuf = gdbuf
               mapinfo2use = gmapinfo
               END
             2: BEGIN
               dbuf = bdbuf
               mapinfo2use = bmapinfo
               END
        ENDCASE
        IF SIZE(dbuf,/TYPE) EQ 1 THEN dbuf = FIX(dbuf)
        dims  = SIZE(dbuf,/DIMENSIONS)
        xdim  = dims[0]
        ydim  = dims[1]
      ENDELSE

      tmpdims     = SIZE(dbuf,/DIMENSIONS)
;;;ckt,2005nov29      reverse_idx = N_ELEMENTS(tmpdims)

;;;ckt,2005nov29      slide_image_mtk, HIST_EQUAL(BYTSCL(REVERSE(dbuf,reverse_idx))), mapinfo2use

;;;ckt,feb2006 help,dbuf
      slide_image_mtk, dbuf, mapinfo2use

      END
    'EXIT': BEGIN
      WIDGET_CONTROL, event.top, /DESTROY
      END
    'RED_PLANE': BEGIN
      grid_field_dim  = 'RED PLANE:   ' +                                      $
                        (*((*ptr).grid_list_ptr))[(*ptr).grid_list_idx] +      $
                        '::'+                                                  $
                        (*((*ptr).field_list_ptr))[(*ptr).field_list_idx]

      IF (*ptr).dim_list_idx[0] GE 0 THEN                                      $
       grid_field_dim   = grid_field_dim+'['+                                  $
       STRTRIM((*ptr).dim_list_idx[0]+1,2)+']'
      IF (*ptr).dim_list_idx[1] GE 0 THEN                                      $
       grid_field_dim   = grid_field_dim+'['+                                  $
       STRTRIM((*ptr).dim_list_idx[1]+1,2)+']'

      (*ptr).red_plane_field = grid_field_dim
      WIDGET_CONTROL, (*ptr).red_plane_lbl, SET_VALUE = grid_field_dim
      WIDGET_CONTROL, (*ptr).create_btn, /SENSITIVE
      END
    'GREEN_PLANE': BEGIN
      grid_field_dim  = 'GREEN PLANE: ' +                                      $
                        (*((*ptr).grid_list_ptr))[(*ptr).grid_list_idx] +      $
                        '::'+                                                  $
                        (*((*ptr).field_list_ptr))[(*ptr).field_list_idx]

      IF (*ptr).dim_list_idx[0] GE 0 THEN                                      $
       grid_field_dim   = grid_field_dim+'['+                                  $
       STRTRIM((*ptr).dim_list_idx[0]+1,2)+']'
      IF (*ptr).dim_list_idx[1] GE 0 THEN                                      $
       grid_field_dim   = grid_field_dim+'['+                                  $
       STRTRIM((*ptr).dim_list_idx[1]+1,2)+']'

      (*ptr).green_plane_field = grid_field_dim
      WIDGET_CONTROL, (*ptr).green_plane_lbl, SET_VALUE = grid_field_dim
      WIDGET_CONTROL, (*ptr).create_btn, /SENSITIVE
      END
    'BLUE_PLANE': BEGIN
      grid_field_dim  = 'BLUE PLANE:  ' +                                      $
                        (*((*ptr).grid_list_ptr))[(*ptr).grid_list_idx] +      $
                        '::'+                                                  $
                        (*((*ptr).field_list_ptr))[(*ptr).field_list_idx]

      IF (*ptr).dim_list_idx[0] GE 0 THEN                                      $
       grid_field_dim   = grid_field_dim+'['+                                  $
       STRTRIM((*ptr).dim_list_idx[0]+1,2)+']'
      IF (*ptr).dim_list_idx[1] GE 0 THEN                                      $
       grid_field_dim   = grid_field_dim+'['+                                  $
       STRTRIM((*ptr).dim_list_idx[1]+1,2)+']'

      (*ptr).blue_plane_field = grid_field_dim
      WIDGET_CONTROL, (*ptr).blue_plane_lbl, SET_VALUE = grid_field_dim
      WIDGET_CONTROL, (*ptr).create_btn, /SENSITIVE
      END
    'CLEAR_RED': BEGIN
      WIDGET_CONTROL, (*ptr).red_plane_lbl,                                    $
         SET_VALUE = 'RED PLANE:   No Selection'
      (*ptr).red_plane_field = ''
      IF (*ptr).green_plane_field EQ '' AND (*ptr).blue_plane_field EQ '' THEN $
        WIDGET_CONTROL, (*ptr).create_btn, SENSITIVE = 0
      END
    'CLEAR_GREEN': BEGIN
      WIDGET_CONTROL, (*ptr).green_plane_lbl,                                  $
         SET_VALUE = 'GREEN PLANE: No Selection'
      (*ptr).green_plane_field = ''
      IF (*ptr).red_plane_field EQ '' AND (*ptr).blue_plane_field EQ '' THEN   $
        WIDGET_CONTROL, (*ptr).create_btn, SENSITIVE = 0
      END
    'CLEAR_BLUE': BEGIN
      WIDGET_CONTROL, (*ptr).blue_plane_lbl,                                   $
         SET_VALUE = 'BLUE PLANE:  No Selection'
      (*ptr).blue_plane_field = ''
      IF (*ptr).red_plane_field EQ '' AND (*ptr).green_plane_field EQ '' THEN  $
        WIDGET_CONTROL, (*ptr).create_btn, SENSITIVE = 0
      END
    'GRID_LIST': BEGIN
      WIDGET_CONTROL, (*ptr).file_lbl, GET_VALUE = txt
      IF STRTRIM(txt,2) EQ 'FILE: None Selected' THEN RETURN
      (*ptr).grid_list_idx  = event.index
      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_FILE_GRID_TO_FIELDLIST( file,                            $
      ;                                       grid_name,                       $
      ;                                       n_fields,                        $
      ;                                       field_list )
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
      routine = 'MTK_FILE_GRID_TO_FIELDLIST'
      status  = MTK_FILE_GRID_TO_FIELDLIST(                                    $
                  (*ptr).current_file,                                         $
                  (*((*ptr).grid_list_ptr))[(*ptr).grid_list_idx],             $
                  n_fields,                                                    $
                  field_list )
      IF status THEN BEGIN
        msg     = ['Problem with routine '+routine+'... exiting...']
        res     = DIALOG_MESSAGE(msg, /ERROR)
        RETURN
      ENDIF

;;;ckt,2005nov29 print,'field_list 1 = ',field_list


;;;ckt,2005nov29 FIELD_SEPARATOR	= ','
;;;ckt,2005nov29 hdfeosID	= EOS_GD_OPEN( (*ptr).current_file )
;;;ckt,2005nov29 gridID		= EOS_GD_ATTACH( hdfeosID, $
;;;ckt,2005nov29                     (*((*ptr).grid_list_ptr))[(*ptr).grid_list_idx] )
;;;ckt,2005nov29 nField		= EOS_GD_INQFIELDS(gridID, tmp_fList, rank, numberType )
;;;ckt,2005nov29 field_list	= STR_SEP( tmp_fList, FIELD_SEPARATOR )
;;;ckt,2005nov29
;;;ckt,2005nov29 iStat		= EOS_GD_FIELDINFO( 			$
;;;ckt,2005nov29 			gridID,				$
;;;ckt,2005nov29 			field_list[0],			$
;;;ckt,2005nov29 			rank,				$
;;;ckt,2005nov29 			dims,				$
;;;ckt,2005nov29 			nType,				$
;;;ckt,2005nov29 			dimNames )
;;;ckt,2005nov29 dimNames	= STR_SEP( dimNames, FIELD_SEPARATOR )
;;;ckt,2005nov29 dims		= dims[0:N_ELEMENTS(dimNames)-1]
;print,'dims=',dims
;print,'dimNames = ',dimNames
;;;ckt,2005nov29 iStat	= EOS_GD_DETACH( gridID )
;;;ckt,2005nov29 iStat	= EOS_GD_close( hdfeosID )



      PTR_FREE, (*ptr).field_list_ptr
      (*ptr).field_list_ptr    = PTR_NEW(STRTRIM(field_list,2))
      (*ptr).field_list_idx    = 0
      WIDGET_CONTROL, (*ptr).field_list, SET_VALUE=field_list, SET_LIST_SELECT=0

      routine = 'MTK_FILE_GRID_FIELD_TO_DIMLIST'
      status  = MTK_FILE_GRID_FIELD_TO_DIMLIST(                                $
                  (*ptr).current_file,                                         $
                  (*((*ptr).grid_list_ptr))[(*ptr).grid_list_idx],             $
                  (*((*ptr).field_list_ptr))[(*ptr).field_list_idx],           $
                  n_dims,                                                      $
                  dim_list,                                                    $
                  dim_size )
      IF status THEN BEGIN
          n_dims = 0
      ENDIF

      IF n_dims LE 0 THEN BEGIN
       WIDGET_CONTROL, (*ptr).dim_list[0], SET_VALUE=['Not Available']
       WIDGET_CONTROL, (*ptr).dim_list[1], SET_VALUE=['Not Available']
       WIDGET_CONTROL, (*ptr).extra_dim_lbls[0], SET_VALUE = 'EXTRA DIMENSION LIST #1'
       WIDGET_CONTROL, (*ptr).extra_dim_lbls[1], SET_VALUE = 'EXTRA DIMENSION LIST #2'
       (*ptr).dim_list_idx[*] = -1
      ENDIF ELSE BEGIN
       (*ptr).dim_list_idx[*] = -1
       FOR m = 0, n_dims-1 DO BEGIN
        WIDGET_CONTROL,                                                        $
         (*ptr).dim_list[m],                                                   $
         SET_VALUE=STRTRIM(dim_list[m],2)+STRTRIM(LINDGEN(dim_size[m])+1L,2),  $
         SET_LIST_SELECT=0
         WIDGET_CONTROL, (*ptr).extra_dim_lbls[m], SET_VALUE = STRUPCASE(STRTRIM(dim_list[m],2))
         (*ptr).dim_list_idx[m]   = 0
       ENDFOR
       IF n_dims EQ 1 THEN BEGIN
        WIDGET_CONTROL, (*ptr).dim_list[1], SET_VALUE=['Not Available']
        WIDGET_CONTROL, (*ptr).extra_dim_lbls[1], SET_VALUE = 'EXTRA DIMENSION LIST #2'
       ENDIF
;ckt,feb2006 print,'n_dims=',n_dims
;ckt,feb2006 print,'dim_list=',dim_list
;ckt,feb2006 print,'dim_size=',dim_size
      ENDELSE



      END
    'FIELD_LIST': BEGIN
      WIDGET_CONTROL, (*ptr).file_lbl, GET_VALUE = txt
      IF STRTRIM(txt,2) EQ 'FILE: None Selected' THEN RETURN
      (*ptr).field_list_idx  = event.index
;ckt,feb2006 print,'(*ptr).current_file=',(*ptr).current_file
;ckt,feb2006 print,'(*((*ptr).grid_list_ptr))[(*ptr).grid_list_idx]=',(*((*ptr).grid_list_ptr))[(*ptr).grid_list_idx]
;ckt,feb2006 print,'(*((*ptr).field_list_ptr))[(*ptr).field_list_idx]=',(*((*ptr).field_list_ptr))[(*ptr).field_list_idx]
      routine = 'MTK_FILE_GRID_FIELD_TO_DIMLIST'
      status  = MTK_FILE_GRID_FIELD_TO_DIMLIST(                                $
                  (*ptr).current_file,                                         $
                  (*((*ptr).grid_list_ptr))[(*ptr).grid_list_idx],             $
                  (*((*ptr).field_list_ptr))[(*ptr).field_list_idx],           $
                  n_dims,                                                      $
                  dim_list,                                                    $
                  dim_size )
      IF status THEN BEGIN
          n_dims = 0
      ENDIF

      IF n_dims LE 0 THEN BEGIN
       WIDGET_CONTROL, (*ptr).dim_list[0], SET_VALUE=['Not Available']
       WIDGET_CONTROL, (*ptr).dim_list[1], SET_VALUE=['Not Available']
       WIDGET_CONTROL, (*ptr).extra_dim_lbls[0], SET_VALUE = 'EXTRA DIMENSION LIST #1'
       WIDGET_CONTROL, (*ptr).extra_dim_lbls[1], SET_VALUE = 'EXTRA DIMENSION LIST #2'
       (*ptr).dim_list_idx[*] = -1
      ENDIF ELSE BEGIN
       (*ptr).dim_list_idx[*] = -1
       FOR m = 0, n_dims-1 DO BEGIN
        WIDGET_CONTROL,                                                        $
         (*ptr).dim_list[m],                                                   $
         SET_VALUE=STRTRIM(dim_list[m],2)+STRTRIM(LINDGEN(dim_size[m])+1L,2),  $
         SET_LIST_SELECT=0
         WIDGET_CONTROL, (*ptr).extra_dim_lbls[m], SET_VALUE = STRUPCASE(STRTRIM(dim_list[m],2))
         (*ptr).dim_list_idx[m]   = 0
       ENDFOR
       IF n_dims EQ 1 THEN BEGIN
        WIDGET_CONTROL, (*ptr).dim_list[1], SET_VALUE=['Not Available']
        WIDGET_CONTROL, (*ptr).extra_dim_lbls[1], SET_VALUE = 'EXTRA DIMENSION LIST #2'
       ENDIF
;ckt,feb2006 print,'n_dims=',n_dims
;ckt,feb2006 print,'dim_list=',dim_list
;ckt,feb2006 print,'dim_size=',dim_size
      ENDELSE

      END
    'DIM_LIST1': BEGIN
      IF (*ptr).dim_list_idx[0] GE 0 THEN (*ptr).dim_list_idx[0] = event.index
      END
    'DIM_LIST2': BEGIN
      IF (*ptr).dim_list_idx[1] GE 0 THEN (*ptr).dim_list_idx[1] = event.index
      END
    'START_BLOCK': BEGIN
      WIDGET_CONTROL, (*ptr).sb_cwf, GET_VALUE = sb
      (*ptr).start_block = sb[0]
      END
    'END_BLOCK': BEGIN
      WIDGET_CONTROL, (*ptr).eb_cwf, GET_VALUE = eb
      (*ptr).end_block = eb[0]
      END
    'SELECT_FILE': BEGIN
      use_gui   = 1
      file      = get_file(use_gui)
      IF STRTRIM(file,2) EQ '' THEN RETURN
      WIDGET_CONTROL, (*ptr).file_lbl, SET_VALUE = 'FILE: '+STRTRIM(file,2)
      (*ptr).current_file  = STRTRIM(file,2)
      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_FILE_TO_GRIDLIST( file,                                  $
      ;                                 n_grids,                               $
      ;                                 grid_list )
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
      routine = 'MTK_FILE_TO_GRIDLIST'
      status  = MTK_FILE_TO_GRIDLIST(                                          $
                  (*ptr).current_file,                                         $
                  n_grids,                                                     $
                  grid_list )
      IF status THEN BEGIN
        msg     = ['Problem with routine '+routine+'... exiting...']
        res     = DIALOG_MESSAGE(msg, /ERROR)
        RETURN
      ENDIF

;;;ckt,2005nov29 print,'grid_list = ',grid_list
;;;ckt,2005nov29 help,grid_list

;;;ckt,2005nov29 nGrid = EOS_GD_INQGRID( (*ptr).current_file, grid_list, LENGTH = strBufSz )
;;;ckt,2005nov29 grid_list = STRTRIM(STR_SEP(grid_list,','),2)



      PTR_FREE, (*ptr).grid_list_ptr
      (*ptr).grid_list_ptr    = PTR_NEW(STRTRIM(grid_list,2))
      (*ptr).grid_list_idx    = 0
      WIDGET_CONTROL, (*ptr).grid_list, SET_VALUE=grid_list, SET_LIST_SELECT=0
      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_FILE_GRID_TO_FIELDLIST( file,                            $
      ;                                       grid_name,                       $
      ;                                       n_fields,                        $
      ;                                       field_list )
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
      routine = 'MTK_FILE_GRID_TO_FIELDLIST'
;print,'grid_list[0]=',grid_list[0]
      status  = MTK_FILE_GRID_TO_FIELDLIST(                                    $
                  (*ptr).current_file,                                         $
                  grid_list[0],                                                $
                  n_fields,                                                    $
                  field_list )
      IF status THEN BEGIN
        msg     = ['Problem with routine '+routine+'... exiting...']
        res     = DIALOG_MESSAGE(msg, /ERROR)
        RETURN
      ENDIF

;;;ckt,2005nov29 print,'field_list 2 = ',field_list
;;;ckt,2005nov29 help,field_list

;;;ckt,2005nov29 FIELD_SEPARATOR	= ','
;;;ckt,2005nov29 hdfeosID	= EOS_GD_OPEN( (*ptr).current_file )
;;;ckt,2005nov29 gridID		= EOS_GD_ATTACH( hdfeosID, grid_list[0] )
;;;ckt,2005nov29 nField		= EOS_GD_INQFIELDS(gridID, tmp_fList, rank, numberType )
;;;ckt,2005nov29 field_list	= STR_SEP( tmp_fList, FIELD_SEPARATOR )

;;;ckt,2005nov29 iStat		= EOS_GD_FIELDINFO( 			$
;;;ckt,2005nov29 			gridID,				$
;;;ckt,2005nov29 			field_list[0],			$
;;;ckt,2005nov29 			rank,				$
;;;ckt,2005nov29 			dims,				$
;;;ckt,2005nov29 			nType,				$
;;;ckt,2005nov29 			dimNames )
;;;ckt,2005nov29 dimNames	= STR_SEP( dimNames, FIELD_SEPARATOR )
;;;ckt,2005nov29 dims		= dims[0:N_ELEMENTS(dimNames)-1]
;print,'dims=',dims
;print,'dimNames = ',dimNames
;;;ckt,2005nov29 iStat	= EOS_GD_DETACH( gridID )
;;;ckt,2005nov29 iStat	= EOS_GD_close( hdfeosID )




      PTR_FREE, (*ptr).field_list_ptr
      (*ptr).field_list_ptr    = PTR_NEW(STRTRIM(field_list,2))
      (*ptr).field_list_idx    = 0
      WIDGET_CONTROL, (*ptr).field_list, SET_VALUE=field_list, SET_LIST_SELECT=0

      routine = 'MTK_FILE_GRID_FIELD_TO_DIMLIST'
      status  = MTK_FILE_GRID_FIELD_TO_DIMLIST(                                $
                  (*ptr).current_file,                                         $
                  (*((*ptr).grid_list_ptr))[(*ptr).grid_list_idx],             $
                  (*((*ptr).field_list_ptr))[(*ptr).field_list_idx],           $
                  n_dims,                                                      $
                  dim_list,                                                    $
                  dim_size )
      IF status THEN BEGIN
          n_dims = 0
      ENDIF

      IF n_dims LE 0 THEN BEGIN
       WIDGET_CONTROL, (*ptr).dim_list[0], SET_VALUE=['Not Available']
       WIDGET_CONTROL, (*ptr).dim_list[1], SET_VALUE=['Not Available']
       WIDGET_CONTROL, (*ptr).extra_dim_lbls[0], SET_VALUE = 'EXTRA DIMENSION LIST #1'
       WIDGET_CONTROL, (*ptr).extra_dim_lbls[1], SET_VALUE = 'EXTRA DIMENSION LIST #2'
       (*ptr).dim_list_idx[*] = -1
      ENDIF ELSE BEGIN
       (*ptr).dim_list_idx[*] = -1
       FOR m = 0, n_dims-1 DO BEGIN
        WIDGET_CONTROL,                                                        $
         (*ptr).dim_list[m],                                                   $
         SET_VALUE=STRTRIM(dim_list[m],2)+STRTRIM(LINDGEN(dim_size[m])+1L,2),  $
         SET_LIST_SELECT=0
         WIDGET_CONTROL, (*ptr).extra_dim_lbls[m], SET_VALUE = STRUPCASE(STRTRIM(dim_list[m],2))
         (*ptr).dim_list_idx[m]   = 0
       ENDFOR
       IF n_dims EQ 1 THEN BEGIN
        WIDGET_CONTROL, (*ptr).dim_list[1], SET_VALUE=['Not Available']
        WIDGET_CONTROL, (*ptr).extra_dim_lbls[1], SET_VALUE = 'EXTRA DIMENSION LIST #2'
       ENDIF
;ckt,feb2006 print,'n_dims=',n_dims
;ckt,feb2006 print,'dim_list=',dim_list
;ckt,feb2006 print,'dim_size=',dim_size
      ENDELSE


      ;=========================================================================
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;
      ; MTK CALL: MTK_FILE_TO_BLOCKRANGE( file,                                $
      ;                                   start_block,                         $
      ;                                   end_block )
      ;
      ;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-
      ;=========================================================================
      routine = 'MTK_FILE_TO_BLOCKRANGE'
      status  = MTK_FILE_TO_BLOCKRANGE(                                        $
                  (*ptr).current_file,                                         $
                  start_block,                                                 $
                  end_block )
      IF status THEN BEGIN
        msg     = ['Problem with routine '+routine+'... exiting...']
        res     = DIALOG_MESSAGE(msg, /ERROR)
        RETURN
      ENDIF
      WIDGET_CONTROL, (*ptr).block_lbl,                                        $
        SET_VALUE = 'BLOCK RANGE: '+STRTRIM(start_block,2)+' through '+        $
        STRTRIM(end_block,2)
      bv        = CEIL((end_block-start_block+1)/2+start_block)
      WIDGET_CONTROL, (*ptr).sb_cwf, SET_VALUE = bv
      WIDGET_CONTROL, (*ptr).eb_cwf, SET_VALUE = bv
      (*ptr).start_block    = bv
      (*ptr).end_block      = bv

      WIDGET_CONTROL, (*ptr).select_rgb[0], /SENSITIVE
      WIDGET_CONTROL, (*ptr).select_rgb[1], /SENSITIVE
      WIDGET_CONTROL, (*ptr).select_rgb[2], /SENSITIVE

      WIDGET_CONTROL, (*ptr).red_plane_lbl,                                    $
         SET_VALUE = 'RED PLANE:   No Selection'
      (*ptr).red_plane_field = ''
      IF (*ptr).green_plane_field EQ '' AND (*ptr).blue_plane_field EQ '' THEN $
        WIDGET_CONTROL, (*ptr).create_btn, SENSITIVE = 0

      WIDGET_CONTROL, (*ptr).green_plane_lbl,                                  $
         SET_VALUE = 'GREEN PLANE: No Selection'
      (*ptr).green_plane_field = ''
      IF (*ptr).red_plane_field EQ '' AND (*ptr).blue_plane_field EQ '' THEN   $
        WIDGET_CONTROL, (*ptr).create_btn, SENSITIVE = 0

      WIDGET_CONTROL, (*ptr).blue_plane_lbl,                                   $
         SET_VALUE = 'BLUE PLANE:  No Selection'
      (*ptr).blue_plane_field = ''
      IF (*ptr).red_plane_field EQ '' AND (*ptr).green_plane_field EQ '' THEN  $
        WIDGET_CONTROL, (*ptr).create_btn, SENSITIVE = 0

      END
    ELSE:
  ENDCASE
END
; example3_eh

;1234567890123456789012345678901234567890123456789012345678901234567890123456789
;===============================================================================
; example3
;===============================================================================
PRO example3
  tlb      = WIDGET_BASE(                                                      $
                /COLUMN,                                                       $
                Y_SCROLL_SIZE=(GET_SCREEN_SIZE())[1]*0.85,                     $
                TITLE = 'example3',                                            $
                /BASE_ALIGN_LEFT,                                              $
                EVENT_PRO = 'example3_eh' )
  sb1      = WIDGET_BASE(                                                      $
                tlb,                                                           $
                /ROW,                                                          $
                /ALIGN_CENTER,                                                 $
                /BASE_ALIGN_CENTER )
  lbl1     = WIDGET_LABEL(                                                     $
                sb1,                                                           $
                VALUE = 'FILE: None Selected',                                 $
                /DYNAMIC_RESIZE )
  btn1     = WIDGET_BUTTON(                                                    $
                sb1,                                                           $
                VALUE = 'Select...',                                           $
                UVALUE = 'select_file' )
  sb2      = WIDGET_BASE(                                                      $
                tlb,                                                           $
                /ROW,                                                          $
                /ALIGN_CENTER,                                                 $
                /BASE_ALIGN_CENTER )
  lbl2     = WIDGET_LABEL(                                                     $
                sb2,                                                           $
                VALUE = 'BLOCK RANGE: Not Available',                          $
                /DYNAMIC_RESIZE )
  sb3      = WIDGET_BASE(                                                      $
                tlb,                                                           $
                /COLUMN,                                                       $
                /FRAME,                                                        $
                /ALIGN_CENTER,                                                 $
                /BASE_ALIGN_CENTER )
  lbl3     = WIDGET_LABEL(                                                     $
                sb3,                                                           $
                VALUE = 'GRID LIST',                                           $
                /DYNAMIC_RESIZE )
  list3    = WIDGET_LIST(                                                      $
                sb3,                                                           $
                VALUE = ['Not Available'],                                     $
                UVALUE = 'grid_list',                                          $
                SCR_XSIZE = 600,                                               $
                SCR_YSIZE = 100 )
  sb4      = WIDGET_BASE(                                                      $
                tlb,                                                           $
                /COLUMN,                                                       $
                /FRAME,                                                        $
                /ALIGN_CENTER,                                                 $
                /BASE_ALIGN_CENTER )
  lbl4     = WIDGET_LABEL(                                                     $
                sb4,                                                           $
                VALUE = 'FIELD LIST',                                          $
                /DYNAMIC_RESIZE )
  list4    = WIDGET_LIST(                                                      $
                sb4,                                                           $
                VALUE = ['Not Available'],                                     $
                UVALUE = 'field_list',                                         $
                SCR_XSIZE = 600,                                               $
                SCR_YSIZE = 100 )
  sb5      = WIDGET_BASE(                                                      $
                tlb,                                                           $
                /COLUMN,                                                       $
                /FRAME,                                                        $
                /ALIGN_CENTER,                                                 $
                /BASE_ALIGN_CENTER )
  lbl5a     = WIDGET_LABEL(                                                    $
                sb5,                                                           $
                VALUE = 'EXTRA DIMENSION LIST #1',                             $
                /DYNAMIC_RESIZE )
  list5a    = WIDGET_LIST(                                                     $
                sb5,                                                           $
                VALUE = ['Not Available'],                                     $
                UVALUE = 'dim_list1',                                          $
                SCR_XSIZE = 600,                                               $
                SCR_YSIZE = 100 )
  lbl5b     = WIDGET_LABEL(                                                    $
                sb5,                                                           $
                VALUE = 'EXTRA DIMENSION LIST #2',                             $
                /DYNAMIC_RESIZE )
  list5b    = WIDGET_LIST(                                                     $
                sb5,                                                           $
                VALUE = ['Not Available'],                                     $
                UVALUE = 'dim_list2',                                          $
                SCR_XSIZE = 600,                                               $
                SCR_YSIZE = 100 )
  sb6      = WIDGET_BASE(                                                      $
                tlb,                                                           $
                /ROW,                                                          $
                /FRAME,                                                        $
                /ALIGN_CENTER,                                                 $
                /BASE_ALIGN_CENTER )
  cwf6     = CW_FIELD(                                                         $
                sb6,                                                           $
                TITLE = 'Starting Block:',                                     $
                VALUE = 1,                                                     $
                /INTEGER,                                                      $
                UVALUE = 'start_block' )
  cwf7     = CW_FIELD(                                                         $
                sb6,                                                           $
                TITLE = 'Ending Block:',                                       $
                VALUE = 180,                                                   $
                /INTEGER,                                                      $
                UVALUE = 'end_block' )
  sb7      = WIDGET_BASE(                                                      $
                tlb,                                                           $
                /ROW,                                                          $
                /FRAME,                                                        $
                /ALIGN_CENTER,                                                 $
                /BASE_ALIGN_CENTER )
  btn7     = WIDGET_BUTTON(                                                    $
                sb7,                                                           $
                VALUE = 'Select For Red Plane',                                $
                SENSITIVE = 0,                                                 $
                UVALUE = 'red_plane' )
  btn8     = WIDGET_BUTTON(                                                    $
                sb7,                                                           $
                VALUE = 'Select For Green Plane',                              $
                SENSITIVE = 0,                                                 $
                UVALUE = 'green_plane' )
  btn9     = WIDGET_BUTTON(                                                    $
                sb7,                                                           $
                VALUE = 'Select For Blue Plane',                               $
                SENSITIVE = 0,                                                 $
                UVALUE = 'blue_plane' )
  sb8      = WIDGET_BASE(                                                      $
                tlb,                                                           $
                /COLUMN,                                                       $
                /FRAME,                                                        $
                /ALIGN_CENTER,                                                 $
                /BASE_ALIGN_CENTER )
  sb8a     = WIDGET_BASE(                                                      $
                sb8,                                                           $
                /ROW,                                                          $
                /ALIGN_LEFT )
  lbl8a    = WIDGET_LABEL(                                                     $
                sb8a,                                                          $
                VALUE = 'RED PLANE:   No Selection',                           $
                /DYNAMIC_RESIZE )
  btn8a    = WIDGET_BUTTON(                                                    $
                sb8a,                                                          $
                VALUE = ' Clear Red Plane ',                                   $
                UVALUE = 'clear_red' )
  sb8b     = WIDGET_BASE(                                                      $
                sb8,                                                           $
                /ROW,                                                          $
                /ALIGN_LEFT )
  lbl8b    = WIDGET_LABEL(                                                     $
                sb8b,                                                          $
                VALUE = 'GREEN PLANE: No Selection',                           $
                /DYNAMIC_RESIZE )
  btn8b    = WIDGET_BUTTON(                                                    $
                sb8b,                                                          $
                VALUE = 'Clear Green Plane',                                   $
                UVALUE = 'clear_green' )
  sb8c     = WIDGET_BASE(                                                      $
                sb8,                                                           $
                /ROW,                                                          $
                /ALIGN_LEFT )
  lbl8c    = WIDGET_LABEL(                                                     $
                sb8c,                                                          $
                VALUE = 'BLUE PLANE:  No Selection',                           $
                /DYNAMIC_RESIZE )
  btn8c    = WIDGET_BUTTON(                                                    $
                sb8c,                                                          $
                VALUE = 'Clear Blue Plane ',                                   $
                UVALUE = 'clear_blue' )
  btn10    = WIDGET_BUTTON(                                                    $
                tlb,                                                           $
                VALUE = 'C R E A T E   V I E W E R',                           $
                /ALIGN_CENTER,                                                 $
                SENSITIVE = 0,                                                 $
                UVALUE = 'create_viewer' )
  btn11    = WIDGET_BUTTON(                                                    $
                tlb,                                                           $
                VALUE = 'E X I T',                                             $
                /ALIGN_CENTER,                                                 $
                UVALUE = 'exit' )
  WIDGET_CONTROL, tlb, /REALIZE

  ptr      = PTR_NEW( {                                                        $
                        file_lbl         : lbl1,                               $
                        current_file     : '',                                 $
                        grid_list        : list3,                              $
                        field_list       : list4,                              $
                        dim_list         : [list5a,list5b],                    $
                        block_lbl        : lbl2,                               $
                        extra_dim_lbls   : [lbl5a,lbl5b],                      $
                        sb_cwf           : cwf6,                               $
                        eb_cwf           : cwf7,                               $
                        grid_list_ptr    : PTR_NEW(),                          $
                        field_list_ptr   : PTR_NEW(),                          $
                        dim_list_ptr     : PTR_NEW(),                          $
                        dim_size_ptr     : PTR_NEW(),                          $
                        grid_list_idx    : 0,                                  $
                        field_list_idx   : 0,                                  $
                        dim_list_idx     : [-1,-1],                            $
                        start_block      : 1,                                  $
                        end_block        : 180,                                $
                        select_rgb       : [btn7,btn8,btn9],                   $
                        create_btn       : btn10,                              $
                        red_plane_lbl    : lbl8a,                              $
                        green_plane_lbl  : lbl8b,                              $
                        blue_plane_lbl   : lbl8c,                              $
                        red_plane_field  : '',                                 $
                        green_plane_field: '',                                 $
                        blue_plane_field : '' } )
 WIDGET_CONTROL, tlb, SET_UVALUE = ptr

  XMANAGER, 'example3', tlb, EVENT_HANDLER = 'example3_eh'

  PTR_FREE, ptr
END
; example3

