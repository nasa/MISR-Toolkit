@utilities.pro

;1234567890123456789012345678901234567890123456789012345678901234567890123456789
;===============================================================================
; do_mapping
;===============================================================================
PRO do_mapping, path, block_start, block_end, USE_GUI = iuse_gui
	use_gui	= 0
	IF KEYWORD_SET(iuse_gui) THEN use_gui = 1
	msg	= [								$
		'With the routine MTK_SETREGION_BY_PATH_BLOCKRANGE, a "region"',$
		'can be established.  An associated structure for the region',	$
		'contains geographical information that can be used',		$
		'to determine limits and location.  In this example, the',	$
		'MTK_SETREGION_BY_PATH_BLOCKRANGE routine is first called',	$
		'to establish the extent of coverage of the data within the ',	$
		'file.  The starting block and ending block are used as input ',$
		'in order to return an encompassing region.',			$
		'',								$
		'Syntax: status =',						$
		'     MTK_SETREGION_BY_PATH_BLOCKRANGE(path, bs, be, region)',	$
		'     where:',							$
		'          path    = path number',				$
		'          bs      = starting block number for region to be'+	$
		' defined',							$
		'          be      = ending block number for region to be'+	$
		' defined',							$
		'          region  = returned structure containing geographic'+	$
		' extent information',				$
		'',								$
		'region = {',							$
		'              geo_ctr_lat : center latitude of region,',	$
		'              geo_ctr_lon : center longitude of region,',	$
		'              hextent_xlat: half extent of latitudinal range',	$
		'              hextent_ylon: half extent of longitudinal range',$
		'',								$
		'STARTING BLOCK = '+STRTRIM(block_start,2),			$
		'ENDING BLOCK   = '+STRTRIM(block_end,2),			$
		'' ]
	display_msg, msg, use_gui
	;=======================================================================
	;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
	;
	; MTK CALL: MTK_SETREGION_BY_PATH_BLOCKRANGE( path,                    $
        ;                                             block_start,             $
        ;                                             block_end,               $
        ;                                             region)
	;
	;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
	;=======================================================================
	status	= MTK_SETREGION_BY_PATH_BLOCKRANGE(				$
		path, block_start, block_end, region)
	IF status NE 0 THEN							$
		MESSAGE, 'Problem with function '+				$
			'MTK_SETREGION_BY_PATH_BLOCKRANGE'
	clat	= region.geo_ctr_lat
	clon	= region.geo_ctr_lon

	status	= MTK_LATLON_TO_SOMXY(						$
		path, region.geo_ctr_lat, region.geo_ctr_lon,			$
		center_som_x, center_som_y )
	IF status NE 0 THEN							$
		MESSAGE, 'Problem with function '+				$
			'MTK_LATLON_TO_SOMXY'
	corner1_x= center_som_x - region.hextent_xlat
	corner1_y= center_som_y - region.hextent_ylon
	corner2_x= corner1_x
	corner2_y= center_som_y + region.hextent_ylon
	corner3_x= center_som_x + region.hextent_xlat
	corner3_y= corner2_y
	corner4_x= corner3_x
	corner4_y= corner1_y

	msg	= [								$
		'With the routine MTK_LATLON_TO_SOMXY, the center of the',	$
		'defined region can be converted from lat/lon to SOM X/Y ',	$
		"which is needed for calculating the region's four",		$
		'cornerpoints in SOM, as follows:',				$
		'',								$
		'      corner1_x= center_som_x - region.hextent_xlat',		$
		'      corner1_y= center_som_y - region.hextent_ylon',		$
		'      corner2_x= corner1_x',					$
		'      corner2_y= center_som_y + region.hextent_ylon',		$
		'      corner3_x= center_som_x + region.hextent_xlat',		$
		'      corner3_y= corner2_y',					$
		'      corner4_x= corner3_x',					$
		'      corner4_y= corner1_y',					$
		'',								$
		'Syntax: status =',						$
		'     MTK_LATLON_TO_SOMXY(path, lat, lon, som_x, som_y)',	$
		'     where:',							$
		'          path    = path number',				$
		'          lat     = latitude of point to convert to SOM',	$
		'          lon     = longitude of point to convert to SOM',	$
		'          som_x   = latitude converted to SOM X',		$
		'          som_y   = longitude converted to SOM Y',		$
		'',								$
		'SOM X = '+STRTRIM(center_som_x,2),				$
		'SOM Y = '+STRTRIM(center_som_y,2),				$
		'',								$
		'Corner 1 X = '+STRTRIM(corner1_x,2),				$
		'Corner 1 Y = '+STRTRIM(corner1_y,2),				$
		'',								$
		'Corner 2 X = '+STRTRIM(corner2_x,2),				$
		'Corner 2 Y = '+STRTRIM(corner2_y,2),				$
		'',								$
		'Corner 3 X = '+STRTRIM(corner3_x,2),				$
		'Corner 3 Y = '+STRTRIM(corner3_y,2),				$
		'',								$
		'Corner 4 X = '+STRTRIM(corner4_x,2),				$
		'Corner 4 Y = '+STRTRIM(corner4_y,2),				$
		'' ]

	display_msg, msg, use_gui
	;=======================================================================
	;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
	;
	; MTK CALL: MTK_SOMXY_TO_LATLON(path, somx, somy, lat, lon)
	;
	;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
	;=======================================================================
	status	= MTK_SOMXY_TO_LATLON(						$
		path, corner1_x, corner1_y, corner1_lat, corner1_lon )
	IF status NE 0 THEN							$
		MESSAGE, 'Problem with function '+				$
			'MTK_SOMXY_TO_LATLON'
	;=======================================================================
	;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
	;
	; MTK CALL: MTK_SOMXY_TO_LATLON(path, somx, somy, lat, lon)
	;
	;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
	;=======================================================================
	status	= MTK_SOMXY_TO_LATLON(						$
		path, corner2_x, corner2_y, corner2_lat, corner2_lon )
	IF status NE 0 THEN							$
		MESSAGE, 'Problem with function '+				$
			'MTK_SOMXY_TO_LATLON'
	;=======================================================================
	;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
	;
	; MTK CALL: MTK_SOMXY_TO_LATLON(path, somx, somy, lat, lon)
	;
	;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
	;=======================================================================
	status	= MTK_SOMXY_TO_LATLON(						$
		path, corner3_x, corner3_y, corner3_lat, corner3_lon )
	IF status NE 0 THEN							$
		MESSAGE, 'Problem with function '+				$
			'MTK_SOMXY_TO_LATLON'
	;=======================================================================
	;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
	;
	; MTK CALL: MTK_SOMXY_TO_LATLON(path, somx, somy, lat, lon)
	;
	;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
	;=======================================================================
	status	= MTK_SOMXY_TO_LATLON(						$
		path, corner4_x, corner4_y, corner4_lat, corner4_lon )
	IF status NE 0 THEN							$
		MESSAGE, 'Problem with function '+				$
			'MTK_SOMXY_TO_LATLON'
	minlat	= MIN([corner1_lat,corner2_lat,corner3_lat,corner4_lat],MAX=maxlat)
	minlon	= MIN([corner1_lon,corner2_lon,corner3_lon,corner4_lon],MAX=maxlon)
	winsize	= ROUND(GET_SCREEN_SIZE()*0.8)
	lat_size= winsize[1]
	lon_size= winsize[0]

	WINDOW, /FREE, XSIZE = lon_size, YSIZE = lat_size
	win	= !D.WINDOW

	MAP_SET,								$
		0.0,								$
		clon,								$
		0.0,								$
		/NOBORDER,							$
		XMARGIN = 0.0,							$
		YMARGIN = 0.0,							$
		/CONTINENTS,							$
		LIMIT = [minlat,minlon,maxlat,maxlon],				$
		/ISOTROPIC
	msg	= [								$
		'Again using MTK_SETREGION_BY_PATH_BLOCKRANGE and',		$
		'MTK_SOMXY_TO_LATLON, lat/lon cornerpoints for each individual',$
		'block in the file can be calculated for the purposes of',	$
		'plotting.  MTK_SETREGION_BY_PATH_BLOCKRANGE is called in a',	$
		'loop which is executed for all block numbers.  Each time the',	$
		'routine is called, the starting block and ending block values',$
		'are both the value of the loop counter.  This establishes a',	$
		'bounding region for each block, from which the corner lat/lon',$
		'values can be derived, as above',				$
		'']
	display_msg, msg, use_gui
	FOR i = block_start, block_end DO BEGIN
		;===============================================================
		; MTK CALL: MTK_SETREGION_BY_PATH_BLOCKRANGE(path, block_start,
		;           block_end, region)
		;===============================================================
		;===============================================================
		;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---
		;
		; MTK CALL: MTK_SETREGION_BY_PATH_BLOCKRANGE( path,             $
		;                                             block_start,      $
		;                                             block_end,        $
		;                                             region)
		;
		;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---
		;===============================================================
		status	= MTK_SETREGION_BY_PATH_BLOCKRANGE(			$
			path, i, i, region)
		IF status NE 0 THEN						$
			MESSAGE, 'Problem with function '+			$
				'MTK_SETREGION_BY_PATH_BLOCKRANGE'
		;===============================================================
		;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---
		;
		; MTK CALL: MTK_SOMXY_TO_LATLON(path, somx, somy, lat, lon)
		;
		;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---
		;===============================================================
		status	= MTK_LATLON_TO_SOMXY(					$
			path, region.geo_ctr_lat, region.geo_ctr_lon,		$
			center_som_x, center_som_y )
		IF status NE 0 THEN						$
			MESSAGE, 'Problem with function '+			$
				'MTK_LATLON_TO_SOMXY'
		corner1_x= center_som_x - region.hextent_xlat
		corner1_y= center_som_y - region.hextent_ylon
		corner2_x= corner1_x
		corner2_y= center_som_y + region.hextent_ylon
		corner3_x= center_som_x + region.hextent_xlat
		corner3_y= corner2_y
		corner4_x= corner3_x
		corner4_y= corner1_y

		;===============================================================
		;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---
		;
		; MTK CALL: MTK_SOMXY_TO_LATLON(path, somx, somy, lat, lon)
		;
		;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---
		;===============================================================
		status	= MTK_SOMXY_TO_LATLON(					$
			path, corner1_x, corner1_y, corner1_lat, corner1_lon )
		IF status NE 0 THEN						$
			MESSAGE, 'Problem with function '+			$
				'MTK_SOMXY_TO_LATLON'
		;===============================================================
		;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---
		;
		; MTK CALL: MTK_SOMXY_TO_LATLON(path, somx, somy, lat, lon)
		;
		;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---
		;===============================================================
		status	= MTK_SOMXY_TO_LATLON(					$
			path, corner2_x, corner2_y, corner2_lat, corner2_lon )
		IF status NE 0 THEN						$
			MESSAGE, 'Problem with function '+			$
				'MTK_SOMXY_TO_LATLON'
		;===============================================================
		;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---
		;
		; MTK CALL: MTK_SOMXY_TO_LATLON(path, somx, somy, lat, lon)
		;
		;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---
		;===============================================================
		status	= MTK_SOMXY_TO_LATLON(					$
			path, corner3_x, corner3_y, corner3_lat, corner3_lon )
		IF status NE 0 THEN						$
			MESSAGE, 'Problem with function '+			$
				'MTK_SOMXY_TO_LATLON'
		;===============================================================
		;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---
		;
		; MTK CALL: MTK_SOMXY_TO_LATLON(path, somx, somy, lat, lon)
		;
		;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---
		;===============================================================
		status	= MTK_SOMXY_TO_LATLON(					$
			path, corner4_x, corner4_y, corner4_lat, corner4_lon )
		IF status NE 0 THEN						$
			MESSAGE, 'Problem with function '+			$
				'MTK_SOMXY_TO_LATLON'
			
		PLOTS,								$
			[corner1_lon,						$
			 corner2_lon,						$
			 corner3_lon,						$
			 corner4_lon,						$
			 corner1_lon],						$
			[corner1_lat,						$
			 corner2_lat,						$
			 corner3_lat,						$
			 corner4_lat,						$
			 corner1_lat],						$
			/DATA,							$
			COLOR = convert_pseudocolor2truecolor(255B,0B,0B)

	ENDFOR
END
; do_mapping

;1234567890123456789012345678901234567890123456789012345678901234567890123456789
;===============================================================================
; example1
;===============================================================================
PRO example1, NO_GUI = no_gui
	
	use_gui = 1
	IF KEYWORD_SET(no_gui) AND NOT vm_used() THEN use_gui = 0
	
	msg	= [								$
		'This program demonstrates the use of some of the MTK',		$
		'coordinate conversion routines.  The first step will be to',	$
		'select a data file in the Mtk_testdata'+PATH_SEP()+'in'+	$
		PATH_SEP() + ' directory' ]
		
	display_msg, msg, use_gui
   
	;=======================================================================
	; GET FILE
	;=======================================================================
	file	= get_file(use_gui)
	IF STRTRIM(file,2) EQ '' THEN RETURN
	IF NOT use_gui THEN BEGIN
   		PRINT, ''
   		PRINT, 'File: '+file
   		PRINT, ''
	ENDIF
	
	;=======================================================================
	;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
	;
	; MTK CALL: MTK_FILE_TO_PATH(file, path)
	;
	;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
	;=======================================================================
	status	= MTK_FILE_TO_PATH(file, path)
	IF status NE 0 THEN MESSAGE, 'Problem with function MTK_FILE_TO_PATH'
	
	msg	= [								$
		'Using the routine MTK_FILE_TO_PATH, the path corresponding',	$
		'to the file can be determined.',				$
		'',								$
		'Syntax: status = MTK_FILE_TO_PATH(file, path)',		$
		'     where:',							$
		'          file = fully-qualified MISR data file name',		$
		'          path = variable in which the path is returned',	$
		'',								$
		'PATH = '+STRTRIM(path,2),					$
		'' ]
	display_msg, msg, use_gui
			
	;=======================================================================
	;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
	;
	; MTK CALL: MTK_FILE_TO_BLOCKRANGE(file, block_start, block_end)
	;
	;---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK---MTK-----
	;=======================================================================
	status	= MTK_FILE_TO_BLOCKRANGE(file, block_start, block_end)
	IF status NE 0 THEN							$
		MESSAGE, 'Problem with function MTK_FILE_TO_BLOCKRANGE'
	
	msg	= [								$
		'Using the routine MTK_FILE_TO_BLOCKRANGE, the range of valid',	$
		'blocks within the file can be determined.',			$
		'',								$
		'Syntax: status = MTK_FILE_TO_BLOCKRANGE(file, bs, be)',	$
		'     where:',							$
		'          file = fully-qualified MISR data file name',		$
		'          bs   = variable in which the starting block number'+	$
		' is returned',					$
		'          be   = variable in which the ending block number'+	$
		' is returned',					$
		'',								$
		'STARTING BLOCK = '+STRTRIM(block_start,2),			$
		'ENDING BLOCK   = '+STRTRIM(block_end,2),			$
		'' ]
	
	display_msg, msg, use_gui
		
	do_mapping,								$
		path,								$
		block_start,							$
		block_end,							$
		USE_GUI = use_gui
		
	msg	= [								$
		'End of example 1' ]
	display_msg, msg, use_gui
		
	IF !D.WINDOW GE 0 THEN WDELETE, !D.WINDOW
END
; example1
