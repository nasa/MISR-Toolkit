!
!Copyright (C) 1996 Hughes and Applied Research Corporation
!
!Permission to use, modify, and distribute this software and its documentation 
!for any purpose without fee is hereby granted, provided that the above 
!copyright notice appear in all copies and that both that copyright notice and 
!this permission notice appear in supporting documentation.
!

! Working Buffer Sizes
      integer HDFE_MAXMEMBUF
      integer HDFE_NAMBUFSIZE
      integer HDFE_DIMBUFSIZE

! Field Merge
      integer HDFE_NOMERGE
      integer HDFE_AUTOMERGE

! XXentries Modes
      integer HDFE_NENTDIM
      integer HDFE_NENTMAP
      integer HDFE_NENTIMAP
      integer HDFE_NENTGFLD
      integer HDFE_NENTDFLD

! GCTP projection codes
      integer GCTP_GEO
      integer GCTP_UTM
      integer GCTP_SPCS
      integer GCTP_ALBERS
      integer GCTP_LAMCC
      integer GCTP_MERCAT
      integer GCTP_PS
      integer GCTP_POLYC
      integer GCTP_EQUIDC
      integer GCTP_TM
      integer GCTP_STEREO
      integer GCTP_LAMAZ
      integer GCTP_AZMEQD
      integer GCTP_GNOMON
      integer GCTP_ORTHO
      integer GCTP_GVNSP
      integer GCTP_SNSOID
      integer GCTP_EQRECT
      integer GCTP_MILLER
      integer GCTP_VGRINT
      integer GCTP_HOM
      integer GCTP_ROBIN
      integer GCTP_SOM
      integer GCTP_ALASKA
      integer GCTP_GOOD
      integer GCTP_MOLL
      integer GCTP_IMOLL
      integer GCTP_HAMMER
      integer GCTP_WAGIV
      integer GCTP_WAGVII
      integer GCTP_OBLEQA
      integer GCTP_ISINUS1
      integer GCTP_CEA
      integer GCTP_BCEA
      integer GCTP_ISINUS

! Compression Modes
      integer HDFE_COMP_NONE
      integer HDFE_COMP_RLE
      integer HDFE_COMP_NBIT
      integer HDFE_COMP_SKPHUFF
      integer HDFE_COMP_DEFLATE
      integer HDFE_COMP_SZIP
      
! Tiling Codes
      integer HDFE_NOTILE
      integer HDFE_TILE

! Swath Subset Modes
      integer HDFE_MIDPOINT
      integer HDFE_ENDPOINT
      integer HDFE_ANYPOINT
      integer HDFE_INTERNAL
      integer HDFE_EXTERNAL
      integer HDFE_NOPREVSUB

! Grid Origin
      integer HDFE_GD_UL
      integer HDFE_GD_UR
      integer HDFE_GD_LL
      integer HDFE_GD_LR

! Grid Pixel Registration
      integer HDFE_CENTER
      integer HDFE_CORNER

! Angle Conversion Codes
      integer HDFE_RAD_DEG
      integer HDFE_DEG_RAD
      integer HDFE_DMS_DEG
      integer HDFE_DEG_DMS
      integer HDFE_RAD_DMS
      integer HDFE_DMS_RAD

! SZIP paramters
      integer SZ_EC, SZ_NN
      
! EASE GRID PARAMETERS
      integer BCEA_COLS25       ! total number of columns for EASE grid 
      integer BCEA_ROWS25       ! total number of rows for EASE grid 
      float BCEA_CELL_M         ! Cell size for EASE grid 
      float BCEA_RE_M           ! Earth radius used in GCTP projection tools
                                ! for Behrmann Cylindrical Equal Area proj.
      float DEFAULT_BCEA_LTRUESCALE !Latitude of true scale in DMS 
      float BCEA_COS_PHI1
      float PI
      float EASE_GRID_DEFAULT_UPLEFT_LON
      float EASE_GRID_DEFAULT_UPLEFT_LAT
      float EASE_GRID_DEFAULT_LOWRGT_LON
      float EASE_GRID_DEFAULT_LOWRGT_LAT



! Working Buffer Sizes
      parameter ( HDFE_MAXMEMBUF  = 1048576)
      parameter ( HDFE_NAMBUFSIZE = 32000)
      parameter ( HDFE_DIMBUFSIZE = 64000)

! Field Merge
      parameter ( HDFE_NOMERGE   = 0)
      parameter ( HDFE_AUTOMERGE = 1)

! XXentries Modes
      parameter ( HDFE_NENTDIM   = 0)
      parameter ( HDFE_NENTMAP   = 1)
      parameter ( HDFE_NENTIMAP  = 2)
      parameter ( HDFE_NENTGFLD  = 3)
      parameter ( HDFE_NENTDFLD  = 4)

! GCTP projection codes
      parameter ( GCTP_GEO       = 0)
      parameter ( GCTP_UTM       = 1)
      parameter ( GCTP_SPCS      = 2)
      parameter ( GCTP_ALBERS    = 3)
      parameter ( GCTP_LAMCC     = 4)
      parameter ( GCTP_MERCAT    = 5)
      parameter ( GCTP_PS        = 6)
      parameter ( GCTP_POLYC     = 7)
      parameter ( GCTP_EQUIDC    = 8)
      parameter ( GCTP_TM        = 9)
      parameter ( GCTP_STEREO    = 10)
      parameter ( GCTP_LAMAZ     = 11)
      parameter ( GCTP_AZMEQD    = 12)
      parameter ( GCTP_GNOMON    = 13)
      parameter ( GCTP_ORTHO     = 14)
      parameter ( GCTP_GVNSP     = 15)
      parameter ( GCTP_SNSOID    = 16)
      parameter ( GCTP_EQRECT    = 17)
      parameter ( GCTP_MILLER    = 18)
      parameter ( GCTP_VGRINT    = 19)
      parameter ( GCTP_HOM       = 20)
      parameter ( GCTP_ROBIN     = 21)
      parameter ( GCTP_SOM       = 22)
      parameter ( GCTP_ALASKA    = 23)
      parameter ( GCTP_GOOD      = 24)
      parameter ( GCTP_MOLL      = 25)
      parameter ( GCTP_IMOLL     = 26)
      parameter ( GCTP_HAMMER    = 27)
      parameter ( GCTP_WAGIV     = 28)
      parameter ( GCTP_WAGVII    = 29)
      parameter ( GCTP_OBLEQA    = 30)
      parameter ( GCTP_ISINUS1   = 31)
      parameter ( GCTP_CEA       = 97)
      parameter ( GCTP_BCEA      = 98)
      parameter ( GCTP_ISINUS    = 99)

! Compression Modes
      parameter ( HDFE_COMP_NONE     = 0)
      parameter ( HDFE_COMP_RLE      = 1)    
      parameter ( HDFE_COMP_NBIT     = 2)
      parameter ( HDFE_COMP_SKPHUFF  = 3)    
      parameter ( HDFE_COMP_DEFLATE  = 4)
      parameter ( HDFE_COMP_SZIP     = 5)

! Tiling Codes
      parameter ( HDFE_NOTILE         = 0)
      parameter ( HDFE_TILE           = 1)

! Swath Subset Modes
      parameter ( HDFE_MIDPOINT       = 0)
      parameter ( HDFE_ENDPOINT       = 1)
      parameter ( HDFE_ANYPOINT       = 2)
      parameter ( HDFE_INTERNAL       = 0)
      parameter ( HDFE_EXTERNAL       = 1)
      parameter ( HDFE_NOPREVSUB      = -1)

! Grid Origin
      parameter ( HDFE_GD_UL         = 0)
      parameter ( HDFE_GD_UR         = 1)
      parameter ( HDFE_GD_LL         = 2)
      parameter ( HDFE_GD_LR         = 3)

! Grid Pixel Registration
      parameter ( HDFE_CENTER        = 0)
      parameter ( HDFE_CORNER        = 1)

! Angle Conversion Codes
      parameter ( HDFE_RAD_DEG      = 0)
      parameter ( HDFE_DEG_RAD      = 1)
      parameter ( HDFE_DMS_DEG      = 2)
      parameter ( HDFE_DEG_DMS      = 3)
      parameter ( HDFE_RAD_DMS      = 4)
      parameter ( HDFE_DMS_RAD      = 5)

!     SZIP parameters
      parameter (SZ_EC = 4)
      parameter (SZ_NN = 32)

! EASE GRID PARAMETERS
      parameter( BCEA_COLS25             = 1383)
      parameter( BCEA_ROWS25             = 586)
      parameter( BCEA_CELL_M             = 25067.525)
      parameter( BCEA_RE_M               = 6371228.0)
      parameter( DEFAULT_BCEA_LTRUESCALE = 30.00)
! BCEA_COS_PHI1 = cos(DEFAULT_BCEA_LTRUESCALE *3.141592653589793238 /180.0)
      parameter( BCEA_COS_PHI1                 = 0.8660254)
      parameter( PI                            = 3.141592653589793238)
      parameter( EASE_GRID_DEFAULT_UPLEFT_LON  = -180.0)
      parameter( EASE_GRID_DEFAULT_UPLEFT_LAT  = 86.72)
      parameter( EASE_GRID_DEFAULT_LOWRGT_LON  = 180.0)
      parameter( EASE_GRID_DEFAULT_LOWRGT_LAT  = -86.72)
