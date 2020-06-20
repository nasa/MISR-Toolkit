#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf.h>
#include <HdfEosDef.h>
#include <proj.h>
#include "misrproj.h"
#include "MisrCoordQuery.h"
#include "errormacros.h"

#define MAXNDIM	   10

typedef struct {
  int  block;
  float line;
  float sample;
} pts_t;

int npts = 40;
pts_t pts[] = { {   1,      -0.5,      -0.5 }, \
	        {   1, -0.499999, -0.499999 }, \
	        {   1,       0.0,       0.0 }, \
	        {   1,       0.5,       0.5 }, \
	        {   1,     127.0,     511.0 }, \
	        {   1,     127.5,     511.5 }, \
	        {   1,     511.0,    2047.0 }, \
	        {   1,     511.5,    2047.5 }, \
	        {   1,    101.97,     64.23 }, \
	        {   1,     101.0,      64.0 }, \
                {  65,      -0.5,      -0.5 }, \
	        {  65, -0.499999, -0.499999 }, \
	        {  65,       0.0,       0.0 }, \
	        {  65,       0.5,       0.5 }, \
	        {  65,     127.0,     511.0 }, \
	        {  65,     127.5,     511.5 }, \
	        {  65,     511.0,    2047.0 }, \
	        {  65,     511.5,    2047.5 }, \
	        {  65,    101.97,     64.23 }, \
	        {  65,     101.0,      64.0 }, \
                {  91,      -0.5,      -0.5 }, \
	        {  91, -0.499999, -0.499999 }, \
	        {  91,       0.0,       0.0 }, \
	        {  91,       0.5,       0.5 }, \
	        {  91,     127.0,     511.0 }, \
	        {  91,     127.5,     511.5 }, \
	        {  91,     511.0,    2047.0 }, \
	        {  91,     511.5,    2047.5 }, \
	        {  91,    101.97,     64.23 }, \
	        {  91,     101.0,      64.0 }, \
                { 180,      -0.5,      -0.5 }, \
	        { 180, -0.499999, -0.499999 }, \
	        { 180,       0.0,       0.0 }, \
	        { 180,       0.5,       0.5 }, \
	        { 180,     127.0,     511.0 }, \
	        { 180,     127.5,     511.5 }, \
	        { 180,     511.0,    2047.0 }, \
	        { 180,     511.5,    2047.5 }, \
	        { 180,    101.97,     64.23 }, \
	        { 180,     101.0,      64.0 }, \
};

int main(int argc, char *argv[]) {

  int32		   fid = FAIL;
  int32		   gid = FAIL;
  int		   igrid, i;
  int32		   ngrid;
  int32		   nline, nsample;
  double	   lat_r, lon_r;
  double	   savelon_r1, savelon_r2;
  double	   somx, somy;
  int		   b;
  float		   l, s;
  int32		   strbufsize;
  char		   *filepath;
  char		   **gridname;
  char		   *gridlist;
  float64	   ulc[2], lrc[2];
  int32		   spherecode, zonecode, projcode;
  float64	   projparam[NPROJ];
  float32	   offset[NOFFSET];
  long		   iflg;
  int		   status;
  char		   diffflg;
  int32		   ndim;
  int32		   dim[MAXNDIM];
  char		   dimlist[STRLEN];
  intn		   hdfeos_status_code;
  void		   *mem_status_code;
  long		   (*for_trans[MAXPROJ+1])();
  long		   (*inv_trans[MAXPROJ+1])();

  /* --------------- */
  /* Check arguments */
  /* --------------- */

  if (argc != 2) {
    fprintf(stderr, "Usage: %s hdfeos_grid_file\n", argv[0]);
    exit(1);
  }
  filepath = argv[1];

  /* ---------------------------------------------------- */
  /* Inquire and allocate memory for the hdfeos gridnames */
  /* This is only require if you need the gridnames       */
  /* ---------------------------------------------------- */

  hdfeos_status_code = GDinqgrid(filepath, NULL, &strbufsize);
  HDFEOS_ERROR_CHECK("GDinqgrid");

  mem_status_code = gridlist = (char *)malloc(strbufsize+1);
  MEM_ERROR_CHECK("malloc");

  hdfeos_status_code = ngrid = GDinqgrid(filepath, gridlist, NULL);
  HDFEOS_ERROR_CHECK("GDinqgrid");

  mem_status_code = gridname = (char **)malloc(ngrid * sizeof(char *));
  MEM_ERROR_CHECK("malloc");

  gridname[0] = strtok(gridlist, ",");
  for (igrid = 1; igrid < ngrid; igrid++) gridname[igrid] = strtok(NULL,",");

  /* ------------------------- */
  /* Open the hdfeos grid file */
  /* ------------------------- */

  hdfeos_status_code = fid = GDopen(filepath, DFACC_READ);
  HDFEOS_ERROR_CHECK("GDopen");

  /* ---------------------------------------- */
  /* Loop through all the grids because I can */
  /* ---------------------------------------- */

  for (igrid = 0; igrid < ngrid; igrid++) {

    /* ---------------------------- */
    /* Attach to the grid of choice */
    /* ---------------------------- */

    hdfeos_status_code = gid = GDattach(fid, gridname[igrid]);
    HDFEOS_ERROR_CHECK("GDattach");

    /* --------------------------------------------------------------- */
    /* Inquire grid dimensions to check number of blocks               */
    /* Inquire grid info to get the number of lines/sample and ulc/lrc */
    /* Inquire SOM relative block offsets                              */
    /* Initialize misr block/line/sample projection routines           */
    /* --------------------------------------------------------------- */

    hdfeos_status_code = ndim = GDinqdims(gid, dimlist, dim);
    HDFEOS_ERROR_CHECK("GDinqdims");

    /* Block dimension is always last dimension returned by GDinqdims. */
    if (dim[ndim - 1] != NBLOCK) MTK_ERROR("File does not have 180 blocks");

    hdfeos_status_code = GDgridinfo(gid, &nline, &nsample, ulc, lrc);
    HDFEOS_ERROR_CHECK("GDgridinfo");

    hdfeos_status_code = GDblkSOMoffset(gid, offset, NOFFSET, "r");
    HDFEOS_ERROR_CHECK("GDblkSOMoffset");

    status = misr_init(NBLOCK, nline, nsample, offset, ulc, lrc);
    if (status) MTK_ERROR("misr_init");

    printf("\nFilename (path/orbit): %s\n", filepath);
    printf("Gridname: %s\n", gridname[igrid]);
    printf("Lines/Samples: (%d, %d)\n", nline, nsample);
    printf("ULC (x,y) (m): (%f, %f)\n", ulc[0], ulc[1]);
    printf("LRC (x,y) (m): (%f, %f)\n", lrc[0], lrc[1]);
    printf("Block offsets: (%f", offset[0]);
    for (i = 1; i < NOFFSET; i++) printf(", %f", offset[i]);
    printf(")\n");

    /* ------------------------------------------------------------ */
    /* Inquire grid projection info to get project codes/parameters */
    /* Initialize gctp SOM forward and inverse projection routines  */
    /* ------------------------------------------------------------ */

    hdfeos_status_code = GDprojinfo(gid, &projcode, &zonecode,
				    &spherecode, projparam);
    HDFEOS_ERROR_CHECK("GDprojinfo");

    for_init((long)projcode, (long)zonecode, (double*)projparam,
	     (long)spherecode, NULL, NULL, &iflg, for_trans);
    if (iflg) MTK_ERROR("for_init");

    inv_init((long)projcode, (long)zonecode, (double*)projparam,
	     (long)spherecode, NULL, NULL, &iflg, inv_trans);
    if (iflg) MTK_ERROR("inv_init");

    printf("GCTP projection code: %d\n", projcode);
    printf("GCTP zone code (not used for SOM): %d\n", zonecode);
    printf("GCTP sphere code: %d\n", spherecode);
    printf("GCTP projection parameters: (%f",projparam[0]);
    for (i = 1; i < NPROJ; i++) printf(", %f", projparam[i]);
    printf(")\n");

    /* --------------------------------------------------------------------- */
    /* Detach from the grid because we don't need it anymore in this example */
    /* We would need it if we go on to access fields, so don't detach here   */
    /* --------------------------------------------------------------------- */

    if (gid != FAIL) GDdetach(gid);

    /* ----------------------------------------- */
    /* Loop over some inverse transformations    */
    /* (b,l.l,s.s) -> (X,Y) -> (lat,lon)         */
    /* and over some forward transformations     */
    /* (lat,lon) -> (X,Y) -> (b,l.l,s.s)         */
    /* ----------------------------------------- */

    printf("    (blk,     line  ,    sample  )    "
           "(        SOM X    ,        SOM Y    )    "
	   "(    Lat   ,     Lon   )\n");

    for (i = 0; i < npts; i++) {

      b = pts[i].block;
      l = pts[i].line;
      s = pts[i].sample;

      /* -------------------------------------------------------- */
      /* Inverse transformation (b,l.l,s.s) -> (X,Y) -> (lat,lon) */
      /* -------------------------------------------------------- */

      misrinv(b, l, s, &somx, &somy);      /* (b,l.l,s.s) -> (X,Y) */
      sominv(somx, somy, &lon_r, &lat_r);  /* (X,Y) -> (lat,lon) */

      printf("%2d: (%3d,%11.6f,%12.6f) -> (%17.6f,%17.6f) -> "
             "(%10.6f,%11.6f) --|\n",
	     i, b, l, s, somx, somy, lat_r * R2D, lon_r * R2D);

      /* -------------------------------------------------------- */
      /* Forward transformation (lat,lon) -> (X,Y) -> (b,l.l,s.s) */
      /* -------------------------------------------------------- */

      somfor(lon_r, lat_r, &somx, &somy);  /* (lat,lon) -> (X,Y) */
      misrfor(somx, somy, &b, &l, &s);     /* (X,Y) -> (b,l.l,s.s) */

      if (b != pts[i].block) diffflg = '*';
      else diffflg = ' ';

      printf("  %c (%3d,%11.6f,%12.6f) <- (%17.6f,%17.6f) <- "
	     "(%10.6f,%11.6f) <-|\n",
	     diffflg, b, l, s, somx, somy, lat_r * R2D, lon_r * R2D);

      /* -------------------------------------------------- */
      /* Save the longitude of block 91 to find location of */
      /* equator crossing				    */
      /* -------------------------------------------------- */

      if (pts[i].block == 91 && 
	  pts[i].line == 0.0 &&
	  pts[i].sample == 0.0) {
	savelon_r1 = lon_r;
      }
      if (pts[i].block == 91 && 
	  pts[i].line == (float)(nline-1) &&
	  pts[i].sample == (float)(nsample-1)) {
	savelon_r2 = lon_r;
      }
    }

    /* --------------------------------------------------- */
    /* Determine block/line/sample of the equator crossing */
    /* approximately in the center of the block            */
    /* --------------------------------------------------- */

    lat_r = 0.0;
    if (savelon_r1 < 0.0 && savelon_r2 > 0.0 ||
	savelon_r1 > 0.0 && savelon_r2 < 0.0) {
      lon_r = (savelon_r1 - savelon_r2) / 2.0;
    } else {
      lon_r = (savelon_r1 + savelon_r2) / 2.0;
    }

    /* -------------------------------------------------------- */
    /* Forward transformation (lat,lon) -> (X,Y) -> (b,l.l,s.s) */
    /* -------------------------------------------------------- */

    somfor(lon_r, lat_r, &somx, &somy);  /* (lat,lon) -> (X,Y) */
    misrfor(somx, somy, &b, &l, &s);     /* (X,Y) -> (b,l.l,s.s) */

    printf("%2d: (%3d,%11.6f,%12.6f) <- (%17.6f,%17.6f) <- "
	   "(%10.6f,%11.6f) = equator crossing\n",
	   npts, b, l, s, somx, somy, lat_r * R2D, lon_r * R2D);

    /* -------------------------------------- */
    /* Extreme upper left corner (not center) */
    /* -------------------------------------- */

    somx = ulc[0];
    somy = lrc[1];		   /* Notice the switch from ulc[1]. */

    sominv(somx, somy, &lon_r, &lat_r);
    misrfor(somx, somy, &b, &l, &s);

    printf("%2d: (%3d,%11.6f,%12.6f) <- (%17.6f,%17.6f) -> "
	   "(%10.6f,%11.6f) = ulc of block 1\n",
	   npts+1, b, l, s, somx, somy, lat_r * R2D, lon_r * R2D);

    /* ------------------- ------------------- */
    /* Extreme lower right corner (not center) */
    /* -------------------- ------------------ */

    somx = lrc[0];
    somy = ulc[1];		   /* Notice the switch from lrc[1]. */

    sominv(somx, somy, &lon_r, &lat_r);
    misrfor(somx, somy, &b, &l, &s);

    printf("%2d: (%3d,%11.6f,%12.6f) <- (%17.6f,%17.6f) -> "
	   "(%10.6f,%11.6f) = lrc of block 1\n",
	   npts+2, b, l, s, somx, somy, lat_r * R2D, lon_r * R2D);

    /* -------------------------------------- */
    /* Origin of SOM projection for this path */
    /* -------------------------------------- */

    somx = 0.0;
    somy = 0.0;

    sominv(somx, somy, &lon_r, &lat_r);
    misrfor(somx, somy, &b, &l, &s);

    printf("%2d: (%3d,%11.6f,%12.6f) <- (%17.6f,%17.6f) -> "
	   "(%10.6f,%11.6f) = SOM origin (long of asc node)\n",
	   npts+3, b, l, s, somx, somy, lat_r * R2D, lon_r * R2D);

    /* --------------------------------------------------- */
    /* Origin of SOM projection plus 180 degrees longitude */
    /* --------------------------------------------------- */

    lat_r = 0.0;
    lon_r = (lon_r > 0.0 ? lon_r - (180.0*D2R) : lon_r + (180.0*D2R));

    somfor(lon_r, lat_r, &somx, &somy);
    misrfor(somx, somy, &b, &l, &s);

    printf("%2d: (%3d,%11.6f,%12.6f) <- (%17.6f,%17.6f) <- "
	   "(%10.6f,%11.6f) = SOM origin plus 180 in long.\n",
	   npts+4, b, l, s, somx, somy, lat_r * R2D, lon_r * R2D);

    /* --------------------------------------- */
    /* Equator crossing using SOM X from above */
    /* --------------------------------------- */

    somy = 0.0;

    sominv(somx, somy, &lon_r, &lat_r);
    misrfor(somx, somy, &b, &l, &s);

    printf("%2d: (%3d,%11.6f,%12.6f) <- (%17.6f,%17.6f) -> "
	   "(%10.6f,%11.6f) = equator crossing\n",
	   npts+5, b, l, s, somx, somy, lat_r * R2D, lon_r * R2D);

  }

  if (fid != FAIL) GDclose(fid);
  if (gridlist) free(gridlist);
  if (gridname) free(gridname);

  printf("\nNotes:\n\n"
"1) Given a block, fractional line and fraction sample triplet the\n"
"   following transformations performed:\n\n"
"    Inverse transformation:     (b,l.l,s.s) -> (X,Y) -> (lat,lon) -|\n"
"                            |--------------------------------------|\n"
"    Forward transformation: |-> (lat,lon) -> (X,Y) -> (b,l.l,s.s)\n\n"
"2) The transforms marked with a * did not reproduce the same\n"
"   answer either because of rounding errors in the GCTP codes or because\n" 
"   they are out of bounds of the particular grid.  The misr transform\n"
"   routines (misr_init, misrfor and misrinv) are designed to handle out\n"
"   of bounds conditions and return all -1's.  This enables a resampling\n"
"   routine to determine whether resampling can be done or not, if these\n"
"   routines are used for reprojection.\n\n"
"3) Notice that the ULC Y/LRC Y values returned by gridinfo are incorrectly \n"
"   switched when compared to transform number 0, 5 or 7 (depending\n"
"   on resolution).\n\n"
"4) Also note that the ULC/LRC values returned by gridinfo are for block 1\n"
"   extreme pixel edges (not pixel centers).\n\n"
"5) Note that SOM X is always increasing as blocks increase (in fact,\n"
"   SOM X is zero meters at the longitude of the ascending node - the\n"
"   5th parameter of projection paramters).  SOM Y tends to be mostly\n"
"   positive in the Northern blocks and negative in the Southern blocks.\n"
"   Each SOM path is a separate projection with the origin at the\n"
"   night side equator and the longitude of the ascending node.\n\n"
"6) The block offsets are the number of 1.1km subregions from the\n"
"   previous block.  The first offset is relative first block.\n\n"
"7) The 4th and 5th projection parameter are in the format of packed\n"
"   dddmmmsss.ss as documented in the GCTP codes (see paksz.c).\n\n"
"8) MISR uses the GCTP SOM projection A which specifies the inclination\n"
"   angle and longitude of the ascending node instead of path number\n\n"
"9) The last six transformations compute various special case locations.\n"
"   Note the direction of the transform arrows.  Can you determine why the\n"
"   lrc of block 1 is actually in block 2?  Hint: it is not the pixel\n"
"   center, but rather the edge.\n\n"
"10) Last note.  Remember that the SOM projection is singular at the poles\n"
"    and thus undefined there.\n\n"
	 );
  exit(0);
}
