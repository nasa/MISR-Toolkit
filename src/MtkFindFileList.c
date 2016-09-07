/*===========================================================================
=                                                                           =
=                             MtkFindFileList                               =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrToolkit.h"
#include "MisrError.h"
#include <getopt.h>
#include <stdio.h>		/* for printf */
#include <stdlib.h>
#include <string.h>

typedef struct {
  char *searchdir; /* Search Directory */
  char *product;   /* Product */
  char *camera;    /* Camera */
  char *path;      /* Path */
  char *orbit;     /* Orbit */
  char *version;   /* Version */
} argr_type;	   /* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main(int argc, char *argv[])
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  int num_files;                /* Number of files */
  char **filelist = NULL;       /* File list */
  int i;
  argr_type argr = {NULL, NULL, NULL, NULL, NULL, NULL}; /* Parse arguments */

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);


  status = MtkFindFileList(argr.searchdir, argr.product, argr.camera,
			   argr.path, argr.orbit, argr.version,
			   &num_files, &filelist);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkFindFileList failed!\n");
  }

  for (i = 0; i < num_files; ++i)
    printf("%s\n",filelist[i]);

  MtkStringListFree(num_files, &filelist);

  free(argr.searchdir);
  free(argr.product);
  free(argr.camera);
  free(argr.path);
  free(argr.orbit);
  free(argr.version);

  return 0;

ERROR_HANDLE:
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> | <--dir=directory> "
    "<--prod=product> [ <--cam=camera> ] <--path=path> <--orbit=orbit> "
    "<--ver=version>\n\n",func);
  fprintf(stderr, "Where: --dir=directory is top level directory to search. \n");
  fprintf(stderr, "       --prod=product is the product to search for.\n");
  fprintf(stderr, "       --cam=camera is the camera.\n");
  fprintf(stderr, "       --path=path is the path number.\n");
  fprintf(stderr, "       --orbit=orbit is the orbit number.\n");
  fprintf(stderr, "       --ver=version is the version number.\n");
  fprintf(stderr,"Note: All of the above options support regular expressions.\n");
  fprintf(stderr, "       --help is this info\n");
  fprintf(stderr, "\nExample: MtkFindFileList --dir=. --prod=\"GRP.*\" "
          "--cam=DA --path=\".*\" --orbit=\".*\" --ver=F03_0024\n");
 fprintf(stderr, "         MtkFindFileList --dir=. --prod=\"TC.*\" --path=037 --orbit=\".*\" --ver=\".*\"\n");
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE;
  extern char *optarg;
  extern int optind;
  int ch;
  int dflag = 0, rflag = 0, pflag = 0, oflag = 0, vflag = 0;

  /* options descriptor */
  static struct option longopts[] = {
    { "dir",   required_argument, 0, 'd' },
    { "prod",  required_argument, 0, 'r' },
    { "cam",   required_argument, 0, 'c' },
    { "path",  required_argument, 0, 'p' },
    { "orbit", required_argument, 0, 'o' },
    { "ver",   required_argument, 0, 'v' },
    { "help",  no_argument,       0, 'h' },
    { 0,      0,                 0,  0 }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "d:r:c:p:o:v:h", longopts, NULL)) != -1) {
    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_SUCCESS);
      break;
    case 'd':
      argr->searchdir = malloc((strlen(optarg) + 1) * sizeof(char));
      if (argr->searchdir == NULL)
	MTK_ERR_CODE_JUMP(MTK_NULLPTR);
      strcpy(argr->searchdir,optarg);
      dflag = 1;
      break;
    case 'r':
      argr->product = malloc((strlen(optarg) + 1) * sizeof(char));
      if (argr->product == NULL)
	MTK_ERR_CODE_JUMP(MTK_NULLPTR);
      strcpy(argr->product,optarg);
      rflag = 1;
      break;
    case 'c':
      argr->camera = malloc((strlen(optarg) + 1) * sizeof(char));
      if (argr->camera == NULL)
	MTK_ERR_CODE_JUMP(MTK_NULLPTR);
      strcpy(argr->camera,optarg);
      break;
    case 'p':
      argr->path = malloc((strlen(optarg) + 1) * sizeof(char));
      if (argr->path == NULL)
	MTK_ERR_CODE_JUMP(MTK_NULLPTR);
      strcpy(argr->path,optarg);
      pflag = 1;
      break;
    case 'o':
      argr->orbit = malloc((strlen(optarg) + 1) * sizeof(char));
      if (argr->orbit == NULL)
	MTK_ERR_CODE_JUMP(MTK_NULLPTR);
      strcpy(argr->orbit,optarg);
      oflag = 1;
      break;
    case 'v':
      argr->version = malloc((strlen(optarg) + 1) * sizeof(char));
      if (argr->version == NULL)
	MTK_ERR_CODE_JUMP(MTK_NULLPTR);
      strcpy(argr->version,optarg);
      vflag = 1;
      break;
    case '?':
    default:
      MTK_ERR_MSG_JUMP("Invalid arguments");
    }
  }

  if (!(dflag && rflag && pflag && oflag && vflag))
    MTK_ERR_MSG_JUMP("Invalid arguments");

  return MTK_SUCCESS;

ERROR_HANDLE:
  usage(argv[0]);
  return status_code;
}
