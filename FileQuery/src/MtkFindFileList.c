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

#include "MisrFileQuery.h"
#include "MisrError.h"
#include <stdio.h>    /* standard input/output routines.    */
#include <sys/stat.h> /* stat(), etc.                       */
#include <string.h>   /* strstr(), etc.                     */

#ifdef _WIN32
  #include "dirent_win32.h"
#else
  #include <dirent.h>   /* readdir(), etc.                    */
#endif

#ifndef _WIN32
  #include <unistd.h>   /* getcwd(), etc.                     */
#endif

#ifndef REGEXP_WORKAROUND
#include <regex.h>
#endif

#ifndef S_ISDIR
#define S_ISDIR(x) (((x) & S_IFMT) == S_IFDIR)
#endif

#ifndef REG_BASIC	 /* REG_BASIC is not defined in regex.h in Linux */
# define REG_BASIC 0
#endif

#define MAX_DIR_PATH 2048	/* maximal full path we support.      */
#define FILE_LIST_SIZE 20       /* Size to extend file list by */

#ifndef REGEXP_WORKAROUND
static void findfile(const regex_t* preg, int *count, int *max, char ***file_list, MTKt_status *status);
#endif


/** \brief Find files in a directory tree, using regular expressions
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we search the directory \c misr_products for all GRP files related to the DF camera.
 *
 *  \code
 *  status = MtkFindFileList("misr_products", "GRP.*", "DF", ".*", ".*", ".*", &filecnt, &filenames);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkStringListFree() to free the memory used by \c filenames
 */

MTKt_status MtkFindFileList(
  const char *searchdir, /**< [IN] Search Directory */
  const char *product,   /**< [IN] Product */
  const char *camera,    /**< [IN] Camera */
  const char *path,      /**< [IN] Path */
  const char *orbit,     /**< [IN] Orbit */
  const char *version,   /**< [IN] Version */
  int *filecnt,          /**< [OUT] File count */
  char **filenames[]     /**< [OUT] Filenames */ )
{
#ifdef REGEXP_WORKAROUND
	return MTK_FAILURE;
#else // regex enabled    
	MTKt_status status;   /* Return status */
	int status_code = 1;
	int chdir_status = 0;
	char **temp_file_list = NULL;
	char *curr_dir = NULL; /* current directory */
	int max = FILE_LIST_SIZE;
    char* dir_path;		  /* path to the directory. */
    regex_t preg;    
    int count;

    int path_name_size = 1024; /* length of curr_dir */
    struct stat dir_stat; /* used by stat().        */
    char temp[128];


    /* Check Arguments */
    if (searchdir == NULL || product == NULL || path == NULL ||
        orbit == NULL || version == NULL || filecnt == NULL ||
        filenames == NULL)
      MTK_ERR_CODE_JUMP(MTK_NULLPTR);

    dir_path = (char *)searchdir;

    /* make sure the given path refers to a directory. */
    if (stat(dir_path, &dir_stat) == -1) {
	/* perror("stat:"); */
	   MTK_ERR_CODE_JUMP(MTK_FAILURE);
    }

    if (!S_ISDIR(dir_stat.st_mode)) {
	/* fprintf(stderr, "'%s' is not a directory\n", dir_path); */
	   MTK_ERR_CODE_JUMP(MTK_FAILURE);
    }
    
    /* save current directory
       allocate a larger buffer if needed */
    curr_dir = (char*)malloc(path_name_size * sizeof(char));
    while (getcwd(curr_dir,path_name_size) == NULL)
    {
      path_name_size += 1024;
      curr_dir = (char*)realloc(curr_dir,path_name_size * sizeof(char));
    }

    /* change into the given directory. */
    if (chdir(dir_path) == -1) {
	/*fprintf(stderr, "Cannot change to directory '%s': ", dir_path);
	perror(""); */
	   MTK_ERR_CODE_JUMP(MTK_FAILURE);
    }

    /* Create regular expression */
    if (camera == NULL)
      sprintf(temp,"MISR_AM1_%s_P%s_O%s_%s.hdf",product,path,
              orbit,version);
    else if (strlen(camera) == 0)
      sprintf(temp,"MISR_AM1_%s_P%s_O%s_%s.hdf",product,path,
              orbit,version);
    else
      sprintf(temp,"MISR_AM1_%s_P%s_O%s_%s_%s.hdf",product,path,
              orbit,camera,version);

    regcomp(&preg,temp,REG_BASIC | REG_NOSUB);

    /* Allocated memory */
    max = FILE_LIST_SIZE;
    temp_file_list = (char**)calloc(max,sizeof(int*));
    count = 0;

    /* recursively scan the directory for the given file name pattern. */
    status = MTK_SUCCESS;
    findfile(&preg,&count,&max,&temp_file_list,&status);

    *filecnt = count;
    *filenames = temp_file_list;

    /* restore current directory */
    chdir_status = chdir(curr_dir);
    if (chdir_status) {
        perror("chdir error: ");
    }

    regfree(&preg);
    free(curr_dir);

    return MTK_SUCCESS;

ERROR_HANDLE:
  if (temp_file_list != NULL)
    MtkStringListFree(max, &temp_file_list);

  if (curr_dir != NULL)
  {
    chdir_status = chdir(curr_dir); /* restore current directory */
    if (chdir_status) {
        perror("chdir error: ");
    }
    free(curr_dir);
  }

  return status_code;
#endif
}

#ifndef REGEXP_WORKAROUND
static void findfile(const regex_t* preg, int *count, int *max, char ***file_list, MTKt_status *status)
{
    DIR* dir;			/* pointer to the scanned directory. */
    struct dirent* entry;	/* pointer to one directory entry.   */
    char cwd[MAX_DIR_PATH+1];	/* current working directory.        */
    struct stat dir_stat;       /* used by stat().                   */
    int len;
    char **temp_file_list;
    int i;

    /* first, save path of current working directory */
    if (!getcwd(cwd, MAX_DIR_PATH+1))
    {
	/*perror("getcwd:");*/
	  return;
    }

    /* open the directory for reading */
    dir = opendir(".");
    if (!dir)
    {
	/*fprintf(stderr, "Cannot read directory '%s': ", cwd);
	perror("");*/
	  return;
    }

    /* scan the directory, traversing each sub-directory, and */
    /* matching the pattern for each file name.               */
    while ((entry = readdir(dir)))
    {
	  /* check if the pattern matchs. */
      if (entry && regexec(preg, entry->d_name, 0, NULL, 0) == 0)
      {
	    len = strlen(cwd) + strlen(entry->d_name) + 2;
	    if (*count < *max)
	       (*file_list)[*count] = (char*)malloc(len * sizeof(char));
	    else /* Extend List */
        {
	      temp_file_list = (char**)calloc(*max + FILE_LIST_SIZE, sizeof(int*));
	      if (temp_file_list == NULL)
          {
		    /*fprintf(stderr,"Error extending memory\n");*/
		    *status = MTK_FAILURE;
		    return;
	      }
	       
	      *max += FILE_LIST_SIZE;
	      for (i = 0; i < *count; ++i)
		  temp_file_list[i] = (*file_list)[i];
	       
	      temp_file_list[*count] = (char*)malloc(len * sizeof(char));
	      *file_list = temp_file_list;
	    }

	    strcpy((*file_list)[*count],cwd);
	    strcat((*file_list)[*count],"/");
	    strcat((*file_list)[*count],entry->d_name);
	    ++*count;
	  }
      
      /* check if the given entry is a directory. */
      if (stat(entry->d_name, &dir_stat) == -1)
      {
	    /*perror("stat:");*/
	    continue;
      }
	
	  /* skip the "." and ".." entries, to avoid loops. */
	  if (strcmp(entry->d_name, ".") == 0)
	    continue;
	  
	  if (strcmp(entry->d_name, "..") == 0)
	    continue;
	  
	  /* is this a directory? */
      if (S_ISDIR(dir_stat.st_mode))
      {
        /* Change into the new directory */
        if (chdir(entry->d_name) == -1)
        {
	      /*fprintf(stderr, "Cannot chdir into '%s': ", entry->d_name);
	        perror("");*/
	      continue;
        }
	    /* check this directory */
	    findfile(preg,count,max,file_list,status);

        /* finally, restore the original working directory. */
        if (chdir("..") == -1)
        {
	        /*fprintf(stderr, "Cannot chdir back to '%s': ", cwd);
	        perror("");*/
	      return;
        }
      }
    }
    
    closedir(dir);
}
#endif
