/*---------------------------------------------------------------------------*/
/*                                                                           */
/*      COPYRIGHT       1994, 1995, 1996     Applied Research Corporation    */
/*                                                                           */
/*                      1997, 1998           Space Applications Corporation  */
/*                                                                           */
/*                      ALL RIGHTS RESERVED                                  */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*****************************************************************************
BEGIN_FILE_PROLOG:

FILENAME:

	odldef_prototypes.h

DESCRIPTION:

	This file contains function prototypes that are specific to the
	CUC Tools

AUTHOR:
	Ray Milburn / Steven Myers & Associates

HISTORY:
	02-Feb-99 RM Initial version

END_FILE_PROLOG:
*****************************************************************************/

#ifndef odldef_prototypes
#define odldef_prototypes

/*****************************************************************
    Function prototypes.
*****************************************************************/

void yyunput(int c);
int yyparse();
int yylook();
int yywrap();
int yyinput();
int yyback(int *p,int m);
int yylex();
int yyerror(char *msg);
char ODLPeekString(int pos);
char *ODLStoreString (char c);
void ODLNewString ();
char ODLBackupString ();
void ODLKillString ();
int ODLStringLength ();

#endif
