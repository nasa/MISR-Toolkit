/*===========================================================================
=                                                                           =
=                              misr_types.h                                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef _MISRTypes_h
#define _MISRTypes_h 1

#include "Byte.h"
#include "UInt16.h"
#include "Int16.h"
#include "UInt32.h"
#include "Int32.h"
#include "Float32.h"
#include "Float64.h"
#include "Str.h"
#include "Url.h"
#include "Array.h"
#include "Structure.h"
#include "Sequence.h"
#include "Grid.h"

#include "MisrToolkit.h"

/* ------------------------------------------------------------------------ */
/* MISRByte								    */
/* ------------------------------------------------------------------------ */

extern Byte * NewByte(const string &n = "");

class MISRByte: public Byte {
public:
    MISRByte(const string &n = "");
    virtual ~MISRByte() {}

    virtual BaseType *ptr_duplicate();

    virtual bool read(const string &dataset);
};

/* ------------------------------------------------------------------------ */
/* MISRUInt16								    */
/* ------------------------------------------------------------------------ */

extern UInt16 * NewUInt16(const string &n = "");

class MISRUInt16: public UInt16 {
public:
    MISRUInt16(const string &n = "");
    virtual ~MISRUInt16() {}

    virtual BaseType *ptr_duplicate();

    virtual bool read(const string &dataset);
};

/* ------------------------------------------------------------------------ */
/* MISRInt16								    */
/* ------------------------------------------------------------------------ */

extern Int16 * NewInt16(const string &n = "");

class MISRInt16: public Int16 {
public:
    MISRInt16(const string &n = "");
    virtual ~MISRInt16() {}

    virtual BaseType *ptr_duplicate();

    virtual bool read(const string &dataset);
};

/* ------------------------------------------------------------------------ */
/* MISRUInt32								    */
/* ------------------------------------------------------------------------ */

extern UInt32 * NewUInt32(const string &n = "");

class MISRUInt32: public UInt32 {
public:
    MISRUInt32(const string &n = "");
    virtual ~MISRUInt32() {}

    virtual BaseType *ptr_duplicate();

    virtual bool read(const string &dataset);
};

/* ------------------------------------------------------------------------ */
/* MISRInt32								    */
/* ------------------------------------------------------------------------ */

extern Int32 * NewInt32(const string &n = "");

class MISRInt32: public Int32 {
public:
    MISRInt32(const string &n = "");
    virtual ~MISRInt32() {}

    virtual BaseType *ptr_duplicate();

    virtual bool read(const string &dataset);
};

/* ------------------------------------------------------------------------ */
/* MISRFloat32								    */
/* ------------------------------------------------------------------------ */

extern Float32 * NewFloat32(const string &n = "");

class MISRFloat32: public Float32 {
public:
    MISRFloat32(const string &n = "");
    virtual ~MISRFloat32() {}

    virtual BaseType *ptr_duplicate();

    virtual bool read(const string &dataset);
};

/* ------------------------------------------------------------------------ */
/* MISRFloat64								    */
/* ------------------------------------------------------------------------ */

extern Float64 * NewFloat64(const string &n = "");

class MISRFloat64: public Float64 {
public:
    MISRFloat64(const string &n = "");
    virtual ~MISRFloat64() {}

    virtual BaseType *ptr_duplicate();

    virtual bool read(const string &dataset);
};

/* ------------------------------------------------------------------------ */
/* MISRStr								    */
/* ------------------------------------------------------------------------ */

extern Str * NewStr(const string &n = "");

class MISRStr: public Str {
public:
    MISRStr(const string &n = "");
    virtual ~MISRStr() {}

    virtual BaseType *ptr_duplicate();

    virtual bool read(const string &dataset);
};

/* ------------------------------------------------------------------------ */
/* MISRUrl								    */
/* ------------------------------------------------------------------------ */

extern Url * NewUrl(const string &n = "");

class MISRUrl: public Url {
public:
    MISRUrl(const string &n = "");
    virtual ~MISRUrl() {}

    virtual BaseType *ptr_duplicate();

    virtual bool read(const string &dataset);
};

/* ------------------------------------------------------------------------ */
/* MISRArray								    */
/* ------------------------------------------------------------------------ */

extern Array * NewArray(const string &n = "", MTKt_MapInfo *mapinfo = NULL);

class MISRArray: public Array {
public:
    MTKt_MapInfo global_mapinfo;

    MISRArray(const string &n = "", MTKt_MapInfo *mapinfo = NULL);
    virtual ~MISRArray() {}

    void set_mapinfo(MTKt_MapInfo mapinfo);

    virtual BaseType *ptr_duplicate();

    virtual bool read(const string &dataset);
private:
    MTKt_MapInfo global_mapinfo;

};

/* ------------------------------------------------------------------------ */
/* MISRStructure				       			    */
/* ------------------------------------------------------------------------ */

extern Structure * NewStructure(const string &n = "");

class MISRStructure: public Structure {
public:
    MISRStructure(const string &n = "");
    virtual ~MISRStructure() {}

    virtual BaseType *ptr_duplicate();

    virtual bool read(const string &dataset);
};

/* ------------------------------------------------------------------------ */
/* MISRSequence								    */
/* ------------------------------------------------------------------------ */

extern Sequence * NewSequence(const string &n = "");

class MISRSequence: public Sequence {
public:
    MISRSequence(const string &n = "");
    virtual ~MISRSequence() {}

    virtual BaseType *ptr_duplicate();

    virtual bool read(const string &dataset);
};

/* ------------------------------------------------------------------------ */
/* MISRGrid								    */
/* ------------------------------------------------------------------------ */

extern Grid * NewGrid(const string &n = "");

class MISRGrid: public Grid {
public:
    MISRGrid(const string &n = "");
    virtual ~MISRGrid() {}

    virtual BaseType *ptr_duplicate();

    virtual bool read(const string &dataset);
};

#endif // _MISRTypes_h
