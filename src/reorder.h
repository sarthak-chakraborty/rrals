#ifndef SPLATT_REORDER_H
#define SPLATT_REORDER_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sptensor.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void tt_perm(
  sptensor_t * const tt,
  idx_t const * const perm);

#endif
