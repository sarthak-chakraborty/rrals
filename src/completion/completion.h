#ifndef SPLATT_COMPLETE_H
#define SPLATT_COMPLETE_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../base.h"
#include "../matrix.h"
#include "../sptensor.h"
#include "../timer.h"

#define SPLATT_USE_VALIDATION 1

#ifdef SPLATT_USE_MPI
#include "../splatt_mpi.h"
#endif


#define ALS_BUFSIZE 2048

/*
 * Modes shorter than this are be considered 'dense'.
 * TODO: This should actually be a function of the number of nonzeros and
 * probably the number of threads/ranks.
 */
#define DENSEMODE_THRESHOLD 300

/* swap two val_t pointers */
#define SPLATT_VPTR_SWAP(x,y) \
do {\
  val_t * tmp = (x);\
  (x) = (y);\
  (y) = tmp;\
} while(0)



/******************************************************************************
 * TYPES
 *****************************************************************************/

typedef enum
{
  SPLATT_TC_SGD,
  SPLATT_TC_GD,
  SPLATT_TC_NLCG,
  SPLATT_TC_LBFGS,
  SPLATT_TC_CCD,
  SPLATT_TC_ALS,
  SPLATT_TC_NALGS,
  SPLATT_TC_SPALS,
  SPLATT_TC_RRALS
} splatt_tc_type;


typedef struct
{
  idx_t rank;
  idx_t nmodes;
  splatt_tc_type which;

  idx_t dims[MAX_NMODES];
  val_t * factors[MAX_NMODES];

#ifdef SPLATT_USE_MPI
  matrix_t * globmats[MAX_NMODES];
#endif
} tc_model;



typedef struct
{
  idx_t nmodes;
  val_t learn_rate;
  idx_t max_its;
  double max_seconds;
  val_t regularization[MAX_NMODES];

  val_t * numerator;
  val_t * denominator;

  idx_t nthreads;
  thd_info * thds;

  sp_timer_t tc_time;

  /* CCD++ */
  idx_t num_inner;

  /* GD */
  sp_timer_t grad_time;
  sp_timer_t line_time;

  /* CCD++ */
  sp_timer_t resid_time;
  sp_timer_t dense_time;
  sp_timer_t sparse_time;
  sp_timer_t newcol_time;

  /* some algs handle dense modes separately */
  idx_t num_dense;
  bool isdense[MAX_NMODES];
  idx_t maxdense_dim;

  /* results + convergence */
  idx_t max_badepochs;
  idx_t nbadepochs;
  val_t tolerance;
  idx_t best_epoch;
  val_t best_rmse;
  tc_model * best_model;

#ifdef SPLATT_USE_MPI
  rank_info * rinfo;

  /* send/recv buffers */
  val_t * nbr2globs_buf;
  val_t * local2nbr_buf;
#endif

} tc_ws;






/******************************************************************************
 * TENSOR COMPLETION FUNCTIONS
 *****************************************************************************/

void splatt_tc_sgd(
    sptensor_t * train,
    sptensor_t const * const validate,
    tc_model * const model,
    tc_ws * const ws);


void splatt_tc_als(
    sptensor_t * train,
    sptensor_t * const validate,
    tc_model * const model,
    tc_ws * const ws);


void splatt_tc_gd(
    sptensor_t * train,
    sptensor_t const * const validate,
    tc_model * const model,
    tc_ws * const ws);

void splatt_tc_nlcg(
    sptensor_t * train,
    sptensor_t const * const validate,
    tc_model * const model,
    tc_ws * const ws);

void splatt_tc_lbfgs(
    sptensor_t * train,
    sptensor_t const * const validate,
    tc_model * const model,
    tc_ws * const ws);


void splatt_tc_ccd(
    sptensor_t * train,
    sptensor_t * const validate,
    tc_model * const model,
    tc_ws * const ws);

void splatt_tc_spals(
    sptensor_t * train,
    sptensor_t * const validate,
    tc_model * const model,
    tc_ws * const ws);

void splatt_tc_rrals(
    sptensor_t * train,
    sptensor_t * const validate,
    tc_model * const model,
    tc_ws * const ws,
    int alpha,
    int beta);


/******************************************************************************
 * WORKSPACE FUNCTIONS
 *****************************************************************************/

#define tc_ws_alloc splatt_tc_ws_alloc
/**
* @brief Allocate and initialize a workspace used for tensor completion.
*
* @param model The training data.
* @param model The model we will be computing.
* @param nthreads The number of threads to use during the factorization.
*
* @return The allocated workspace.
*/
tc_ws * tc_ws_alloc(
    sptensor_t const * const train,
    tc_model const * const model,
    idx_t nthreads);



#define tc_ws_free splatt_tc_ws_free
/**
* @brief Free the memory allocated for a workspace.
*
* @param ws The workspace to free.
*/
void tc_ws_free(
    tc_ws * ws);




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


#define tc_rmse splatt_tc_rmse
/**
* @brief Compute the RMSE against of a factorization against a test tensor.
*        RMSE is defined as: sqrt( tc_loss_sq() / nnz ).
*
* @param test The tensor to test against.
* @param model The factorization the evaluate.
* @param ws Workspace to use (thread buffers are accessed).
*
* @return The RMSE.
*/
val_t tc_rmse(
    sptensor_t const * const test,
    tc_model const * const model,
    tc_ws * const ws);


#define tc_mae splatt_tc_mae
/**
* @brief Compute the MAE against of a factorization against a test tensor.
*        MAE is defined as: (\sum fabs(test[i] - predict[i])) / nnz.
*
* @param test The tensor to test against.
* @param model The factorization the evaluate.
* @param ws Workspace to use (thread buffers are accessed).
*
* @return The RMSE.
*/
val_t tc_mae(
    sptensor_t const * const test,
    tc_model const * const model,
    tc_ws * const ws);


#define tc_loss_sq splatt_tc_loss_sq
/**
* @brief Compute the sum-of-squared loss of a factorization. Computes:
*           sum((observed-predicted)^2) over all nonzeros in 'test'.
*
* @param test The tensor to test against.
* @param model The factorization the evaluate.
* @param ws Workspace to use (thread buffers are accessed).
*
* @return The loss.
*/
val_t tc_loss_sq(
    sptensor_t const * const test,
    tc_model const * const model,
    tc_ws * const ws);


#define tc_frob_sq splatt_tc_frob_sq
/**
* @brief Compute the sum of the squared Frobenius norms of a model, including
*        regularization penalties.
*
* @param model The factorization to evaluate.
* @param ws Workspace to use (regularization[:] are accessed).
*
* @return \sum_{m=1}^{M} \lambda_m || A^{(m)} ||_F_2.
*/
val_t tc_frob_sq(
    tc_model const * const model,
    tc_ws const * const ws);



#define predict_val splatt_predict_val
/**
* @brief Predict a value of the nonzero in position 'index'.
*
* @param model The model to use for prediction.
* @param test The sparse tensor to test against.
* @param index The index of the nonzero to predict. test->ind[:][index] used.
* @param buffer A buffer at least of size model->rank.
*
* @return The predicted value.
*/
val_t tc_predict_val(
    tc_model const * const model,
    sptensor_t const * const test,
    idx_t const index,
    val_t * const restrict buffer);


#define tc_predict_val_col splatt_tc_predict_val_col
/**
* @brief Predict a value of the nonzero in position 'index', when the model is
*        stored column-major.
*
* @param model The column-major model to use for prediction.
* @param test The sparse tensor to test against.
* @param index The index of the nonzero to predict. test->ind[:][index] used.
* @param buffer A buffer at least of size model->rank.
*
* @return The predicted value.
*/
val_t tc_predict_val_col(
    tc_model const * const model,
    sptensor_t const * const test,
    idx_t const index,
    val_t * const restrict buffer);



#define tc_model_alloc splatt_tc_model_alloc
/**
* @brief Allocate and randomly initialize a model for tensor completion.
*
* @param train The training data, used for dimensionality of the model.
* @param rank The sought rank of the factorization.
* @param which Which factorization algorithm will be used.
*
* @return An allocated model for tensor completion.
*/
tc_model * tc_model_alloc(
    sptensor_t const * const train,
    idx_t const rank,
    splatt_tc_type const which);


#define tc_model_copy splatt_tc_model_copy
/**
* @brief Allocate and copy a model.
*
* @param model The model to copy.
*
* @return A deep copy of model.
*/
tc_model * tc_model_copy(
    tc_model const * const model);


#define tc_model_free splatt_tc_model_free
/**
* @brief Free the memory allocated for a tensor completion model.
*
* @param model The model to free.
*/
void tc_model_free(
    tc_model * model);



#define tc_converge splatt_tc_converge
/**
* @brief Print progress statistics and determine if a model has converged.
*        Convergence is detected when the RMSE on the validation set has not
*        improved more than 'ws->tolerance' in 'ws->max_badepochs' epochs.
*
*        If improvement is detected, this routine will store the new best model
*        and resulting RMSE.
*
*        Note that this routine does not calculate the objective function for
*        you. Some optimization functions (e.g., gradient descent) compute the
*        objective during training (line search, etc.) and so we save
*        computations by not recomputing those values.
*
* @param train The training set.
* @param validate The validation set.
* @param model The model we are checking.
* @param loss The computed loss of model.
* @param frobsq The summed squared frobenius of the factors.
* @param epoch Which epoch we are on (for printing).
* @param ws Workspace to use (for storing statistics).
*
* @return True if converged, false otherwise.
*/
bool tc_converge(
    sptensor_t const * const train,
    sptensor_t const * const validate,
    tc_model const * const model,
    val_t const loss,
    val_t const frobsq,
    idx_t const epoch,
    tc_ws * const ws);



#ifdef SPLATT_USE_MPI

int mpi_tc_distribute_coarse(
    char const * const train_fname,
    char const * const validate_fname,
    idx_t const * const dims,
    sptensor_t * * train_out,
    sptensor_t * * validate_out,
    rank_info * const rinfo);


int mpi_tc_distribute_med(
    char const * const train_fname,
    char const * const validate_fname,
    idx_t const * const dims,
    sptensor_t * * train_out,
    sptensor_t * * validate_out,
    rank_info * const rinfo);


tc_model * mpi_tc_model_alloc(
    sptensor_t const * const train,
    idx_t const rank,
    splatt_tc_type const which,
    permutation_t const * const perm,
    rank_info * const rinfo);

#endif

#endif
