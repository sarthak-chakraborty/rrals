
#include "completion.h"
#include "../util.h"
#include "../reorder.h"         /* `shuffle_idx` needed for random sampling */

#include "../io.h"
#include "../ccp/ccp.h"
#include "../sort.h"

#include <math.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>

/* TODO: Conditionally include this OR define lapack prototypes below?
 *       What does this offer beyond prototypes? Can we detect at compile time
 *       if we are using MKL vs ATLAS, etc.?
 */
//#include <mkl.h>


/* Use hardcoded 3-mode kernels when possible. Results in small speedups. */
#ifndef USE_3MODE_OPT
#define USE_3MODE_OPT 0
#endif


/******************************************************************************
 * LAPACK PROTOTYPES
 *****************************************************************************/

/*
 * TODO: Can this be done in a better way?
 */

#if   SPLATT_VAL_TYPEWIDTH == 32
  void spotrf_(char *, int *, float *, int *, int *);
  void spotrs_(char *, int *, int *, float *, int *, float *, int *, int *);
  void ssyrk_(char *, char *, int *, int *, char *, char *, int *, char *, char *, int *);

  #define LAPACK_DPOTRF spotrf_
  #define LAPACK_DPOTRS spotrs_
  #define LAPACK_DSYRK ssyrk_
#else
  void dpotrf_(char *, int *, double *, int *, int *);
  void dpotrs_(char *, int *, int *, double *, int *, double *, int *, int *);
  void dsyrk_(char *, char *, int *, int *, double *, double *, int *, double *, double *, int *);

  #define LAPACK_DPOTRF dpotrf_
  #define LAPACK_DPOTRS dpotrs_
  #define LAPACK_DSYRK dsyrk_
#endif


/******************************************************************************
 * Slice-COO (S-COO) data structure
 * - used in HyperTensor [Kaya & Ucar 2016] and GenTen [Phipps & Kolda 2018]
 * - I made up the name and will ping Oguz/Eric/Tammy for their preference.
 * - This should be moved to its own file in src/ so we can use it in other
 *   kernels later.
 *****************************************************************************/
typedef struct
{
  sptensor_t * coo; /* the actual tensor data */

  /* We just pull these out of coo to save some typing */
  idx_t nnz;
  idx_t nmodes;
  idx_t dims[MAX_NMODES];

  /* Equivalent to row_ptr of CSR. Points into slice_nnz */
  idx_t * slice_ptr[MAX_NMODES];

  /* Each is a list of length nnz and indexes into coo->inds and coo->vals */
  idx_t * slice_nnz[MAX_NMODES];
} scoo_t;


scoo_t * scoo_alloc(
  sptensor_t const * const coo)
{
  scoo_t * scoo = splatt_malloc(sizeof(*scoo));

  /* Deep copy the COO tensor.
   * TODO: support shallow copies? */
  scoo->coo = tt_copy(coo);

  scoo->nnz = coo->nnz;
  scoo->nmodes = coo->nmodes;

  for(idx_t m=0; m < coo->nmodes; ++m) {
    scoo->dims[m] = coo->dims[m];

    /* first count the nnz per slice and do an exclusive prefix sum */
    idx_t * slice_counts = tt_get_hist(scoo->coo, m);
    prefix_sum_exc(slice_counts, scoo->dims[m]);

    /* now go over all non-zeros and store their locations in slice_nnz */
    scoo->slice_nnz[m] = splatt_malloc(scoo->nnz * sizeof(**scoo->slice_nnz));
    for(idx_t n=0; n < scoo->nnz; ++n) {
      idx_t const slice = coo->ind[m][n];
      idx_t const ptr = slice_counts[slice]++;
      scoo->slice_nnz[m][ptr] = n;
    }

    /* now right-shift slice_counts into slice_ptrs to turn it back into an
     * exclusive prefix (row_ptr) */
    scoo->slice_ptr[m] = splatt_malloc(
        (scoo->dims[m]+1) * sizeof(**scoo->slice_ptr));
    scoo->slice_ptr[m][0] = 0;
    for(idx_t i=1; i < scoo->dims[m] + 1; ++i) {
      scoo->slice_ptr[m][i] = slice_counts[i-1];
    }
    splatt_free(slice_counts);
  } /* foreach mode */


  /* TODO: move these to unit tests */
  for(idx_t m=0; m < scoo->nmodes; ++m) {
    idx_t nnz = 0;
    for(idx_t i=0; i < scoo->dims[m]; ++i) {
      idx_t index;
      for(index = scoo->slice_ptr[m][i];
          index < scoo->slice_ptr[m][i+1];
          ++index) {

        ++nnz;

        idx_t const ptr = scoo->slice_nnz[m][index];
        idx_t const mode_j = scoo->coo->ind[m][ptr];
        assert(scoo->coo->ind[m][ptr] == i);
      }
    }
    assert(nnz == scoo->nnz);
  }

  return scoo;
}


void scoo_free(
  scoo_t * scoo)
{
  if(scoo == NULL) {
    return;
  }
  for(idx_t m=0; m < scoo->nmodes; ++m) {
    splatt_free(scoo->slice_ptr[m]);
    splatt_free(scoo->slice_nnz[m]);
  }
  tt_free(scoo->coo);
  splatt_free(scoo);
}


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/



static inline void p_add_hada_clear(
  val_t * const restrict accum,
  val_t * const restrict toclear,
  val_t const * const restrict b,
  idx_t const nfactors)
{
  for(idx_t f=0; f < nfactors; ++f) {
    accum[f] += toclear[f] * b[f];
    toclear[f] = 0;
  }
}


/**
* @brief Compute the Cholesky decomposition of the normal equations and solve
*        for out_row. We only compute the upper-triangular portion of 'neqs',
*        so work with the lower-triangular portion when column-major
*        (for Fortran).
*
* @param neqs The NxN normal equations.
* @param[out] out_row The RHS of the equation. Updated in place.
* @param N The rank of the problem.
*/
static inline void p_invert_row(
    val_t * const restrict neqs,
    val_t * const restrict out_row,
    idx_t const N)
{
  char uplo = 'L';
  int order = (int) N;
  int lda = (int) N;
  int info;
  LAPACK_DPOTRF(&uplo, &order, neqs, &lda, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DPOTRF returned %d\n", info);
  }


  int nrhs = 1;
  int ldb = (int) N;
  LAPACK_DPOTRS(&uplo, &order, &nrhs, neqs, &lda, out_row, &ldb, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DPOTRS returned %d\n", info);
  }
}



/**
* @brief Compute DSYRK: out += A^T * A, a rank-k update. Only compute
*        the upper-triangular portion.
*
* @param A The input row(s) to update with.
* @param N The length of 'A'.
* @param nvecs The number of rows in 'A'.
* @param nflush Then number of times this has been performed (this slice).
* @param[out] out The NxN matrix to update.
*/
static inline void p_vec_oprod(
		val_t * const restrict A,
    idx_t const N,
    idx_t const nvecs,
    idx_t const nflush,
    val_t * const restrict out)
{
  char uplo = 'L';
  char trans = 'N';
  int order = (int) N;
  int k = (int) nvecs;
  int lda = (int) N;
  int ldc = (int) N;
  val_t alpha = 1;
  val_t beta = (nflush == 0) ? 0. : 1.;
  LAPACK_DSYRK(&uplo, &trans, &order, &k, &alpha, A, &lda, &beta, out, &ldc);
}



/*
 * RRALS - these are the permutation vectors for each thread.
 * XXX: throw these into a workspace structure or something else not global...
 */
// static idx_t const MAX_THREADS = 1024;


static int findCeil(val_t S[], val_t r, int l, int h){
	int mid;
    while (l < h)
    {
         mid = l + ((h - l) >> 1);
        (r > S[mid]) ? (l = mid + 1) : (h = mid);
    }
    return (S[l] >= r) ? l : 0;
}


#ifndef RRALS_MAX_THREADS
#define RRALS_MAX_THREADS 1024
#endif


static idx_t const PERM_INIT = 128;
// static idx_t perm_i_lengths[MAX_THREADS];
static idx_t perm_i_lengths[RRALS_MAX_THREADS];
// static idx_t * perm_i_global[MAX_THREADS];
static idx_t * perm_i_global[RRALS_MAX_THREADS];

/*
 * Each thread is given a random seed to use for sampling. We pad them to
 * ensure each falls on a different cache line (to avoid false sharing).
 */
static idx_t const SEED_PADDING = 16;
static unsigned int * sample_seeds = NULL;

static void p_process_slice(
    scoo_t const * const scoo,
    idx_t const slice_id,
    idx_t const mode,
    val_t * * mvals,
    idx_t const nfactors,
    val_t * const restrict out_row,
    val_t * const accum,
    val_t * const restrict neqs,
    val_t * const restrict neqs_buf,
    val_t * const neqs_buf_tree,
    idx_t * const nflush,
    val_t ** const lev_score,
    int alpha,
    float beta,
    int **act,
    int **frac,
    int **same,
    double *sampling_time,
    double *mttkrp_time)
{

  struct timeval start_t, stop_t, start_tt, stop_tt;

  idx_t const nmodes = scoo->nmodes;
  val_t const * const restrict vals = scoo->coo->vals;

  idx_t const * const restrict slice_ptr = scoo->slice_ptr[mode];
  idx_t const * const restrict slice_nnz = scoo->slice_nnz[mode];
  idx_t const * const * const inds = scoo->coo->ind;

  idx_t const slice_start = slice_ptr[slice_id];
  idx_t       slice_end   = slice_ptr[slice_id+1];
  idx_t const slice_size = slice_end - slice_start;


  // int *indicator = (int *)malloc((slice_end-slice_start) * sizeof(int));
  // for(int i=0; i < (slice_end-slice_start); i++)
  // 	indicator[i] = 0;

  gettimeofday(&start_tt, NULL);
  // Mode which must be chosen to compute MTTKRP
  idx_t *Modes = (idx_t *)malloc((nmodes-1)*sizeof(idx_t));
  int k=0;
  for(int m=0; m<nmodes; m++){
  	if(mode == m)
  		continue;
  	Modes[k++] = m;
  }

  // Initializing the vector containing leverage score corresponding to the nnz present in the slice
  // val_t sum=0.0;
  val_t *S_pdf = (val_t *)malloc((slice_end - slice_start) * sizeof(val_t));

  // Computing the vector by multiplying the leverage score for the other two modes
  val_t score;
  for(int i=slice_start; i < slice_end; i++){
  	idx_t nnz_index = slice_nnz[i];
  	score = lev_score[Modes[0]][scoo->coo->ind[Modes[0]][nnz_index]] * lev_score[Modes[1]][scoo->coo->ind[Modes[1]][nnz_index]];
  	S_pdf[i-slice_start] = score;
  	// sum += score;
  }

  // // Normalize the leverage score
  // for(int i=0; i<(slice_end-slice_start); i++)
  // 	S_pdf[i] /= sum;

  // // Convert pdf to cdf for easier sampling
  // val_t *S_cdf = (val_t *)malloc((slice_end - slice_start) * sizeof(val_t));
  // S_cdf[0] = S_pdf[0];
  // for(int i=1; i<(slice_end-slice_start); i++)
  // 	S_cdf[i] = S_cdf[i-1] + S_pdf[i];


  for(idx_t f=0; f < nfactors; ++f) {
    accum[f] = 0.;
  }

  /* buffer of rows to form normal equations */
  idx_t bufsize = 0;
  val_t * hada = neqs_buf; /* each row is a hadamard product */

  /* sampling buffers */
  int const tid = splatt_omp_get_thread_num();
  idx_t * perm_i = NULL;


  gettimeofday(&start_t, NULL);
  idx_t sample_threshold;
  if(mode == 0)
    sample_threshold = (alpha/1) * nfactors;
  else if(mode == 2)
  	sample_threshold = alpha * nfactors;
  else
    sample_threshold = alpha * nfactors;
  val_t const sample_rate = beta;
  int sample = 0;

  // if(mode == 1){
  	if(slice_size > sample_threshold) {
	    sample = 1;
	    /* realloc sample buffer if needed */
	    if(slice_size > perm_i_lengths[tid]) {
	      perm_i_lengths[tid] = slice_size;
	      splatt_free(perm_i_global[tid]);
	      perm_i_global[tid] = splatt_malloc(slice_size * sizeof(*perm_i_global));
	    }

	    /* fill buffer with indices and shuffle to get sampled nnz */
	    /* RRALS-TODO: can we intead just sample nnz_ptr[]? Or do an initial shuffle
	     * at the beginning of RRALS (or every few its) and instead just choose
	     * a rand starting index? Then proceed and process non-zeros
	     * [rand_start, (rand_start+sample_size) % end).
	     *
	     * Current implementation is still O(nnz) instead of O(sampled nnz). */
	    perm_i = perm_i_global[tid];
	    for(idx_t n=slice_start; n < slice_end; ++n) {
	      perm_i[n-slice_start] = n;
	    }
	    idx_t const my_sample_size = sample_threshold + ((slice_size-sample_threshold) / sample_rate);
	    idx_t const sample_size = SS_MIN(slice_size, my_sample_size);
	    // quick_shuffle(perm_i, sample_size, &(sample_seeds[tid * SEED_PADDING]));
	    quick_shuffle(perm_i, S_pdf, slice_size, sample_size, &(sample_seeds[tid * SEED_PADDING]));
	    slice_end = slice_start + sample_size;
	  }
	  gettimeofday(&stop_t, NULL);
	  *sampling_time += (stop_t.tv_sec + stop_t.tv_usec/1000000.0)- (start_t.tv_sec + start_t.tv_usec/1000000.0);
  // }
  

  /* store diagnostics */
  act[mode][slice_id] = slice_size;
  frac[mode][slice_id] = slice_end - slice_start;


  /* foreach nnz in slice */
  for(idx_t x = slice_start; x < slice_end; ++x) {
    /* initialize buffers */
    for(idx_t f=0; f < nfactors; ++f) {
      hada[f] = 1.;
    }


    /* which non-zero to process */
    idx_t nnz_ptr = slice_nnz[x];
    gettimeofday(&start_t, NULL);
    if(sample) {
    	// double max_lim = S_cdf[slice_end-slice_start-1] - S_cdf[0];
    	// val_t r = fmod((double)rand(), max_lim) + S_cdf[0];
    	// int indexc = findCeil(S_cdf, r, 0, (slice_end-slice_start-1));
    	// idx_t indexc = x - slice_start;
    	// indicator[indexc] += 1;
    	// nnz_ptr = slice_nnz[indexc];
      nnz_ptr = slice_nnz[perm_i[x - slice_start]];
    }
    gettimeofday(&stop_t, NULL);
    *sampling_time += (stop_t.tv_sec + stop_t.tv_usec/1000000.0)- (start_t.tv_sec + start_t.tv_usec/1000000.0);


    // ///////////////////////////////////////////////////////////////////////////
    // // Check how many same samples has been selected more than once
    // int same_nnz_sampled = 0;
    // for(int i=0; i<(slice_end-slice_start); i++){
    // 	if(indicator[i] > 0)
    // 		same_nnz_sampled += indicator[i]-1;
    // }
    // same[mode][slice_id] = same_nnz_sampled;
    // //////////////////////////////////////////////////////////////////////////


    /* compute hadamard product */
    for(idx_t m=0; m < nmodes; ++m) {
      if(m == mode) {
        continue;
      }
      idx_t const row_id = inds[m][nnz_ptr];
      val_t const * const restrict row = &(mvals[m][row_id * nfactors]);
      for(idx_t f=0; f < nfactors; ++f) {
        hada[f] *= row[f];
      }
    }
    /* accumulate MTTKRP */
    for(idx_t f=0; f < nfactors; ++f) {
      accum[f] += vals[nnz_ptr] * hada[f];
    }

    hada += nfactors;
    /* if buffer is full, flush and accumulate into neqs */
    if(++bufsize == ALS_BUFSIZE) {
      /* add to normal equations and reset hada */
      p_vec_oprod(neqs_buf, nfactors, bufsize, (*nflush)++, neqs);
      bufsize = 0;
      hada = neqs_buf;
    } 

    /* store mttkrp result in RHS of linear system */
    for(idx_t f=0; f < nfactors; ++f) {
      out_row[f] += accum[f];
      accum[f] = 0.;
    }
  }
  gettimeofday(&stop_tt, NULL);
  *mttkrp_time += (stop_tt.tv_sec + stop_tt.tv_usec/1000000.0)- (start_tt.tv_sec + start_tt.tv_usec/1000000.0);

  free(S_pdf);
  // free(S_cdf);
  free(Modes);

  /* flush and accumulate into neqs */
  p_vec_oprod(neqs_buf, nfactors, bufsize, (*nflush)++, neqs);
}





/**
* @brief Compute the i-ith row of the MTTKRP, form the normal equations, and
*        store the new row.
*
* @param scoo The tensor of training data.
* @param slice_id The row to update.
* @param reg Regularization parameter for the i-th row.
* @param model The model to update
* @param ws Workspace.
* @param tid OpenMP thread id.
*/
static void p_update_slice(
    scoo_t const * const scoo,
    idx_t const mode,
    idx_t const slice_id,
    val_t const reg,
    tc_model * const model,
    tc_ws * const ws,
    int const tid,
    val_t ** const lev_score,
    int alpha,
    float beta,
    int **act,
    int **frac,
    int **same,
    double *solving_time,
    double *sampling_time,
    double *mttkrp_time)
{

  struct timeval start, stop;

  idx_t const nmodes = scoo->nmodes;
  idx_t const nfactors = model->rank;

  /* fid is the row we are actually updating */
#ifdef SPLATT_USE_MPI
  assert(slice_id < model->globmats[mode]->I);
  val_t * const restrict out_row = model->globmats[mode]->vals +
      (slice_id * nfactors);
#else
  val_t * const restrict out_row = model->factors[mode] +
      (slice_id * nfactors);
#endif
  val_t * const restrict accum = ws->thds[tid].scratch[1];
  val_t * const restrict neqs  = ws->thds[tid].scratch[2];

  idx_t bufsize = 0; /* how many hada vecs are in mat_accum */
  idx_t nflush = 0;  /* how many times we have flushed to add to the neqs */
  val_t * const restrict mat_accum  = ws->thds[tid].scratch[3];

  val_t * hada = mat_accum;
  val_t * const restrict hada_accum  = ws->thds[tid].scratch[4];

  /* clear out buffers */
  for(idx_t m=0; m < nmodes; ++m) {
    for(idx_t f=0; f < nfactors; ++f) {
      accum[f + (m*nfactors)] = 0.;
    }
    for(idx_t f=0; f < nfactors; ++f) {
      hada_accum[f + (m*nfactors)] = 0.;
    }
  }
  for(idx_t f=0; f < nfactors; ++f) {
    out_row[f] = 0;
  }

  /* grab factors */
  val_t * mats[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = model->factors[m];
  }

  /* do MTTKRP + dsyrk */
  p_process_slice(scoo, slice_id, mode, mats, nfactors, out_row, accum, neqs,
      mat_accum, hada_accum, &nflush, lev_score, alpha, beta, act, frac, same, sampling_time, mttkrp_time);

  gettimeofday(&start, NULL);
  /* add regularization to the diagonal */
  for(idx_t f=0; f < nfactors; ++f) {
    neqs[f + (f * nfactors)] += reg;
  }

  /* solve! */
  p_invert_row(neqs, out_row, nfactors);
  gettimeofday(&stop, NULL);
  *solving_time += (stop.tv_sec + stop.tv_usec/1000000.0) - (start.tv_sec + start.tv_usec/1000000.0);
}



/**
* @brief Update factor[m] which follows a dense mode. This function should be
*        called from inside an OpenMP parallel region!
*
* @param csf The CSF tensor array. csf[m] is a tiled tensor.
* @param m The mode we are updating.
* @param model The current model.
* @param ws Workspace info.
* @param thd_densefactors Thread structures for the dense mode.
* @param tid Thread ID.
*/
static void p_densemode_als_update(
    splatt_csf const * const csf,
    idx_t const m,
    tc_model * const model,
    tc_ws * const ws,
    thd_info * const thd_densefactors,
    int const tid)
{
  idx_t const rank = model->rank;

  /* master thread writes/aggregates directly to the model */
  #pragma omp master
#ifdef SPLATT_USE_MPI
  SPLATT_VPTR_SWAP(thd_densefactors[0].scratch[0], model->globmats[m]->vals);

  idx_t const dense_slices = model->globmats[m]->I;
#else
  SPLATT_VPTR_SWAP(thd_densefactors[0].scratch[0], model->factors[m]);

  idx_t const dense_slices = model->dims[m];
#endif

  /* TODO: this could be better by instead only initializing neqs with beta=0
   * and keeping track of which have been updated. */
  memset(thd_densefactors[tid].scratch[0], 0,
      dense_slices * rank * sizeof(val_t));
  memset(thd_densefactors[tid].scratch[1], 0,
      dense_slices * rank * rank * sizeof(val_t));

  #pragma omp barrier

  /* XXX S-COO needs something here */
#if 0
  /* update each tile in parallel */
  #pragma omp for schedule(dynamic, 1)
  for(idx_t tile=0; tile < csf[m].ntiles; ++tile) {
    p_process_tile(csf+m, tile, model, ws, thd_densefactors, tid);
  }
#endif

  /* aggregate partial products */
  thd_reduce(thd_densefactors, 0,
      dense_slices * rank, REDUCE_SUM);

  /* TODO: this could be better by using a custom reduction which only
   * operates on the upper triangular portion. OpenMP 4 declare reduction
   * would be good here? */
  thd_reduce(thd_densefactors, 1,
      dense_slices * rank * rank, REDUCE_SUM);

  /* save result to model */
  #pragma omp master
#ifdef SPLATT_USE_MPI
  SPLATT_VPTR_SWAP(thd_densefactors[0].scratch[0], model->globmats[m]->vals);
#else
  SPLATT_VPTR_SWAP(thd_densefactors[0].scratch[0], model->factors[m]);
#endif

  #pragma omp barrier

  /* do all of the Cholesky factorizations */
#ifdef SPLATT_USE_MPI
  val_t * const restrict out  = model->globmats[m]->vals;
#else
  val_t * const restrict out  = model->factors[m];
#endif
  val_t const reg = ws->regularization[m];
  #pragma omp for schedule(static, 1)
  for(idx_t i=0; i < dense_slices; ++i) {
    val_t * const restrict neqs_i =
        (val_t *) thd_densefactors[0].scratch[1] + (i * rank * rank);
    /* add regularization */
    for(idx_t f=0; f < rank; ++f) {
      neqs_i[f + (f * rank)] += reg;
    }

    /* Cholesky + solve */
    p_invert_row(neqs_i, out + (i * rank), rank);
  }
}



static val_t *getGram(val_t *A, idx_t nrows, idx_t rank){
  val_t *gram = (val_t *)malloc((rank*rank) * sizeof(val_t));

  val_t sum;
  for(int i=0; i<rank; i++){
    for(int j=0; j<rank; j++){
      sum = 0;
      for(int k=0; k<nrows; k++){
        sum += A[i + k*rank]*A[j + k*rank];
      }
      gram[j + i*rank] = sum;
    }
  }

  return gram;


  // val_t *gram = (val_t *)malloc((rank*rank) * sizeof(val_t));

  // for(int i=0; i<rank; i++)
  //   for(int j=0; j<nrows; j++)
  //     for(int k=0; k<rank; k++)
  //     	gram[i*rank + k] += A[j*rank + i] * A[j*rank + k];
  

  // return gram;

  // val_t *gram = (val_t *)malloc((rank*rank) * sizeof(val_t));

  // val_t sum;
  // for(int i=0; i<rank; i++){
  //   for(int j=0; j<rank; j++){
  //     sum = 0;
  //     for(int k=0; k<nrows; k++){
  //       sum += A[i*nrows + k]*A[j*nrows + k];
  //     }
  //     gram[j + i*rank] = sum;
  //   }
  // }

  // return gram;
}


static val_t *GramInv(val_t *A, idx_t N){
  // int *IPIV = (int *)malloc((N+1) * sizeof(int));
  // int LWORK = N*N;
  // double *WORK = (double *)malloc(LWORK * sizeof(double));
  // int INFO;

  char uplo = 'L';
  int order = (int)N;
  int lda = (int)N;
  int info;

  LAPACK_DPOTRF(&uplo, &order, A, &lda, &info);
  // if(info){
  // 	printf("info dpotrf = %d\n", info);
  // }

  dpotri_(&uplo, &order, A, &lda, &info);
  // if(info){
  // 	printf("info dpotri = %d\n", info);
  // }
  // dgetrf_(&N,&N,A,&N,IPIV,&INFO);
  // if(INFO != 0)
  // 	printf("INFO after dgetrf = %d\n",INFO);

  // dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);
  // if(INFO != 0)
  // 	printf("INFO after dgetri = %d\n",INFO);

  // free(IPIV);
  // free(WORK);

  return A;
}


// static void getLvrgScore(val_t *A, val_t *gram, val_t **lev_score, idx_t rank, idx_t nrows, int factor){
// 	// double * I = (double *)malloc((nrows*rank) * sizeof(double));
// 	// // double * J = (double *)malloc((nrows*nrows) * sizeof(double));

// 	// val_t sum = 0.0;
// 	// for(int i=0; i<nrows; i++){
// 	// 	for(int j=0; j<rank; j++){
// 	// 		for(int k=0; k<rank; k++)
// 	// 			sum += A[i*rank + k] * gram[k*rank + j];

// 	// 		I[i*rank + j] = (double)sum;
// 	// 		sum = 0.0;
// 	// 	}
// 	// }

// 	// sum = 0.0;
// 	// for(int i=0; i<nrows; i++){
// 	// 	// for(int j=0; j<nrows; j++){
// 	// 		for(int k=0; k<rank; k++)
// 	// 			sum += I[i*rank + k] * A[i*rank + k];

// 	// 		lev_score[factor][i] = sum;
// 	// 		// J[i*nrows + j] = (double)sum;
// 	// 		sum = 0.0;
// 	// 	// }
// 	// }

// 	//  // for(int i=0; i<nrows; i++)
// 	//  // 	lev_score[factor][i] = J[i*nrows + i];

// 	// free(I);
// 	// // free(J); 

//   for (int i=0; i<nrows; ++i){
//     for (int j1=0; j1<rank; j1++){
//       for (int j2=0; j2<rank; j2++){
//         lev_score[factor][i] += A[i*rank + j1] * gram[j1*rank + j2] * A[i*rank + j2];
//       }
//     }
//   }
// }



static void getLvrgScore(val_t * A, val_t **lev_score, idx_t rank, idx_t nrows, int factor){
	char jobu = 'S';
	char jobvt = 'N';
	int m = (int)nrows;
	int n = (int)rank;
	int lda = m;
	int ldu = m;
	int ldvt = n;
	double *S = (double *)malloc(n * sizeof(double));
	double *U = (double *)malloc((ldu*n) * sizeof(double));
	double *VT;
	int lwork = -1;
	double wkopt;
	double *work;
	int info;

	double *a = (double *)malloc((nrows*rank) * sizeof(double));
	for(int i=0; i<nrows*rank; i++)
		a[i] = A[i];

	// Query and allocate appropriate workspace
	dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, S, U, &ldu, VT, &ldvt, &wkopt, &lwork, &info);
	if(info)
		printf("info return %d\n",info);
    lwork = (int)wkopt;
    work = (double *)malloc(lwork*sizeof(double));

    /* Compute SVD */
    dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, S, U, &ldu, VT, &ldvt, work, &lwork, &info);
    if(info)
		printf("info return %d\n",info);

    for(int i=0; i<nrows; i++){
    	val_t sum = 0.0;
    	for(int j=0; j<rank; j++)
    		sum += U[i + j*nrows] * U[i + j*nrows];
    	sum = sqrt((double)sum);
    	lev_score[factor][i] = sum;
    }
}



#ifdef SPLATT_USE_MPI

static void p_update_factor_all2all(
    tc_model * const model,
    tc_ws * const ws,
    idx_t const mode)
{
  rank_info * const rinfo = ws->rinfo;
  idx_t const m = mode;

  idx_t const nfactors = model->rank;

  idx_t const nglobrows = model->globmats[m]->I;
  val_t const * const restrict gmatv = model->globmats[m]->vals;

  /* ensure local info is up to date */
  assert(rinfo->ownstart[m] + rinfo->nowned[m] <= model->dims[m]);
  val_t * const restrict matv = model->factors[m];
  par_memcpy(matv + (rinfo->ownstart[m] * nfactors), gmatv,
      rinfo->nowned[m] * nfactors * sizeof(*matv));

  if(rinfo->layer_size[mode] == 1) {
    return;
  }

  /* first prepare all values that I own and need to send */
  idx_t const mat_start = rinfo->mat_start[m];
  idx_t const * const restrict nbr2globs_inds = rinfo->nbr2globs_inds[m];
  idx_t const * const restrict local2nbr_inds = rinfo->local2nbr_inds[m];

  idx_t const nsends = rinfo->nnbr2globs[m];
  idx_t const nrecvs = rinfo->nlocal2nbr[m];

  val_t * const restrict nbr2globs_buf = ws->nbr2globs_buf;
  val_t * const restrict nbr2local_buf = ws->local2nbr_buf;

  /* fill send buffer */
  #pragma omp for
  for(idx_t s=0; s < nsends; ++s) {
    assert(nbr2globs_inds[s] >= mat_start);
    idx_t const row = nbr2globs_inds[s] - mat_start;
    val_t * const restrict buf_row = nbr2globs_buf + (s * nfactors);
    val_t const * const restrict gmat_row = gmatv + (row * nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      buf_row[f] = gmat_row[f];
    }
  }

  /* exchange entries */
  #pragma omp master
  {
    /* grab ptr/disp from rinfo. nbr2local and local2nbr will have the same
     * structure so we just reuse those */
    int const * const restrict nbr2globs_ptr = rinfo->nbr2globs_ptr[m];
    int const * const restrict nbr2local_ptr = rinfo->local2nbr_ptr[m];
    int const * const restrict nbr2globs_disp = rinfo->nbr2globs_disp[m];
    int const * const restrict nbr2local_disp = rinfo->local2nbr_disp[m];

    timer_start(&timers[TIMER_MPI_COMM]);
    MPI_Alltoallv(nbr2globs_buf, nbr2globs_ptr, nbr2globs_disp, SPLATT_MPI_VAL,
                  nbr2local_buf, nbr2local_ptr, nbr2local_disp, SPLATT_MPI_VAL,
                  rinfo->layer_comm[m]);
    timer_stop(&timers[TIMER_MPI_COMM]);
  }
  #pragma omp barrier


  /* now write incoming values to my local matrix */
  #pragma omp for
  for(idx_t r=0; r < nrecvs; ++r) {
    idx_t const row = local2nbr_inds[r];
    assert(row < rinfo->ownstart[m] || row >= rinfo->ownend[m]);
    val_t * const restrict mat_row = matv + (row * nfactors);
    val_t const * const restrict buf_row = nbr2local_buf + (r * nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      mat_row[f] = buf_row[f];
    }
  }
}


static void p_init_mpi(
    sptensor_t const * const train,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t maxlocal2nbr = 0;
  idx_t maxnbr2globs = 0;

  for(idx_t m=0; m < train->nmodes; ++m) {
    maxlocal2nbr = SS_MAX(maxlocal2nbr, ws->rinfo->nlocal2nbr[m]);
    maxnbr2globs = SS_MAX(maxnbr2globs, ws->rinfo->nnbr2globs[m]);
  }

  ws->local2nbr_buf  = splatt_malloc(model->rank*maxlocal2nbr * sizeof(val_t));
  ws->nbr2globs_buf  = splatt_malloc(model->rank*maxnbr2globs * sizeof(val_t));

  /* get initial factors */
  for(idx_t m=0; m < train->nmodes; ++m) {
    p_update_factor_all2all(model, ws, m);
  }
  timer_reset(&timers[TIMER_MPI_COMM]);
}



#endif



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


void splatt_tc_rrals(
    sptensor_t * train,
    sptensor_t * const validate,
    tc_model * const model,
    tc_ws * const ws,
    int alpha,
    float beta)
{
  idx_t const nmodes = train->nmodes;
  idx_t const nfactors = model->rank;

#ifdef SPLATT_USE_MPI
  rank_info * rinfo = ws->rinfo;
  int const rank = rinfo->rank;
#else
  int const rank = 0;
#endif

  // if(rank == 0) {
  //   printf("BUFSIZE=%d\n", ALS_BUFSIZE);
  //   printf("USE_3MODE_OPT=%d\n", USE_3MODE_OPT);
  // }

  /* XXX: temporarily disabling dense mode replication */
  ws->num_dense = 0;
  for(idx_t m=0; m < nmodes; ++m ){
    ws->isdense[m] = 0;
  }

  /* store dense modes redundantly among threads */
  thd_info * thd_densefactors = NULL;
  if(ws->num_dense > 0) {
    thd_densefactors = thd_init(ws->nthreads, 3,
        ws->maxdense_dim * nfactors * sizeof(val_t), /* accum */
        ws->maxdense_dim * nfactors * nfactors * sizeof(val_t), /* neqs */
        ws->maxdense_dim * sizeof(int)); /* nflush */


    // if(rank == 0) {
    //   printf("REPLICATING MODES:");
    //   for(idx_t m=0; m < nmodes; ++m) {
    //     if(ws->isdense[m]) {
    //       printf(" %"SPLATT_PF_IDX, m+1);
    //     }
    //   }
    //   printf("\n\n");
    // }
  }

  /* load-balanced partition each mode for threads */
  idx_t * parts[MAX_NMODES];

  /* Allocate a slice-indexed COO tensor. */
  scoo_t * scoo = scoo_alloc(train);

#ifdef SPLATT_USE_MPI
  sptensor_t * both = NULL;
  if(validate != NULL) {
    both = tt_union(train, validate);
  }
  for(idx_t m=0; m < nmodes; ++m) {
    /* setup communication structures */
    mpi_find_owned(train, m, rinfo);
    if(validate != NULL) {
      mpi_compute_ineed(rinfo, both, m, nfactors, 1);
    } else {
      mpi_compute_ineed(rinfo, train, m, nfactors, 1);
    }
  }
  if(validate != NULL) {
    tt_free(both);
  }
#endif

  for(idx_t m=0; m < nmodes; ++m) {
#ifdef SPLATT_USE_MPI
    /* tt has more nonzeros than any of the modes actually need, so we need
     * to filter them first. */
    sptensor_t * tt_filtered = mpi_filter_tt_1d(train, m,
        rinfo->mat_start[m], rinfo->mat_end[m]);
    assert(tt_filtered->dims[m] == rinfo->mat_end[m] - rinfo->mat_start[m]);
    assert(train->indmap[m] == NULL);
    assert(tt_filtered->indmap[m] == NULL);
#endif

#if 0
    if(ws->isdense[m]) {
      /* standard CSF allocation for sparse modes */
      opts[SPLATT_OPTION_TILE] = SPLATT_DENSETILE;
      opts[SPLATT_OPTION_TILEDEPTH] = 1; /* don't tile dense mode */
#ifdef SPLATT_USE_MPI
      csf_alloc_mode(tt_filtered, CSF_SORTED_MINUSONE, m, csf+m, opts);
#else
      csf_alloc_mode(train, CSF_SORTED_MINUSONE, m, csf+m, opts);
#endif

      parts[m] = NULL;

    } else {
      /* standard CSF allocation for sparse modes */
      opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
#ifdef SPLATT_USE_MPI
      csf_alloc_mode(tt_filtered, CSF_SORTED_MINUSONE, m, csf+m, opts);
#else
      csf_alloc_mode(train, CSF_SORTED_MINUSONE, m, csf+m, opts);
#endif

      parts[m] = csf_partition_1d(csf+m, 0, ws->nthreads);
    }
#endif


#ifdef SPLATT_USE_MPI
    tt_free(tt_filtered);
#ifdef SPLATT_DEBUG
    /* sanity check on nnz */
    idx_t totnnz;
    MPI_Allreduce(&(csf[m].nnz), &totnnz, 1, SPLATT_MPI_IDX, MPI_SUM,
        rinfo->comm_3d);
    assert(totnnz == rinfo->global_nnz);
#endif
#endif
  } /* foreach mode */

#ifdef SPLATT_USE_MPI
  p_init_mpi(train, model, ws);


  /* TERRIBLE HACK for loss computation */
  sptensor_t * train_back = train;
  sptensor_t * tt_filter = mpi_filter_tt_1d(train, 0, rinfo->mat_start[0],
      rinfo->mat_end[0]);
  #pragma omp parallel for
  for(idx_t n=0; n < tt_filter->nnz; ++n) {
    tt_filter->ind[0][n] += rinfo->mat_start[0];
  }
  train = tt_filter;
#endif

  // if(rank == 0) {
  //   printf("\n");
  // }

  sample_seeds = splatt_malloc(
      splatt_omp_get_max_threads() * SEED_PADDING * sizeof(*sample_seeds));

  #pragma omp parallel
  {
    int const tid = splatt_omp_get_thread_num();
    perm_i_lengths[tid] = PERM_INIT;
    perm_i_global[tid] = splatt_malloc(PERM_INIT * sizeof(*perm_i_global));
    sample_seeds[tid * SEED_PADDING] = tid;
  }


  val_t loss = tc_loss_sq(train, model, ws);
  val_t frobsq = tc_frob_sq(model, ws);
  tc_converge(train, validate, model, loss, frobsq, 0, ws);

  

  FILE *f_act = fopen("Actual.csv", "w");
  FILE *f_frac = fopen("Fraction.csv", "w");
  FILE *f_time = fopen("Time.csv", "w");
  FILE *f_lev = fopen("Leverage.csv", "w");
  // FILE *f_same = fopen("Same.csv", "w");

  int **act = (int **)malloc(nmodes*sizeof(int *));
  for(int i=0; i<nmodes; i++){
    act[i] = (int *)malloc((scoo->dims[i])*sizeof(int));
  }

  int **frac = (int **)malloc(nmodes*sizeof(int *));
  for(int i=0; i<nmodes; i++){
    frac[i] = (int *)malloc((scoo->dims[i])*sizeof(int));
  }

  int **same = (int **)malloc(nmodes*sizeof(int *));
  for(int i=0; i<nmodes; i++){
    same[i] = (int *)malloc((scoo->dims[i])*sizeof(int));
  }

  double **time_slice = (double **)malloc(nmodes*sizeof(double *));
  for(int i=0; i<nmodes; i++){
    time_slice[i] = (double *)malloc((scoo->dims[i])*sizeof(double));
  }


  double avg_sampling_time[3] = {0.0, 0.0, 0.0};
  double avg_solving_time[3] = {0.0, 0.0, 0.0};
  double avg_mttkrp_time[3] = {0.0, 0.0, 0.0};
  double avg_tot_time[3] = {0.0, 0.0, 0.0};
  int count = 0;

  double avg_gram_time = 0.0;
  sp_timer_t gram_timer;


  sp_timer_t mode_timer;
  timer_reset(&mode_timer);
  timer_start(&ws->tc_time);


  val_t **lev_score = (val_t **)malloc(nmodes * sizeof(val_t *));
  for(int i=0; i<nmodes; i++)
    lev_score[i] = (val_t *)malloc((model->dims[i])*sizeof(val_t));


  for(idx_t e=1; e < ws->max_its+1; ++e) {
    count++;

    timer_fstart(&gram_timer);
    for(int i=0; i<nmodes; i++){
      // val_t *gram = getGram(model->factors[i], model->dims[i], model->rank);
      // gram = GramInv(gram, model->rank);

      getLvrgScore(model->factors[i], lev_score, model->rank, model->dims[i], i);
    }
    timer_stop(&gram_timer);
    avg_gram_time += gram_timer.seconds;


    #pragma omp parallel
    {
      int const tid = splatt_omp_get_thread_num();

      for(idx_t m=0; m < nmodes; ++m) {
        double solving_time = 0.0;
        double sampling_time = 0.0;
        double mttkrp_time = 0.0;

        #pragma omp master
        timer_fstart(&mode_timer);


        if(ws->isdense[m]) {
          /* XXX */
          //p_densemode_als_update(csf, m, model, ws, thd_densefactors, tid);

        /* dense modes are easy */
        } else {
          /* update each row in parallel */
          /* RRALS-TODO: we can maybe statically load balance this loop using
           * CCP (chains-on-chains partitioning) by using ccp.c:partition_1d()
           */
          #pragma omp for schedule(dynamic, 8) nowait
          for(idx_t i=0; i < scoo->dims[m]; ++i)  {
            struct timeval start_t, stop_t;
            gettimeofday(&start_t, NULL);
            p_update_slice(scoo, m, i, ws->regularization[m], model, ws, tid, lev_score, alpha, beta, act, frac, same, &solving_time, &sampling_time, &mttkrp_time);
            gettimeofday(&stop_t, NULL);
            time_slice[m][i] = (stop_t.tv_sec + stop_t.tv_usec/1000000.0) - (start_t.tv_sec + start_t.tv_usec/1000000.0);
          }
        }

        #pragma omp barrier

#ifdef SPLATT_USE_MPI
        p_update_factor_all2all(model, ws, m);
#endif
        #pragma omp barrier
        #pragma omp master
        {
          timer_stop(&mode_timer);
          if(rank == 0) {
            long long int tot_act = 0;
            long long int tot_frac = 0;
            double tot_time = 0.0;
            // long long int tot_same = 0;
            for(int i=0; i<scoo->dims[m]; i++){
              tot_act += act[m][i];
              tot_frac += frac[m][i];
              tot_time += time_slice[m][i];
              // tot_same += same[m][i];

              fprintf(f_frac, "%d,", frac[m][i]);
              fprintf(f_act, "%d,", act[m][i]);
              fprintf(f_time, "%lf,", time_slice[m][i]);
              fprintf(f_lev, "%lf,", lev_score[m][i]);
              // fprintf(f_same, "%d\n", same[m][i]);
            }

            fprintf(f_frac, "\n");
            fprintf(f_act, "\n");
            fprintf(f_time, "\n");
            fprintf(f_lev, "\n");
            // fprintf(f_same, "\n");

            // printf(" Sampled:%lld 	Same sampled: %lld 	percent:%0.3f\n", tot_frac, tot_same, ((float)tot_same)/tot_frac);

            avg_tot_time[m] += (double)mode_timer.seconds;
            avg_sampling_time[m] += sampling_time;
            avg_mttkrp_time[m] += (mttkrp_time - sampling_time);
            avg_solving_time[m] += solving_time;
            
            // printf("  mode: %"SPLATT_PF_IDX" act: %lld     sampled: %lld    percent: %0.3f\n", m+1, tot_act, tot_frac, ((float)tot_frac)/tot_act);
            // printf("  time: %lf\n", tot_time);
            // printf("  mode: %"SPLATT_PF_IDX" time: %0.3fs\n", m+1,
            //     mode_timer.seconds);
          }
        }
        #pragma omp barrier
      } /* foreach mode */
    } /* end omp parallel */

    /* compute new obj value, print stats, and exit if converged */
    val_t loss = tc_loss_sq(train, model, ws);
    val_t frobsq = tc_frob_sq(model, ws);
    if(tc_converge(train, validate, model, loss, frobsq, e, ws)) {
      break;
    }

  } /* foreach iteration */

    printf("\n");
  for(int i=0; i<nmodes; i++){
    printf("MODE: %d\n-----------\n", i);
    printf("  Total Time: %lf\n", (avg_tot_time[i]/count));
    printf("  Sampling Time: %lf\n", (avg_sampling_time[i]/count));
    printf("  MTTKRP Time: %lf\n", (avg_mttkrp_time[i]/count));
    printf("  Solving Time: %lf\n",(avg_solving_time[i]/count));
    printf("\n");
  }
  printf("GRAM Timer : %lf\n", (avg_gram_time/count));

  #pragma omp parallel
  {
    int const tid = splatt_omp_get_thread_num();
    splatt_free(perm_i_global[tid]);
  }
  splatt_free(sample_seeds);

#ifdef SPLATT_USE_MPI
  /* UNDO TERRIBLE HACK */
  tt_free(train);
  train = train_back;
#endif

  scoo_free(scoo);

  /* cleanup */
  /* XXX disabled dense temporarily */
  if(false && ws->maxdense_dim > 0) {
    thd_free(thd_densefactors, ws->nthreads);
  }
}

