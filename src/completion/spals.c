
#include "completion.h"
#include "../csf.h"
#include "../util.h"
#include "../reorder.h"         /* `shuffle_idx` needed for random sampling */

#include "../io.h"
#include "../sort.h"

#include <math.h>
#include <omp.h>

/* TODO: Conditionally include this OR define lapack prototypes below?
 *       What does this offer beyond prototypes? Can we detect at compile time
 *       if we are using MKL vs ATLAS, etc.?
 */
//#include <mkl.h>


/* Use hardcoded 3-mode kernels when possible. Results in small speedups. */
#ifndef USE_3MODE_OPT
#define USE_3MODE_OPT 1
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



static void p_process_tile3(
    splatt_csf const * const csf,
    idx_t const tile,
    tc_model * const model,
    tc_ws * const ws,
    thd_info * const thd_densefactors,
    int const tid)
{
  csf_sparsity const * const pt = csf->pt + tile;
  /* empty tile */
  if(pt->vals == 0) {
    return;
  }

  idx_t const nfactors = model->rank;

  idx_t const * const restrict sptr = pt->fptr[0];
  idx_t const * const restrict fptr = pt->fptr[1];
  idx_t const * const restrict fids = pt->fids[1];
  idx_t const * const restrict inds = pt->fids[2];

  val_t const * const restrict avals = model->factors[csf->dim_perm[1]];
  val_t const * const restrict bvals = model->factors[csf->dim_perm[2]];
  val_t const * const restrict vals = pt->vals;

  /* buffers */
  val_t * const restrict accum = ws->thds[tid].scratch[1];
  val_t * const restrict mat_accum = ws->thds[tid].scratch[3];

  /* update each slice */
  idx_t const nslices = pt->nfibs[0];
  for(idx_t i=0; i < nslices; ++i) {
    /* fid is the row we are actually updating */
    idx_t const fid = (pt->fids[0] == NULL) ? i : pt->fids[0][i];

    /* replicated structures */
    val_t * const restrict out_row =
        (val_t *) thd_densefactors[tid].scratch[0] + (fid * nfactors);
    val_t * const restrict neqs =
        (val_t *) thd_densefactors[tid].scratch[1] + (fid*nfactors*nfactors);

    idx_t bufsize = 0; /* how many hada vecs are in mat_accum */
    idx_t nflush = 1;  /* how many times we have flushed to add to the neqs */
    val_t * restrict hada = mat_accum;

    /* process each fiber */
    for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
      val_t const * const restrict av = avals  + (fids[fib] * nfactors);

      /* first entry of the fiber is used to initialize accum */
      idx_t const jjfirst  = fptr[fib];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accum[r] = vfirst * bv[r];
        hada[r] = av[r] * bv[r];
      }
      hada += nfactors;
      if(++bufsize == ALS_BUFSIZE) {
        /* add to normal equations */
        p_vec_oprod(mat_accum, nfactors, bufsize, nflush++, neqs);
        hada = mat_accum;
        bufsize = 0;
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[fib]+1; jj < fptr[fib+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accum[r] += v * bv[r];
          hada[r] = av[r] * bv[r];
        }
        hada += nfactors;
        if(++bufsize == ALS_BUFSIZE) {
          /* add to normal equations */
          p_vec_oprod(mat_accum, nfactors, bufsize, nflush++, neqs);
          hada = mat_accum;
          bufsize = 0;
        }
      }

      /* accumulate into output row */
      for(idx_t r=0; r < nfactors; ++r) {
        out_row[r] += accum[r] * av[r];
      }
    } /* foreach fiber */

    /* final flush */
    p_vec_oprod(mat_accum, nfactors, bufsize, nflush++, neqs);
  } /* foreach slice */
}



static void p_process_slice(
    splatt_csf const * const csf,
    idx_t const tile,
    idx_t const i,
    val_t * * mvals,
    idx_t const nfactors,
    val_t * const restrict out_row,
    val_t * const accum,
    val_t * const restrict neqs,
    val_t * const restrict neqs_buf,
    val_t * const neqs_buf_tree,
    idx_t * const nflush,
    val_t **const lev_score,
    int alpha,
    int beta,
    int **act,
    int **frac,
    int mode,
    double *sampling_time,
    double *mttkrp_time,
    double *mttkrptime,
    double *samplingtime);



static void p_process_tile(
    splatt_csf const * const csf,
    idx_t const tile,
    tc_model * const model,
    tc_ws * const ws,
    thd_info * const thd_densefactors,
    int const tid,
    val_t **const lev_score,
    int alpha,
    int beta,
    int **act,
    int **frac,
    int **act_dense,
    int **frac_dense,
    int mode,
    double *sampling_time,
    double *mttkrp_time,
    double *mttkrptime,
    double *samplingtime)
{
  csf_sparsity const * const pt = csf->pt + tile;
  /* empty tile */
  if(pt->vals == 0) {
    return;
  }

  idx_t const nmodes = csf->nmodes;
#if USE_3MODE_OPT
  if(nmodes == 3) {
    p_process_tile3(csf, tile, model, ws, thd_densefactors, tid);
    return;
  }
#endif

  idx_t const nfactors = model->rank;

  /* buffers */
  val_t * const restrict accum = ws->thds[tid].scratch[1];
  val_t * const restrict mat_accum = ws->thds[tid].scratch[3];
  val_t * const restrict hada_accum  = ws->thds[tid].scratch[4];

  val_t * mvals[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = model->factors[csf->dim_perm[m]];
  }

  /* update each slice */
  idx_t const nslices = pt->nfibs[0];
  // printf("%d, %d\n",nslices, tile);


  for(idx_t i=0; i < nslices; ++i) {
    /* fid is the row we are actually updating */
    idx_t const fid = (pt->fids[0] == NULL) ? i : pt->fids[0][i];

    /* replicated structures */
    val_t * const restrict out_row =
        (val_t *) thd_densefactors[tid].scratch[0] + (fid * nfactors);
    val_t * const restrict neqs =
        (val_t *) thd_densefactors[tid].scratch[1] + (fid*nfactors*nfactors);

    idx_t bufsize = 0; /* how many hada vecs are in mat_accum */
    idx_t nflush = 1;  /* how many times we have flushed to add to the neqs */
    val_t * restrict hada = mat_accum;

    /* process each fiber */
    p_process_slice(csf, tile, i, mvals, nfactors, out_row, accum, neqs,
        mat_accum, hada_accum, &nflush, lev_score, alpha, beta, act_dense, frac_dense, tile, sampling_time, mttkrp_time, mttkrptime, samplingtime);
  } /* foreach slice */
}


#ifndef SPALS_MAX_THREADS
#define SPALS_MAX_THREADS 1024
#endif

static idx_t const PERM_INIT = 128;
static idx_t perm_i_lengths[SPALS_MAX_THREADS];
static idx_t * perm_i_global[SPALS_MAX_THREADS];

static idx_t const SEED_PADDING = 16;
static unsigned int * sample_seeds;


static void p_process_slice3(
    splatt_csf const * const csf,
    idx_t const tile,
    idx_t const i,
    val_t const * const restrict A,
    val_t const * const restrict B,
    idx_t const nfactors,
    val_t * const restrict out_row,
    val_t * const restrict accum,
    val_t * const restrict neqs,
    val_t * const restrict neqs_buf,
    idx_t * const nflush,
    int alpha,
    int beta,
    int **act,
    int **frac,
    int mode)
{
  csf_sparsity const * const pt = csf->pt + tile;
  idx_t const * const restrict sptr = pt->fptr[0];
  idx_t const * const restrict fptr = pt->fptr[1];
  idx_t const * const restrict fids = pt->fids[1];
  idx_t const * const restrict inds = pt->fids[2];
  val_t const * const restrict vals = pt->vals;

  idx_t const sample_threshold = alpha * nfactors;
  idx_t const sample_rate = beta;

  int const tid = splatt_omp_get_thread_num();
  idx_t * perm_i = NULL;

  /* clear out accumulation buffer */
  for(idx_t f=0; f < nfactors; ++f) {
    accum[f + nfactors] = 0;
  }

  val_t * hada = neqs_buf;
  idx_t bufsize = 0;


  int tot_nnz = 0;
  int tot_sampled = 0;
  idx_t *nnz_fib = (idx_t *)malloc((sptr[i+1] - sptr[i]) * sizeof(idx_t));



  // Sampling in fibre precalculation (Uniform sampling)
  for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib){
    idx_t const ntotal = fptr[fib+1] - fptr[fib];

    nnz_fib[fib - sptr[i]] = ntotal;
    tot_nnz += ntotal;
  }

  act[mode][i] = tot_nnz;



  // Number of samples required from each slice
  idx_t sample_slice = SS_MIN(tot_nnz, sample_threshold + ((tot_nnz-sample_threshold) / sample_rate));
  // Distribute the # of samples to each fibre based on uniform sampling
  for(int i=0; i < (sptr[i+1] - sptr[i]); i++)
    nnz_fib[i] = (nnz_fib[i] / tot_nnz) * sample_slice;




  /* process each fiber */
  for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
    val_t const * const restrict av = A  + (fids[fib] * nfactors);

    int sample;
    idx_t const start = fptr[fib];
    idx_t const end = fptr[fib+1];

    idx_t const ntotal = end-start;
    idx_t iter_end;

    // Sample form each fibre
    if(ntotal > sample_threshold) {
      sample = 1;

      if(ntotal > perm_i_lengths[tid]) {
        perm_i_lengths[tid] = ntotal;
        splatt_free(perm_i_global[tid]);
        perm_i_global[tid] = splatt_malloc(ntotal * sizeof(*perm_i_global));
      }
      perm_i = perm_i_global[tid];
      for(idx_t n=start; n < end; ++n) {
        perm_i[n-start] = n;
      }
      idx_t sample_size = nnz_fib[fib - sptr[i]];
      // quick_shuffle(perm_i, perm_i, sample_size, sample_size, &(sample_seeds[tid * SEED_PADDING]));
      quick_shuffle(perm_i, sample_size, &(sample_seeds[tid * SEED_PADDING]));
      iter_end = start + sample_size;
    } else {
      sample = 0;
      iter_end = end;
    }

    // for(idx_t jj=start; jj < iter_end; ++jj){
    //   val_t v;
    //   val_t * bv;
    //   if(sample == 1){
    //     v = vals[perm_i[jj - start]];
    //     bv = B + (inds[perm_i[jj - start]] * nfactors);
    //   }
    //   else{
    //     v = vals[jj];
    //     bv = B + (inds[jj] * nfactors);
    //   }
    // }

    /* first entry of the fiber is used to initialize accum */
    // idx_t const jjfirst  = fptr[fib];
    // val_t const vfirst   = vals[jjfirst];
    // val_t const * const restrict bv = B + (inds[jjfirst] * nfactors);
    // for(idx_t r=0; r < nfactors; ++r) {
    //   accum[r] = vfirst * bv[r];
    //   hada[r] = av[r] * bv[r];
    // }

    hada += nfactors;
    if(++bufsize == ALS_BUFSIZE) {
      /* add to normal equations */
      p_vec_oprod(neqs_buf, nfactors, bufsize, (*nflush)++, neqs);
      bufsize = 0;
      hada = neqs_buf;
    }

    /* foreach nnz in fiber */
    for(idx_t jj=start; jj < iter_end; ++jj) {
      val_t v;
      val_t * bv;
      if(sample == 1){
        v = vals[perm_i[jj - start]];
        bv = B + (inds[perm_i[jj - start]] * nfactors);
      }
      else{
        v = vals[jj];
        bv = B + (inds[jj] * nfactors);
      }
      // val_t const v = vals[jj];
      // val_t const * const restrict bv = B + (inds[jj] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accum[r] += v * bv[r];
        hada[r] = av[r] * bv[r];
      }

      hada += nfactors;
      if(++bufsize == ALS_BUFSIZE) {
        /* add to normal equations */
        p_vec_oprod(neqs_buf, nfactors, bufsize, (*nflush)++, neqs);
        bufsize = 0;
        hada = neqs_buf;
      }
    }

    /* accumulate into output row */
    for(idx_t r=0; r < nfactors; ++r) {
      out_row[r] += accum[r] * av[r];
    }

  } /* foreach fiber */

  /* final flush */
  p_vec_oprod(neqs_buf, nfactors, bufsize, (*nflush)++, neqs);
}


/*
 * RRALS - these are the permutation vectors for each thread.
 * XXX: throw these into a workspace structure or something else not global...
 */
#ifndef SPALS_MAX_THREADS
#define SPALS_MAX_THREADS 1024
#endif

// static idx_t const PERM_INIT = 128;
// static idx_t perm_i_lengths[SPALS_MAX_THREADS];
// static idx_t * perm_i_global[SPALS_MAX_THREADS];

/*
 * Each thread is given a random seed to use for sampling. We pad them to
 * ensure each falls on a different cache line (to avoid false sharing).
 */
// static idx_t const SEED_PADDING = 16;
// static unsigned int * sample_seeds;

static void p_process_slice(
    splatt_csf const * const csf,
    idx_t const tile,
    idx_t const i,
    val_t * * mvals,
    idx_t const nfactors,
    val_t * const restrict out_row,
    val_t * const accum,
    val_t * const restrict neqs,
    val_t * const restrict neqs_buf,
    val_t * const neqs_buf_tree,
    idx_t * const nflush,
    val_t **const lev_score,
    int alpha,
    int beta,
    int **act,
    int **frac,
    int mode,
    double *sampling_time,
    double *mttkrp_time,
    double *mttkrptime,
    double *samplingtime)
{
  struct timeval start_t, start_tt, stop_t, stop_tt;
  // idx_t sample_threshold;
  // if(mode == 0)
  //   sample_threshold = 4 * nfactors;
  // else if(mode == 1)
  //   sample_threshold = 0.5 * nfactors;
  // else if(mode == 2)
  //   sample_threshold = 0.01 * nfactors;

  idx_t const sample_threshold = alpha * nfactors;
  idx_t const sample_rate = beta;
  idx_t const nmodes = csf->nmodes;
  csf_sparsity const * const pt = csf->pt + tile;
  val_t const * const restrict vals = pt->vals;
  if(vals == NULL) {
    return;
  }


#if USE_3MODE_OPT
  if(nmodes == 3) {
    p_process_slice3(csf, tile, i, mvals[1], mvals[2], nfactors, out_row,
        accum, neqs, neqs_buf, nflush, alpha, beta, act, frac, mode);
    return;
  }
#endif


  idx_t const * const * const restrict fp = (idx_t const * const *) pt->fptr;
  idx_t const * const * const restrict fids = (idx_t const * const *) pt->fids;
  idx_t const * const restrict inds = fids[nmodes-1];
  val_t const * const restrict lastmat = mvals[nmodes-1];

  idx_t bufsize = 0;
  val_t * hada = neqs_buf;


  // // Mode which must be chosen to compute MTTKRP
  // idx_t *Modes = (idx_t *)malloc((nmodes-1)*sizeof(idx_t));
  // int k=0;
  // for(int m=0; m<nmodes; m++){
  //   if(mode == m)
  //     continue;
  //   Modes[k++] = m;
  // }



  gettimeofday(&start_tt, NULL);
  /* push initial idx initialize idxstack */
  idx_t idxstack[MAX_NMODES];
  idxstack[0] = i;
  for(idx_t m=1; m < nmodes-1; ++m) {
    idxstack[m] = fp[m-1][idxstack[m-1]];
  }

  idx_t const top_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];

  val_t const * const restrict rootrow = mvals[0] + (top_id * nfactors);
  for(idx_t f=0; f < nfactors; ++f) {
    neqs_buf_tree[f] = 1.;
  }

  /* clear out accumulation buffer */
  for(idx_t f=0; f < nfactors; ++f) {
    accum[f + nfactors] = 0;
  }

  /* process each subtree */
  idx_t depth = 0;


  int sample;
  /* strictly, this permutation array should be of size equal to end-start for each iter */
  int const tid = splatt_omp_get_thread_num();
  idx_t * perm_i = NULL;

  int tot_nnz = 0;
  int sampled_nnz = 0;

  while(idxstack[1] < fp[0][i+1]) {

    /* move down to nnz node while forming hada */
    for(; depth < nmodes-2; ++depth) {
      val_t const * const restrict drow
          = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
      val_t const * const restrict cur_buf = neqs_buf_tree + ((depth+0) * nfactors);
      val_t * const restrict nxt_buf = neqs_buf_tree + ((depth+1) * nfactors);

      for(idx_t f=0; f < nfactors; ++f) {
        nxt_buf[f] = cur_buf[f] * drow[f];
      }
    }
    val_t * const restrict last_hada = neqs_buf_tree + (depth * nfactors);
    val_t * const restrict accum_nnz = accum + ((depth+1) * nfactors);

    /* process all nonzeros [start, end) */
    gettimeofday(&start_t, NULL);
    idx_t const start = fp[depth][idxstack[depth]];
    idx_t const end   = fp[depth][idxstack[depth]+1];

    /* create random sample array */
    /* NOTE: could possibly use the shuffle_idx function in reorder.c */
    idx_t const ntotal = end-start;
    idx_t iter_end;
    if(ntotal > sample_threshold) {
      sample = 1;
      /* realloc if needed */
      if(ntotal > perm_i_lengths[tid]) {
        perm_i_lengths[tid] = ntotal;
        splatt_free(perm_i_global[tid]);
        perm_i_global[tid] = splatt_malloc(ntotal * sizeof(*perm_i_global));
      }
      perm_i = perm_i_global[tid];
      for(idx_t n=start; n < end; ++n) {
        perm_i[n-start] = n;
      }
      idx_t sample_size = SS_MIN(ntotal, sample_threshold + ((ntotal-sample_threshold) / sample_rate));
      // quick_shuffle(perm_i, perm_i, sample_size, sample_size, &(sample_seeds[tid * SEED_PADDING]));
      quick_shuffle(perm_i, sample_size, &(sample_seeds[tid * SEED_PADDING]));
      iter_end = start + sample_size;
    } else {
      sample = 0;
      iter_end = end;
    }
    gettimeofday(&stop_t, NULL);
    *sampling_time += (stop_t.tv_sec + stop_t.tv_usec/1000000.0)- (start_t.tv_sec + start_t.tv_usec/1000000.0);
    samplingtime[tid] += (stop_t.tv_sec + stop_t.tv_usec/1000000.0)- (start_t.tv_sec + start_t.tv_usec/1000000.0);


    tot_nnz += ntotal;
    sampled_nnz += iter_end - start;

    for(idx_t jj=start; jj < iter_end; ++jj) {
      val_t v;
      val_t * lastrow;
      if(sample == 1) {
        gettimeofday(&start_t, NULL);
        v = vals[perm_i[jj-start]];
        lastrow = lastmat + (inds[perm_i[jj-start]] * nfactors);
        gettimeofday(&stop_t, NULL);
        *sampling_time += (stop_t.tv_sec + stop_t.tv_usec/1000000.0)- (start_t.tv_sec + start_t.tv_usec/1000000.0);
      } else {
        v = vals[jj];
        lastrow = lastmat + (inds[jj] * nfactors);
      }

      /* process nnz */
      for(idx_t f=0; f < nfactors; ++f) {
        accum_nnz[f] += v * lastrow[f];
        hada[f] = last_hada[f] * lastrow[f];
      }

      /* add to normal equations */
      hada += nfactors;
      if(++bufsize == ALS_BUFSIZE) {
        p_vec_oprod(neqs_buf, nfactors, bufsize, (*nflush)++, neqs);
        bufsize = 0;
        hada = neqs_buf;
      }
    }

    idxstack[depth+1] = end;

    /* propagate MTTKRP up */
    do {
      val_t const * const restrict fibrow
          = mvals[depth] + (fids[depth][idxstack[depth]] * nfactors);
      val_t * const restrict up   = accum + ((depth+0) * nfactors);
      val_t * const restrict down = accum + ((depth+1) * nfactors);

      /*
       * up[:] += down[:] * fibrow[:];
       * down[:] = 0.;
       */
      p_add_hada_clear(up, down, fibrow, nfactors);

      ++idxstack[depth];
      --depth;
    } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
  } /* foreach fiber subtree */ 

  /* accumulate into output row */
  for(idx_t f=0; f < nfactors; ++f) {
    out_row[f] += accum[f + nfactors];
  }

  act[mode][i] = tot_nnz;
  frac[mode][i] = sampled_nnz;

  gettimeofday(&stop_tt, NULL);
  *mttkrp_time += (stop_tt.tv_sec + stop_tt.tv_usec/1000000.0)- (start_tt.tv_sec + start_tt.tv_usec/1000000.0);
  mttkrptime[tid] += (stop_tt.tv_sec + stop_tt.tv_usec/1000000.0)- (start_tt.tv_sec + start_tt.tv_usec/1000000.0);

  /* final flush */
  p_vec_oprod(neqs_buf, nfactors, bufsize, (*nflush)++, neqs);
}







/**
* @brief Compute the i-ith row of the MTTKRP, form the normal equations, and
*        store the new row.
*
* @param csf The tensor of training data.
* @param tile The tile that row i resides in.
* @param i The row to update.
* @param reg Regularization parameter for the i-th row.
* @param model The model to update
* @param ws Workspace.
* @param tid OpenMP thread id.
*/
static void p_update_slice(
    splatt_csf const * const csf,
    idx_t const tile,
    idx_t const i,
    val_t const reg,
    tc_model * const model,
    tc_ws * const ws,
    int const tid,
    val_t ** const lev_score,
    int alpha,
    int beta,
    int **act,
    int **frac,
    int mode,
    double *solving_time,
    double *sampling_time,
    double *mttkrp_time,
    double *mttkrptime,
    double *solvingtime,
    double *samplingtime)
{
  struct timeval start, stop;
  double time_diff;

  idx_t const nmodes = csf->nmodes;
  idx_t const nfactors = model->rank;
  csf_sparsity const * const pt = csf->pt + tile;

  /* fid is the row we are actually updating */
  idx_t const fid = (pt->fids[0] == NULL) ? i : pt->fids[0][i];
#ifdef SPLATT_USE_MPI
  assert(fid < model->globmats[csf->dim_perm[0]]->I);
  val_t * const restrict out_row = model->globmats[csf->dim_perm[0]]->vals +
      (fid * nfactors);
#else
  val_t * const restrict out_row = model->factors[csf->dim_perm[0]] +
      (fid * nfactors);
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
    mats[m] = model->factors[csf->dim_perm[m]];
  }

  /* do MTTKRP + dsyrk */
  p_process_slice(csf, 0, i, mats, nfactors, out_row, accum, neqs, mat_accum,
      hada_accum, &nflush, lev_score, alpha, beta, act, frac, mode, sampling_time, mttkrp_time, mttkrptime, samplingtime);


  /* add regularization to the diagonal */
  gettimeofday(&start, NULL);
  for(idx_t f=0; f < nfactors; ++f) {
    neqs[f + (f * nfactors)] += reg;
  }

  /* solve! */
  p_invert_row(neqs, out_row, nfactors);
  gettimeofday(&stop, NULL);
  *solving_time += (stop.tv_sec + stop.tv_usec/1000000.0) - (start.tv_sec + start.tv_usec/1000000.0);
  solvingtime[tid] += (stop.tv_sec + stop.tv_usec/1000000.0) - (start.tv_sec + start.tv_usec/1000000.0);
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
    int const tid,
    val_t **lev_score,
    int alpha,
    int beta,
    int **act,
    int **frac,
    int mode,
    double *solving_time,
    double *sampling_time,
    double *mttkrp_time,
    double *mttkrptime,
    double *solvingtime,
    double *samplingtime)
{

  struct timeval start, stop;

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


  int **act_dense = (int **)malloc(csf[m].ntiles*sizeof(int *));
  int **frac_dense = (int **)malloc(csf[m].ntiles*sizeof(int *));

  for(int tile=0; tile < csf[m].ntiles; ++tile){
    act_dense[tile] = (int *)malloc(model->dims[m]*sizeof(int));
    frac_dense[tile] = (int *)malloc(model->dims[m]*sizeof(int));

    for(int i=0; i<model->dims[m]; i++){
      act_dense[tile][i] = 0;
      frac_dense[tile][i] = 0;
    }
  }


  /* update each tile in parallel */
  #pragma omp for schedule(dynamic, 1)
  for(idx_t tile=0; tile < csf[m].ntiles; ++tile) {
    p_process_tile(csf+m, tile, model, ws, thd_densefactors, tid, lev_score, alpha, beta, act, frac, act_dense, frac_dense, mode, sampling_time, mttkrp_time, mttkrptime, samplingtime);
  }

  for(int tile=0; tile < csf[m].ntiles; tile++){
    for(int i=0; i<model->dims[m]; i++){
      act[mode][i] += act_dense[tile][i];
      frac[mode][i] += frac_dense[tile][i];
    }
  }

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
    gettimeofday(&start, NULL);
    val_t * const restrict neqs_i =
        (val_t *) thd_densefactors[0].scratch[1] + (i * rank * rank);
    /* add regularization */
    for(idx_t f=0; f < rank; ++f) {
      neqs_i[f + (f * rank)] += reg;
    }

    /* Cholesky + solve */
    p_invert_row(neqs_i, out + (i * rank), rank);
    gettimeofday(&stop, NULL);
    *solving_time += (stop.tv_sec + stop.tv_usec/1000000.0) - (start.tv_sec + start.tv_usec/1000000.0);
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
}



// extern 'C' {
//     // LU decomoposition of a general matrix
//     void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

//     // generate inverse of a matrix given its LU decomposition
//     void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
// }

static void GramInv(val_t *A, idx_t N){
  int *IPIV = (int *)malloc((N+1) * sizeof(int));
  int LWORK = N*N;
  double *WORK = (double *)malloc(LWORK * sizeof(double));
  int INFO;

  dgetrf_(&N,&N,A,&N,IPIV,&INFO);
  dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

  free(IPIV);
  free(WORK);
}



static void getLvrgScore(val_t *A, val_t *gram, val_t **lev_score, idx_t rank, idx_t nrows, int factor){
  for (int i=0; i<nrows; ++i){
    for (int j1=0; j1<rank; j1++){
      for (int j2=0; j2<rank; j2++){
        lev_score[factor][i] += A[i*rank + j1] * gram[j1*rank + j2] * A[i*rank + j2];
      }
    }
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


void splatt_tc_spals(
    sptensor_t * train,
    sptensor_t * const validate,
    tc_model * const model,
    tc_ws * const ws,
    int alpha,
    int beta)
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

  splatt_csf csf[MAX_NMODES];

  /* convert training data to CSF-ALLMODE */
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS] = ws->nthreads;
  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ALLMODE;

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
  }

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


  int **act = (int **)malloc(nmodes*sizeof(int *));
  for(int i=0; i<nmodes; i++){
    act[i] = (int *)malloc((model->dims[i])*sizeof(int));
  }

  int **frac = (int **)malloc(nmodes*sizeof(int *));
  for(int i=0; i<nmodes; i++){
    frac[i] = (int *)malloc((model->dims[i])*sizeof(int));
  }

  double **time_slice = (double **)malloc(nmodes*sizeof(double *));
  for(int i=0; i<nmodes; i++){
    time_slice[i] = (double *)malloc((model->dims[i])*sizeof(double));
  }


  double *mttkrptime = (double *)malloc(ws->nthreads * sizeof(double));
  for(int i=0; i<ws->nthreads; i++)
    mttkrptime[i] = 0.0;
  double *solvingtime = (double *)malloc(ws->nthreads * sizeof(double));
  for(int i=0; i<ws->nthreads; i++)
    solvingtime[i] = 0.0;
  double *samplingtime = (double *)malloc(ws->nthreads * sizeof(double));
  for(int i=0; i<ws->nthreads; i++)
    samplingtime[i] = 0.0;


  double avg_sampling_time[3] = {0.0, 0.0, 0.0};
  double avg_solving_time[3] = {0.0, 0.0, 0.0};
  double avg_mttkrp_time[3] = {0.0, 0.0, 0.0};
  double avg_tot_time[3] = {0.0, 0.0, 0.0};
  int count = 0;


  sp_timer_t mode_timer;
  timer_reset(&mode_timer);
  timer_start(&ws->tc_time);


  val_t **lev_score = (val_t **)malloc(nmodes * sizeof(val_t *));
  for(int i=0; i<nmodes; i++)
    lev_score[i] = (val_t *)malloc((model->dims[i])*sizeof(val_t));


  for(idx_t e=1; e < ws->max_its+1; ++e) {
    count++;

    for(int i=0; i<nmodes; i++){
      val_t *gram = getGram(model->factors[i], model->dims[i], model->rank);
      GramInv(gram, model->rank);

      getLvrgScore(model->factors[i], gram, lev_score, model->rank, model->dims[i], i);
    }

    #pragma omp parallel
    {
      int const tid = splatt_omp_get_thread_num();

      for(idx_t m=0; m < nmodes; ++m) {
        double solving_time = 0.0;
        double sampling_time = 0.0;
        double mttkrp_time = 0.0;
        double tottime_mode3;

        #pragma omp master
        timer_fstart(&mode_timer);

        if(ws->isdense[m]) {
          // struct timeval start_t, stop_t;
          // gettimeofday(&start_t, NULL);
          // p_densemode_als_update(csf, m, model, ws, thd_densefactors, tid, alpha, beta, act, frac, m, &solving_time, &sampling_time, &mttkrp_time);
          // gettimeofday(&stop_t, NULL);
          // tottime_mode3 = (stop_t.tv_sec + stop_t.tv_usec/1000000.0) - (start_t.tv_sec + start_t.tv_usec/1000000.0);

        /* dense modes are easy */
        } else {
          /* update each row in parallel */
          for(idx_t i=parts[m][tid]; i < parts[m][tid+1]; ++i) {
            struct timeval start_t, stop_t;
            gettimeofday(&start_t, NULL);
            p_update_slice(csf+m, 0, i, ws->regularization[m], model, ws, tid, lev_score, alpha, beta, act, frac, m, &solving_time, &sampling_time, &mttkrp_time, mttkrptime, solvingtime, samplingtime);
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
            for(int i=0; i<model->dims[m]; i++){
              tot_act += act[m][i];
              tot_frac += frac[m][i];
              tot_time += time_slice[m][i];

              // fprintf(f_frac, "%d,", frac[m][i]);
              // fprintf(f_act, "%d,", act[m][i]);
              // fprintf(f_time, "%lf,", time_slice[m][i]);
            }

            // fprintf(f_frac, "\n");
            // fprintf(f_act, "\n");
            // fprintf(f_time, "\n");

            avg_tot_time[m] += (double)mode_timer.seconds;
            avg_sampling_time[m] += sampling_time;
            avg_mttkrp_time[m] += (mttkrp_time - sampling_time);
            avg_solving_time[m] += solving_time;



            printf("  mode: %"SPLATT_PF_IDX" act: %lld     sampled: %lld    percent: %0.3f\n", m+1, tot_act, tot_frac, ((float)tot_frac)/tot_act);
            // if(m==2)
            //   printf("  Total time: %lf\n", tottime_mode3);
            // else
            //   printf("  Total time: %lf\n", tot_time);
            // printf("  Solving time: %lf\n", solving_time);
            // printf("  Sampling Time: %lf\n", sampling_time);
            // printf("  MTTKRP Time: %lf\n", mttkrp_time);
            // printf("\n");

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

    // printf("\n");
    // for(int i=0; i<ws->nthreads; i++)
    //   printf("  Sampling Time[%d]: %lf\n", i,samplingtime[i]/count);
    // printf("\n");

    // for(int i=0; i<ws->nthreads; i++)
    //   printf("  MTTKRP[%d]: %lf\n", i,(mttkrptime[i]-samplingtime[i])/count);
    // printf("\n");

    // for(int i=0; i<ws->nthreads; i++)
    //   printf("  Solving Time[%d]: %lf\n", i,solvingtime[i]/count);
    printf("\n");
  for(int i=0; i<nmodes; i++){
    printf("MODE: %d\n-----------\n", i);
    printf("  Total Time: %lf\n", (avg_tot_time[i]/count));
    printf("  Sampling Time: %lf\n", (avg_sampling_time[i]/count));
    printf("  MTTKRP Time: %lf\n", (avg_mttkrp_time[i]/count));
    printf("  Solving Time: %lf\n",(avg_solving_time[i]/count));
    printf("\n");
  }

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

  /* cleanup */
  for(idx_t m=0; m < nmodes; ++m) {
    csf_free_mode(csf+m);
    splatt_free(parts[m]);
  }
  if(ws->maxdense_dim > 0) {
    thd_free(thd_densefactors, ws->nthreads);
  }
}

