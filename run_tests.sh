#!/bin/bash

DSET=netflix
ALG=spals
NSEEDS=1 # how many seeds?
RANKS="10 40"; # which ranks to evaluate with?



# CHANGE THESE BASED ON YOUR MACHINE SETUP
LOGDIR=${HOME}/results/rrals/waterloo/strong-scaling/${DSET}
TENSOR_DIR=${HOME}/tensors/complete
DSET_FILES="${TENSOR_DIR}/${DSET}_train.bin ${HOME}/tensors/complete/${DSET}_validate.bin ${HOME}/tensors/complete/${DSET}_test.bin";

#
# Environment variables
#

# MKL should only use one thread because each thread is calling BLAS in
# parallel. This is almost always superfluous because the BLAS library should
# determine that it is being run in parallel and not try to parallelize. We
# export this just to be safe when using MKL.
export MKL_NUM_THREADS=1

# Tell GNU OpenMP to bind threads to cores
export OMP_PLACES=cores

# Tell Intel OpenMP to bind threads to cores
export KMP_AFFINITY=granularity=fine,compact,1

# numactl is used to bind/stripe memory to sockets. Specifically, we want to
# ensure that memory is striped across the machine when we run across both
# sockets. You can read about NUMA nodes and first-touch allocation for more
# information and why we want to do this.
echo "Using numactl: $(which numactl)"

mkdir -p ${LOGDIR}
for RANK in ${RANKS};
do
  for REP in `seq 1 ${NSEEDS}`;
  do
    EXE="./build/Linux-x86_64/bin/splatt complete ${DSET_FILES} -i 10 --nowrite --seed=${REP} -r${RANK} -a ${ALG}"

    for THREADS in 1 2 4 8;
    do
      export OMP_NUM_THREADS=${THREADS}
      export KMP_HW_SUBSET=1s,8c,1t
      echo ${THREADS}
      NUMACMD="numactl --interleave=0"

      LOGFILE=${LOGDIR}/${DSET}.${ALG}.r${RANK}.th${THREADS}.seed${REP}.txt
      `${NUMACMD} ${EXE} > ${LOGFILE}`
    done

    for THREADS in 16;
    do
      export OMP_NUM_THREADS=${THREADS}
      export KMP_HW_SUBSET=2s,16c,1t
      echo ${THREADS}
      NUMACMD="numactl --interleave=0,1"

      LOGFILE=${LOGDIR}/${DSET}.${ALG}.r${RANK}.th${THREADS}.seed${REP}.txt
      `${NUMACMD} ${EXE} > ${LOGFILE}`
    done
  done
done
