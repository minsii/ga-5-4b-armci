/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2016 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

/****************************************************************
 *     Contigure ARMCI PUT/GET/ACC Bandwidth & Latency          *
 ***************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <armci.h>
#include <mpi.h>

#define SKIP_DEFAULT 500
#define LOOP_DEFAULT 100000
#define SKIP_LARGE  100
#define LOOP_LARGE  5000
#define LARGE_MESSAGE_SIZE 65536
#define LARGE_NPROCS 512

#define MIN_MESSAGE_SIZE (8)
#define MAX_MESSAGE_SIZE (1024*1024*4)
#define MESSAGE_TYPE double
#define ARMCI_ACC_DATATYPE ARMCI_ACC_DBL

/* tells to use ARMCI_Malloc_local instead of plain malloc */
#define MALLOC_LOC
#define TIMER MPI_Wtime
#define max(a, b) (a > b ? a : b)

static int nprocs, me;
static int message_size = MIN_MESSAGE_SIZE;     /* default */
static int loop_input = 0;
static int skip = SKIP_DEFAULT, loop = LOOP_DEFAULT;
static MESSAGE_TYPE *loc_buf = NULL;
static void **remote_ptrs = NULL;

/* Test get by default if no compiling option specified */
#if !defined(TEST_GET) && !defined(TEST_PUT) && !defined(TEST_ACC)
#define TEST_GET
#endif

#if defined(TEST_GET)
static const char *op_name = "get";
#define OP_MACRO(dst_buf, src_buf, bytes, dst)  ARMCI_Get(src_buf, dst_buf, bytes, dst);        /* remote is Get source */

#elif defined(TEST_PUT)
static const char *op_name = "put";
#define OP_MACRO(src_buf, dst_buf, bytes, dst)  ARMCI_Put(src_buf, dst_buf, bytes, dst);

#elif defined(TEST_ACC)
static const char *op_name = "acc";
#define OP_MACRO(src_buf, dst_buf, count, dst)  {                               \
        double scale = 1.0;                                                     \
        ARMCI_Acc(ARMCI_ACC_DATATYPE, &scale, src_buf, dst_buf, count, dst);    \
    }

#endif

typedef enum {
    ARMCI_PERF_CONTIG_P2P,
    ARMCI_PERF_CONTIG_ALL2ONE,
    ARMCI_PERF_CONTIG_ONE2ALL
} ARMCI_Perf_contig_mode_t;

typedef enum {
    ARMCI_PERF_CONTIG_LOCALITY_NONE,
    ARMCI_PERF_CONTIG_LOCALITY_INTER,
    ARMCI_PERF_CONTIG_LOCALITY_SHM
} ARMCI_Perf_contig_locality_t;

static ARMCI_Perf_contig_mode_t test_mode = ARMCI_PERF_CONTIG_P2P;
static const char *test_mode_names[3] = { "p2p", "all2one", "one2all" };

static ARMCI_Perf_contig_locality_t check_locality_flag = ARMCI_PERF_CONTIG_LOCALITY_NONE;

static void fill_array(MESSAGE_TYPE * arr, int count)
{
    int i;
    for (i = 0; i < count; i++)
        arr[i] = (MESSAGE_TYPE) i;
}

static void init_arrays(int cnt)
{
#ifdef MALLOC_LOC
    loc_buf = (MESSAGE_TYPE *) ARMCI_Malloc_local(cnt * sizeof(MESSAGE_TYPE));
#else
    loc_buf = (MESSAGE_TYPE *) malloc(cnt * sizeof(MESSAGE_TYPE));
#endif
    assert(loc_buf != NULL);

    remote_ptrs = malloc(sizeof(void *) * nprocs);
    ARMCI_Malloc(remote_ptrs, (cnt * sizeof(MESSAGE_TYPE)));

    /* ARMCI - initialize the data window */
    fill_array((MESSAGE_TYPE *) remote_ptrs[me], cnt);
}

static void free_arrays(void)
{
    /* cleanup */
    ARMCI_Free(remote_ptrs[me]);
    free(remote_ptrs);

#ifdef MALLOC_LOC
    ARMCI_Free_local(loc_buf);
#else
    free(loc_buf);
#endif

    loc_buf = NULL;
    remote_ptrs = NULL;
}

static void run_nmsg_sz(void)
{
    int cnt = 0, cnt_min = 0, cnt_max = 0, dst = 1;

    /* convert bytes to counts of type */
    cnt_min = max(1, MIN_MESSAGE_SIZE / sizeof(MESSAGE_TYPE));
    cnt_max = max(1, MAX_MESSAGE_SIZE / sizeof(MESSAGE_TYPE));

    /* memory allocation */
    init_arrays(cnt_max);
    MPI_Barrier(MPI_COMM_WORLD);

    /* only rank 0 does the work */
    if (me == 0) {
        printf("%6s %6s %6s %8s %10s %10s\n", "mode", "op", "bytes", "loop", "usec", "MB/s");
        fflush(stdout);
    }

    /* rank 0 issues to rank 1 */
    dst = 1;
    if (me == 0) {
        for (cnt = cnt_min; cnt < cnt_max; cnt *= 2) {
            int skip = SKIP_DEFAULT, loop = LOOP_DEFAULT;
            int bytes = cnt * sizeof(MESSAGE_TYPE);
            int x;

            double t0 = 0.0, t = 0.0;
            double latency, bandwidth;

            if (bytes > LARGE_MESSAGE_SIZE) {
                skip = SKIP_LARGE;
                loop = LOOP_LARGE;
            }

            for (x = 0; x < skip; x++) {
                OP_MACRO(loc_buf, remote_ptrs[dst], bytes, dst);
            }

            t0 = TIMER();
            for (x = 0; x < loop; x++) {
                OP_MACRO(loc_buf, remote_ptrs[dst], bytes, dst);
            }
            t = TIMER() - t0;

            latency = t / loop * (1000 * 1000); /* us */
            bandwidth = bytes / latency;

            printf("%6s %6s %6d %8d %8.2f %8.2f\n",
                   test_mode_names[test_mode], op_name, bytes, loop, latency, bandwidth);
        }
    }

    ARMCI_AllFence();
    MPI_Barrier(MPI_COMM_WORLD);

    /* cleanup */
    free_arrays();
}

static void run_nprocs_one2all(MESSAGE_TYPE * loc_buf, void **remote_ptrs, int cnt)
{
    int x, dst, bytes;
    double t0 = 0.0, t = 0.0;
    double latency, msg_rate;

    bytes = cnt * sizeof(MESSAGE_TYPE);

    /* rank 0 issues to all processes */
    if (me == 0) {
        for (x = 0; x < skip; x++) {
            for (dst = 0; dst < nprocs; dst++) {
                OP_MACRO(loc_buf, remote_ptrs[dst], bytes, dst);
            }
        }

        t0 = TIMER();
        for (x = 0; x < loop; x++) {
            for (dst = 0; dst < nprocs; dst++) {
                OP_MACRO(loc_buf, remote_ptrs[dst], bytes, dst);
            }
        }
        t = TIMER() - t0;       /* measure time for local completion */
    }

    ARMCI_AllFence();
    MPI_Barrier(MPI_COMM_WORLD);

    if (me == 0) {
        latency = t / loop;     /* s */
        msg_rate = nprocs / latency / 1000;     /* kmsg/s */

        printf("%6s %6s %8d %6d %9d %8.2f %8.2f\n",
               test_mode_names[test_mode], op_name, bytes, nprocs, loop,
               latency * (1000 * 1000) /* us */ , msg_rate);
    }
}

static void run_nprocs_all2one(MESSAGE_TYPE * loc_buf, void **remote_ptrs, int cnt)
{
    int x, dst, bytes;
    double t0 = 0.0, t = 0.0, avg_t = 0.0;
    double latency, msg_rate;

    bytes = cnt * sizeof(MESSAGE_TYPE);

    /* all processes issue to rank 0 */
    dst = 0;
    for (x = 0; x < skip; x++) {
        OP_MACRO(loc_buf, remote_ptrs[dst], bytes, dst);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    t0 = TIMER();
    for (x = 0; x < loop; x++) {
        OP_MACRO(loc_buf, remote_ptrs[dst], bytes, dst);
        MPI_Barrier(MPI_COMM_WORLD);   /* ensure local completion on all processes */
    }
    t = TIMER() - t0;

    ARMCI_AllFence();
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Reduce(&t, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (me == 0) {
        avg_t = avg_t / nprocs; /* average time */
        latency = avg_t / loop; /* s */
        msg_rate = 1 / latency / 1000;  /* kmsg/s */

        printf("%6s %6s %8d %6d %9d %8.2f %8.2f\n",
               test_mode_names[test_mode], op_name, bytes, nprocs, loop,
               latency * (1000 * 1000) /* us */ , msg_rate);
    }
}

static void run_nprocs(void)
{
    int cnt = 0, bytes = 0, dst = 1;

    /* convert bytes to counts of type */
    cnt = max(1, message_size / sizeof(MESSAGE_TYPE));
    bytes = cnt * sizeof(MESSAGE_TYPE);

    /* memory allocation */
    init_arrays(cnt);
    MPI_Barrier(MPI_COMM_WORLD);

    /* only rank 0 does the work */
    if (me == 0) {
        printf("%6s %6s %8s %8s %8s %10s %10s\n",
               "mode", "op", "bytes", "nprocs", "loop", "usec", "KMessage/s");
        fflush(stdout);
    }

    if (nprocs > LARGE_NPROCS || bytes > LARGE_MESSAGE_SIZE) {
        skip = SKIP_LARGE;
        loop = LOOP_LARGE;
    }

    /* overwritten by user specified loop */
    if (loop_input > 0) {
        loop = loop_input;
        skip = max(10, loop / 100);
    }

    switch (test_mode) {
    case ARMCI_PERF_CONTIG_ALL2ONE:
        run_nprocs_all2one(loc_buf, remote_ptrs, cnt);
        break;
    case ARMCI_PERF_CONTIG_ONE2ALL:
        run_nprocs_one2all(loc_buf, remote_ptrs, cnt);
        break;
    }

    /* cleanup */
    free_arrays();
}


static int check_locality(void)
{
    MPI_Comm shm_comm = MPI_COMM_NULL;
    int shm_nprocs = 0;
    int err = 0;

    /* get the first process on every node */
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
    MPI_Comm_size(shm_comm, &shm_nprocs);

    if (me == 0) {
        switch (check_locality_flag) {
        case ARMCI_PERF_CONTIG_LOCALITY_INTER:
            printf("check ARMCI_PERF_CONTIG_LOCALITY_INTER\n");
            /* Rank 0 should be on separate node alone */
            if (shm_nprocs > 1) {
                err++;
                fprintf(stderr, "Wrong process locality: expect inter-node only, "
                        "but rank 0 has %d local processes\n", shm_nprocs);
                fflush(stdout);
            }
            break;
        case ARMCI_PERF_CONTIG_LOCALITY_SHM:
            /* All processes should be on the same node */
            if (shm_nprocs != nprocs) {
                err++;
                fprintf(stderr, "Wrong process locality: expect shm only, "
                        "but rank 0 has %d local processes while %d processes "
                        "exist in the world\n", shm_nprocs, nprocs);
                fflush(stdout);
            }
            printf("check ARMCI_PERF_CONTIG_LOCALITY_SHM, %d != %d, err=%d\n", shm_nprocs, nprocs,
                   err);
            break;
        }
    }

    MPI_Bcast(&err, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&shm_comm);

    return err;
}

int main(int argc, char **argv)
{
    int err = 0;
    MPI_Init(&argc, &argv);
    ARMCI_Init_args(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs < 2) {
        if (me == 0) {
            fprintf(stderr, "Input Parameters: <mode: p2p(default) >\n");
            fprintf(stderr, "   OR Parameters: <mode: all2one | one2all> \\"
                    "                  <message_size|default %d> <loop_size|default %d> \\"
                    "                  <check process locality: no (default) | inter | shm>\n",
                    MIN_MESSAGE_SIZE, LOOP_DEFAULT);
            fprintf(stderr, "USAGE: processes >= 2, nprocs=%d \n", nprocs);
            fflush(stdout);
        }
        goto exit;
    }

    if (argc > 1) {
        if (strcmp(argv[1], "p2p") == 0) {
            test_mode = ARMCI_PERF_CONTIG_P2P;
        }
        else if (strcmp(argv[1], "all2one") == 0) {
            test_mode = ARMCI_PERF_CONTIG_ALL2ONE;
        }
        else if (strcmp(argv[1], "one2all") == 0) {
            test_mode = ARMCI_PERF_CONTIG_ONE2ALL;
        }
    }

    /* Parameters only valid for group patterns. */
    if (test_mode != ARMCI_PERF_CONTIG_P2P) {
        if (argc > 2) {
            message_size = atoi(argv[2]);
        }
        if (argc > 3) {
            loop_input = atoi(argv[3]);
        }
        if (argc > 4) {
            if (strcmp(argv[4], "inter") == 0) {
                check_locality_flag = ARMCI_PERF_CONTIG_LOCALITY_INTER;
            }
            else if (strcmp(argv[4], "shm") == 0) {
                check_locality_flag = ARMCI_PERF_CONTIG_LOCALITY_SHM;
            }
        }

        if (check_locality_flag != ARMCI_PERF_CONTIG_LOCALITY_NONE) {
            err = check_locality();
            if (err)
                goto exit;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    switch (test_mode) {
    case ARMCI_PERF_CONTIG_P2P:
        run_nmsg_sz();
        break;
    case ARMCI_PERF_CONTIG_ALL2ONE:
    case ARMCI_PERF_CONTIG_ONE2ALL:
        run_nprocs();
        break;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* done */
  exit:
    ARMCI_Finalize();
    MPI_Finalize();
    return (0);
}
