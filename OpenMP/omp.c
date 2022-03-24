#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE (1024 * 1024)
#define N_THREAD 8
#define PRINT_TITLE(str) printf("===== [ %s ] =====\n", str);
#define CHECK_ANSWER(str, sum)                                                 \
  do {                                                                         \
    if (sum == 1048576)                                                        \
      printf("%s: ✅\n", str);                                                  \
    else                                                                       \
      printf("%s: ❌\n", str);                                                  \
  } while (0)

int main() {

  int sum;

  int *A = (int *)malloc(sizeof(int) * SIZE);
  for (int i = 0; i < SIZE; i++) {
    A[i] = 1;
  }

  omp_set_num_threads(N_THREAD);

  // OMP Parallel
  PRINT_TITLE("OMP Parallel")
#pragma omp parallel
  {
    printf("Thread Id: %d\n", omp_get_thread_num());
  }
  printf("\n");

  // OMP Nested
  PRINT_TITLE("OMP Nested")
  omp_set_nested(1);
#pragma omp parallel num_threads(3)
  {
    int parend_tid = omp_get_thread_num();
    printf("Lv 1 - thread %d\n", parend_tid);
#pragma omp parallel num_threads(2)
    { printf("\tLv 2 - thread %d of %d\n", omp_get_thread_num(), parend_tid); }
  }
  omp_set_nested(0);
  printf("\n");

  // OMP For
  PRINT_TITLE("OMP For")
#pragma omp parallel for num_threads(3)
  for (int i = 0; i < 12; i++) {
    printf("[%d] by thread %d\n", i, omp_get_thread_num());
  }
  printf("\n");

  // OMP Schedule
  PRINT_TITLE("OMP Schedule")
#pragma omp parallel for schedule(static, 2) num_threads(3)
  for (int i = 0; i < 12; i++) {
    printf("[%d] by thread %d\n", i, omp_get_thread_num());
  }
  printf("\n");

  // OMP Sections
  PRINT_TITLE("OMP Sections")
#pragma omp parallel sections num_threads(2)
  {
#pragma omp section
    { printf("From section 1\n"); }
#pragma omp section
    { printf("From section 2\n"); }
  }
  printf("\n");

  // OMP Barrier Single
  PRINT_TITLE("OMP Barrier Single")
#pragma omp parallel
  {
    int t_id = omp_get_thread_num();
    if (t_id == 0) {
      printf("Number of threads: %d\n", omp_get_num_threads());
      if (omp_in_parallel()) {
        printf("Threads in parallel\n");
      }
    }
#pragma omp barrier
    printf("Thread Id: %d\n", t_id);
#pragma omp single
    printf("Only executed by thread %d\n", omp_get_thread_num());
  }
  printf("\n");

  // OMP Critical
  PRINT_TITLE("OMP Critical")
  int critical_sum = 0;
#pragma omp parallel for
  for (int i = 0; i < SIZE; i++) {
#pragma omp critical
    critical_sum += A[i];
  }
  CHECK_ANSWER("Sum by Critical", critical_sum);

  // OMP Atomic
  PRINT_TITLE("OMP Atomic")
  int atomic_sum = 0;
#pragma omp parallel for
  for (int i = 0; i < SIZE; i++) {
#pragma omp atomic
    atomic_sum += A[i];
  }
  CHECK_ANSWER("Sum by Atomic", atomic_sum);

  // OMP Lock
  PRINT_TITLE("OMP Lock")
  int lock_sum = 0;
  omp_lock_t omp_lock;
  omp_init_lock(&omp_lock);
#pragma omp parallel for
  for (int i = 0; i < SIZE; i++) {
    omp_set_lock(&omp_lock);
    lock_sum += A[i];
    omp_unset_lock(&omp_lock);
  }
  omp_destroy_lock(&omp_lock);
  CHECK_ANSWER("Sum by Lock", lock_sum);

  // OMP Reduction
  PRINT_TITLE("OMP Reduction")
  int reduction_sum = 0;
#pragma omp parallel for reduction(+ : reduction_sum)
  for (int i = 0; i < SIZE; i++) {
    reduction_sum += A[i];
  }
  CHECK_ANSWER("Sum by Reduction", reduction_sum);

  return 0;
}