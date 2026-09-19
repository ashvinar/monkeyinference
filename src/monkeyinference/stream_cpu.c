/* P-core STREAM read: disjoint slices, QoS USER_INTERACTIVE, DCE-proof sink.
 *
 * Build: clang -O3 -pthread -shared -o stream_cpu.dylib stream_cpu.c
 */
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <pthread/qos.h>
#endif

typedef struct {
  const float *p;
  size_t n;
  volatile int *go;
  volatile int *run;
  uint64_t bytes;
  float sink;
  int pad;
} Worker;

static void *worker(void *arg) {
#ifdef __APPLE__
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
  Worker *w = (Worker *)arg;
  while (!*w->go) {
  }
  float sink = 0.0f;
  uint64_t iters = 0;
  const float *p = w->p;
  size_t n = w->n;
  while (*w->run) {
    float acc = 0.0f;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      acc += p[i] + p[i + 1] + p[i + 2] + p[i + 3] + p[i + 4] + p[i + 5] +
             p[i + 6] + p[i + 7];
    }
    for (; i < n; i++) {
      acc += p[i];
    }
    sink += acc;
    iters++;
  }
  w->sink = sink;
  w->bytes = iters * n * sizeof(float);
  return NULL;
}

/* Fill [0, n) with 1.0f so pages are resident. */
void cpu_touch(float *p, size_t n) {
  for (size_t i = 0; i < n; i++) {
    p[i] = 1.0f;
  }
}

/*
 * Run `nthreads` read-loops over disjoint slices of `p[0..n)` until
 * *run becomes 0. Caller sets *go=1 to start. Returns sum of bytes read.
 */
uint64_t cpu_read_join(Worker *ws, pthread_t *ths, int nthreads) {
  uint64_t total = 0;
  float sink = 0.0f;
  for (int t = 0; t < nthreads; t++) {
    pthread_join(ths[t], NULL);
    total += ws[t].bytes;
    sink += ws[t].sink;
  }
  if (sink == 1e30f) {
    return 0; /* keep sink live */
  }
  return total;
}

int cpu_read_spawn(const float *p, size_t n, int nthreads, volatile int *go,
                   volatile int *run, Worker *ws, pthread_t *ths) {
  if (nthreads < 1 || nthreads > 16) {
    return -1;
  }
  size_t chunk = n / (size_t)nthreads;
  for (int t = 0; t < nthreads; t++) {
    ws[t].p = p + (size_t)t * chunk;
    ws[t].n = (t == nthreads - 1) ? (n - (size_t)t * chunk) : chunk;
    ws[t].go = go;
    ws[t].run = run;
    ws[t].bytes = 0;
    ws[t].sink = 0.0f;
    if (pthread_create(&ths[t], NULL, worker, &ws[t]) != 0) {
      *run = 0;
      *go = 1;
      for (int j = 0; j < t; j++) {
        pthread_join(ths[j], NULL);
      }
      return -1;
    }
  }
  return 0;
}
