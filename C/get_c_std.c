#include <stdio.h>

int main() {
  printf("C Standard: ");
#ifdef __STRICT_ANSI__
  printf("C");
#else
  printf("GNU");
#endif
#ifdef __STDC_VERSION__
  switch (__STDC_VERSION__) {
  case 199901:
    printf("99\n");
    break;
  case 201112:
    printf("11\n");
    break;
  case 201710:
    printf("17\n");
    break;
  }
#else
  printf("89\n");
#endif
  return 0;
}
