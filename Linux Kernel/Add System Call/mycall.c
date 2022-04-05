#include <stdio.h>
#include <unistd.h>

int main() {

  long ret;
  char message[40];

  ret = syscall(600, 56, message);
  printf("Message from kernel: %s\n", message);
  printf("Return value: %ld\n", ret);

  return 0;
}