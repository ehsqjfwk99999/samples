#include <emscripten.h> // import emsdk path
#include <stdio.h>

EM_JS(void, EM_JS_Function, (), {console.log('Function from EM_JS')})

int main() {
  printf("From main: WASM Loaded!\n");
  printf("-----------------------\n");

  emscripten_run_script("console.log('Run JavaScript in C code!')");

  int jsInt = emscripten_run_script_int("returnInt()");
  printf("Int from JS returnInt(): %d\n", jsInt);
  char *jsString = emscripten_run_script_string("returnString()");
  printf("String from JS returnString(): %s\n", jsString);

  EM_JS_Function();

  return 123;
}

void ccallFunction() { printf("From ccallFunction!\n"); }
