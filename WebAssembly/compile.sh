emcc src/simple.c -s WASM=1 -s EXPORTED_FUNCTIONS="['_main', '_ccallFunction']" -o public/simple.js
