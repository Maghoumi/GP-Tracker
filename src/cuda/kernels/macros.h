#pragma once

// stack macros
#define STACK_SIZE 128
#define push(A) do { sp++;stack[sp]=A; if(sp >= STACK_SIZE) printf("FUCK!");} while(false)
#define pop(A) do{ A=stack[sp];sp--; }while(false)
