#pragma once
#include "types.h"

__device__
bool parseExpression(char* expression, DescribeData data) {
	float stack[STACK_SIZE];
	int sp = -1;
	int k = 0;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while(expression[k] != 0)
	{
		switch(expression[k])
		{
			case 1: {float second; pop(second);float first; pop(first);push(first + second);}break;
			case 2: {float second; pop(second);float first; pop(first);push(first - second);}break;
			case 3: {float second; pop(second);float first; pop(first);push(first * second);}break;
			case 4: {float second; pop(second);float first; pop(first);if (second == 0.0) push(1.0);else push(first / second);}break;
			case 5: {float first; pop(first);push(-first);}break;
			case 6: {float top; pop(top);push(exp(top));}break;
			case 7: {float top; pop(top);push(logf(fabs(top)));}break;
			case 8: {float forth; pop(forth);float third; pop(third);float second; pop(second);float first; pop(first);if (first > second) push(third); else push(forth);}break;
			case 9: {float second; pop(second);float first; pop(first);if (first > second) push(first); else push(second);}break;
			case 10: {float second; pop(second);float first; pop(first);if (first < second) push(first); else push(second);}break;
			case 11: {float top; pop(top);push(cos(top));}break;
			case 12: {float top; pop(top);push(sin(top));}break;
			case 13: {push(0.0);}break;
			case 14: {push(1.0);}break;
			case 15: {push(2.0);}break;
			case 16: {push(3.0);}break;
			case 17: {char args[4];for (int i = 0 ; i < 4 ; i++)args[i] = (char)(expression[++k]);push (*((float*) args));}break;
			case 18: {float top; pop(top);if (top == 0.0) push(data.input[tid].y/255.0);else if (top == 1.0) push (data.input[tid].z/255.0);else if (top == 2.0) push (data.input[tid].w/255.0);else if (top == 3.0) {float4 value = data.input[tid];push ((value.y + value.z + value.w)/( 3 * 255.0));}}break;
			case 19: {float top; pop(top);if (top == 0.0) push(data.smallAvg[tid].y/255.0);else if (top == 1.0) push (data.smallAvg[tid].z/255.0);else if (top == 2.0) push (data.smallAvg[tid].w/255.0);else if (top == 3.0) {float4 value = data.smallAvg[tid];push ((value.y + value.z + value.w)/( 3 * 255.0));}}break;
			case 20: {float top; pop(top);if (top == 0.0) push(data.mediumAvg[tid].y/255.0);else if (top == 1.0) push (data.mediumAvg[tid].z/255.0);else if (top == 2.0) push (data.mediumAvg[tid].w/255.0);else if (top == 3.0) {float4 value = data.mediumAvg[tid];push ((value.y + value.z + value.w)/( 3 * 255.0));}}break;
			case 21: {float top; pop(top);if (top == 0.0) push(data.largeAvg[tid].y/255.0);else if (top == 1.0) push (data.largeAvg[tid].z/255.0);else if (top == 2.0) push (data.largeAvg[tid].w/255.0);else if (top == 3.0) {float4 value = data.largeAvg[tid];push ((value.y + value.z + value.w)/( 3 * 255.0));}}break;
			case 22: {float top; pop(top);if (top == 0.0) push(data.smallSd[tid].y/255.0);else if (top == 1.0) push (data.smallSd[tid].z/255.0);else if (top == 2.0) push (data.smallSd[tid].w/255.0);else if (top == 3.0) {float4 value = data.smallSd[tid];push ((value.y + value.z + value.w)/( 3 * 255.0));}}break;
			case 23: {float top; pop(top);if (top == 0.0) push(data.mediumSd[tid].y/255.0);else if (top == 1.0) push (data.mediumSd[tid].z/255.0);else if (top == 2.0) push (data.mediumSd[tid].w/255.0);else if (top == 3.0) {float4 value = data.mediumSd[tid];push ((value.y + value.z + value.w)/( 3 * 255.0));}}break;
			case 24: {float top; pop(top);if (top == 0.0) push(data.largeSd[tid].y/255.0);else if (top == 1.0) push (data.largeSd[tid].z/255.0);else if (top == 2.0) push (data.largeSd[tid].w/255.0);else if (top == 3.0) {float4 value = data.largeSd[tid];push ((value.y + value.z + value.w)/( 3 * 255.0));}}break;
			default:printf("Unrecognized OPCODE in the expression tree!");break;
		} // end switch

		k++;
	} // end while


	float result;
	pop(result);

	if (sp != -1)
		printf("Stack pointer not -1 but is %d! \t", sp);

	return (result > 0);
}
