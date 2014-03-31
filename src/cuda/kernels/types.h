#pragma once

typedef struct {
	float4* input;
	float4* smallAvg;
	float4* mediumAvg;
	float4* largeAvg;
	float4* smallSd;
	float4* mediumSd;
	float4* largeSd;
} DescribeData;

__device__
DescribeData make_DescribeData(float4* input,
		float4* smallAvg, float4* mediumAvg, float4* largeAvg,
		float4* smallSd, float4* mediumSd, float4* largeSd)
{
	DescribeData result;
	result.input = input;

	result.smallAvg = smallAvg;
	result.mediumAvg = mediumAvg;
	result.largeAvg = largeAvg;

	result.smallSd = smallSd;
	result.mediumSd = mediumSd;
	result.largeSd = largeSd;

	return result;
}
