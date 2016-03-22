#include "Reduction.h"

__global__ void multiplyAndReducePt1(float *J, float *res, float *redArr, int w, int h) {
	extern __shared__ float sdata[];
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	size_t tx = threadIdx.x * 27;
	if(idx < w*h) {
		sdata[tx+0 ] = J[idx*6  ] *   J[idx*6  ];
		sdata[tx+1 ] = J[idx*6  ] *   J[idx*6+1];
		sdata[tx+2 ] = J[idx*6  ] *   J[idx*6+2];
		sdata[tx+3 ] = J[idx*6  ] *   J[idx*6+3];
		sdata[tx+4 ] = J[idx*6  ] *   J[idx*6+4];
		sdata[tx+5 ] = J[idx*6  ] *   J[idx*6+5];
		sdata[tx+6 ] = J[idx*6+1] *   J[idx*6+1];
		sdata[tx+7 ] = J[idx*6+1] *   J[idx*6+2];
		sdata[tx+8 ] = J[idx*6+1] *   J[idx*6+3];
		sdata[tx+9 ] = J[idx*6+1] *   J[idx*6+4];
		sdata[tx+10] = J[idx*6+1] *   J[idx*6+5];
		sdata[tx+11] = J[idx*6+2] *   J[idx*6+2];
		sdata[tx+12] = J[idx*6+2] *   J[idx*6+3];
		sdata[tx+13] = J[idx*6+2] *   J[idx*6+4];
		sdata[tx+14] = J[idx*6+2] *   J[idx*6+5];
		sdata[tx+15] = J[idx*6+3] *   J[idx*6+3];
		sdata[tx+16] = J[idx*6+3] *   J[idx*6+4];
		sdata[tx+17] = J[idx*6+3] *   J[idx*6+5];
		sdata[tx+18] = J[idx*6+4] *   J[idx*6+4];
		sdata[tx+19] = J[idx*6+4] *   J[idx*6+5];
		sdata[tx+20] = J[idx*6+5] *   J[idx*6+5];
		sdata[tx+21] = J[idx*6  ] * res[idx    ];
		sdata[tx+22] = J[idx*6+1] * res[idx    ];
		sdata[tx+23] = J[idx*6+2] * res[idx    ];
		sdata[tx+24] = J[idx*6+3] * res[idx    ];
		sdata[tx+25] = J[idx*6+4] * res[idx    ];
		sdata[tx+26] = J[idx*6+5] * res[idx    ];
	} else {
		sdata[tx+0 ] = 0.0f;
		sdata[tx+1 ] = 0.0f;
		sdata[tx+2 ] = 0.0f;
		sdata[tx+3 ] = 0.0f;
		sdata[tx+4 ] = 0.0f;
		sdata[tx+5 ] = 0.0f;
		sdata[tx+6 ] = 0.0f;
		sdata[tx+7 ] = 0.0f;
		sdata[tx+8 ] = 0.0f;
		sdata[tx+9 ] = 0.0f;
		sdata[tx+10] = 0.0f;
		sdata[tx+11] = 0.0f;
		sdata[tx+12] = 0.0f;
		sdata[tx+13] = 0.0f;
		sdata[tx+14] = 0.0f;
		sdata[tx+15] = 0.0f;
		sdata[tx+16] = 0.0f;
		sdata[tx+17] = 0.0f;
		sdata[tx+18] = 0.0f;
		sdata[tx+19] = 0.0f;
		sdata[tx+20] = 0.0f;
		sdata[tx+21] = 0.0f;
		sdata[tx+22] = 0.0f;
		sdata[tx+23] = 0.0f;
		sdata[tx+24] = 0.0f;
		sdata[tx+25] = 0.0f;
		sdata[tx+26] = 0.0f;
	}
	__syncthreads();
	size_t index2 = 0;
	for (size_t i = blockDim.x/2; i>0; i>>=1) {
		if (threadIdx.x < i) {
			index2 = tx + i*27;
			sdata[tx   ] += sdata[index2   ];
			sdata[tx+1 ] += sdata[index2+1 ];
			sdata[tx+2 ] += sdata[index2+2 ];
			sdata[tx+3 ] += sdata[index2+3 ];
			sdata[tx+4 ] += sdata[index2+4 ];
			sdata[tx+5 ] += sdata[index2+5 ];
			sdata[tx+6 ] += sdata[index2+6 ];
			sdata[tx+7 ] += sdata[index2+7 ];
			sdata[tx+8 ] += sdata[index2+8 ];
			sdata[tx+9 ] += sdata[index2+9 ];
			sdata[tx+10] += sdata[index2+10];
			sdata[tx+11] += sdata[index2+11];
			sdata[tx+12] += sdata[index2+12];
			sdata[tx+13] += sdata[index2+13];
			sdata[tx+14] += sdata[index2+14];
			sdata[tx+15] += sdata[index2+15];
			sdata[tx+16] += sdata[index2+16];
			sdata[tx+17] += sdata[index2+17];
			sdata[tx+18] += sdata[index2+18];
			sdata[tx+19] += sdata[index2+19];
			sdata[tx+20] += sdata[index2+20];
			sdata[tx+21] += sdata[index2+21];
			sdata[tx+22] += sdata[index2+22];
			sdata[tx+23] += sdata[index2+23];
			sdata[tx+24] += sdata[index2+24];
			sdata[tx+25] += sdata[index2+25];
			sdata[tx+26] += sdata[index2+26];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		size_t bidx = blockIdx.x*27;
		redArr[bidx   ] = sdata[0 ];
		redArr[bidx+1 ] = sdata[1 ];
		redArr[bidx+2 ] = sdata[2 ];
		redArr[bidx+3 ] = sdata[3 ];
		redArr[bidx+4 ] = sdata[4 ];
		redArr[bidx+5 ] = sdata[5 ];
		redArr[bidx+6 ] = sdata[6 ];
		redArr[bidx+7 ] = sdata[7 ];
		redArr[bidx+8 ] = sdata[8 ];
		redArr[bidx+9 ] = sdata[9 ];
		redArr[bidx+10] = sdata[10];
		redArr[bidx+11] = sdata[11];
		redArr[bidx+12] = sdata[12];
		redArr[bidx+13] = sdata[13];
		redArr[bidx+14] = sdata[14];
		redArr[bidx+15] = sdata[15];
		redArr[bidx+16] = sdata[16];
		redArr[bidx+17] = sdata[17];
		redArr[bidx+18] = sdata[18];
		redArr[bidx+19] = sdata[19];
		redArr[bidx+20] = sdata[20];
		redArr[bidx+21] = sdata[21];
		redArr[bidx+22] = sdata[22];
		redArr[bidx+23] = sdata[23];
		redArr[bidx+24] = sdata[24];
		redArr[bidx+25] = sdata[25];
		redArr[bidx+26] = sdata[26];
	}
}

__global__ void reducePt2(float *d_redArr, float *d_redArr2, int n) {
	extern __shared__ float sdata[];
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int tx = threadIdx.x*27;
	// load input into __shared__ memory
	if(i < n) {
		sdata[tx+0 ] = d_redArr[i*27   ];
		sdata[tx+1 ] = d_redArr[i*27+1 ];
		sdata[tx+2 ] = d_redArr[i*27+2 ];
		sdata[tx+3 ] = d_redArr[i*27+3 ];
		sdata[tx+4 ] = d_redArr[i*27+4 ];
		sdata[tx+5 ] = d_redArr[i*27+5 ];
		sdata[tx+6 ] = d_redArr[i*27+6 ];
		sdata[tx+7 ] = d_redArr[i*27+7 ];
		sdata[tx+8 ] = d_redArr[i*27+8 ];
		sdata[tx+9 ] = d_redArr[i*27+9 ];
		sdata[tx+10] = d_redArr[i*27+10];
		sdata[tx+11] = d_redArr[i*27+11];
		sdata[tx+12] = d_redArr[i*27+12];
		sdata[tx+13] = d_redArr[i*27+13];
		sdata[tx+14] = d_redArr[i*27+14];
		sdata[tx+15] = d_redArr[i*27+15];
		sdata[tx+16] = d_redArr[i*27+16];
		sdata[tx+17] = d_redArr[i*27+17];
		sdata[tx+18] = d_redArr[i*27+18];
		sdata[tx+19] = d_redArr[i*27+19];
		sdata[tx+20] = d_redArr[i*27+20];
		sdata[tx+21] = d_redArr[i*27+21];
		sdata[tx+22] = d_redArr[i*27+22];
		sdata[tx+23] = d_redArr[i*27+23];
		sdata[tx+24] = d_redArr[i*27+24];
		sdata[tx+25] = d_redArr[i*27+25];
		sdata[tx+26] = d_redArr[i*27+26];
	} else {
		sdata[tx+0 ] = 0.0f;
		sdata[tx+1 ] = 0.0f;
		sdata[tx+2 ] = 0.0f;
		sdata[tx+3 ] = 0.0f;
		sdata[tx+4 ] = 0.0f;
		sdata[tx+5 ] = 0.0f;
		sdata[tx+6 ] = 0.0f;
		sdata[tx+7 ] = 0.0f;
		sdata[tx+8 ] = 0.0f;
		sdata[tx+9 ] = 0.0f;
		sdata[tx+10] = 0.0f;
		sdata[tx+11] = 0.0f;
		sdata[tx+12] = 0.0f;
		sdata[tx+13] = 0.0f;
		sdata[tx+14] = 0.0f;
		sdata[tx+15] = 0.0f;
		sdata[tx+16] = 0.0f;
		sdata[tx+17] = 0.0f;
		sdata[tx+18] = 0.0f;
		sdata[tx+19] = 0.0f;
		sdata[tx+20] = 0.0f;
		sdata[tx+21] = 0.0f;
		sdata[tx+22] = 0.0f;
		sdata[tx+23] = 0.0f;
		sdata[tx+24] = 0.0f;
		sdata[tx+25] = 0.0f;
		sdata[tx+26] = 0.0f;
	}
	__syncthreads();
	// block-wide reduction in __shared__ mem
	size_t index2 = 0;
	for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if(threadIdx.x < offset)
		{
			index2 = tx + offset*27;
			// add a partial sum upstream to our own
			sdata[tx   ] += sdata[index2   ];
			sdata[tx+1 ] += sdata[index2+1 ];
			sdata[tx+2 ] += sdata[index2+2 ];
			sdata[tx+3 ] += sdata[index2+3 ];
			sdata[tx+4 ] += sdata[index2+4 ];
			sdata[tx+5 ] += sdata[index2+5 ];
			sdata[tx+6 ] += sdata[index2+6 ];
			sdata[tx+7 ] += sdata[index2+7 ];
			sdata[tx+8 ] += sdata[index2+8 ];
			sdata[tx+9 ] += sdata[index2+9 ];
			sdata[tx+10] += sdata[index2+10];
			sdata[tx+11] += sdata[index2+11];
			sdata[tx+12] += sdata[index2+12];
			sdata[tx+13] += sdata[index2+13];
			sdata[tx+14] += sdata[index2+14];
			sdata[tx+15] += sdata[index2+15];
			sdata[tx+16] += sdata[index2+16];
			sdata[tx+17] += sdata[index2+17];
			sdata[tx+18] += sdata[index2+18];
			sdata[tx+19] += sdata[index2+19];
			sdata[tx+20] += sdata[index2+20];
			sdata[tx+21] += sdata[index2+21];
			sdata[tx+22] += sdata[index2+22];
			sdata[tx+23] += sdata[index2+23];
			sdata[tx+24] += sdata[index2+24];
			sdata[tx+25] += sdata[index2+25];
			sdata[tx+26] += sdata[index2+26];
		}
		__syncthreads();
	}
	// finally, thread 0 writes the result
	if(threadIdx.x == 0)
	{
		// note that the result is per-block
		// not per-thread
		size_t bidx = blockIdx.x*27;
		d_redArr2[bidx   ] = sdata[0 ];
		d_redArr2[bidx+1 ] = sdata[1 ];
		d_redArr2[bidx+2 ] = sdata[2 ];
		d_redArr2[bidx+3 ] = sdata[3 ];
		d_redArr2[bidx+4 ] = sdata[4 ];
		d_redArr2[bidx+5 ] = sdata[5 ];
		d_redArr2[bidx+6 ] = sdata[6 ];
		d_redArr2[bidx+7 ] = sdata[7 ];
		d_redArr2[bidx+8 ] = sdata[8 ];
		d_redArr2[bidx+9 ] = sdata[9 ];
		d_redArr2[bidx+10] = sdata[10];
		d_redArr2[bidx+11] = sdata[11];
		d_redArr2[bidx+12] = sdata[12];
		d_redArr2[bidx+13] = sdata[13];
		d_redArr2[bidx+14] = sdata[14];
		d_redArr2[bidx+15] = sdata[15];
		d_redArr2[bidx+16] = sdata[16];
		d_redArr2[bidx+17] = sdata[17];
		d_redArr2[bidx+18] = sdata[18];
		d_redArr2[bidx+19] = sdata[19];
		d_redArr2[bidx+20] = sdata[20];
		d_redArr2[bidx+21] = sdata[21];
		d_redArr2[bidx+22] = sdata[22];
		d_redArr2[bidx+23] = sdata[23];
		d_redArr2[bidx+24] = sdata[24];
		d_redArr2[bidx+25] = sdata[25];
		d_redArr2[bidx+26] = sdata[26];
	}
}

// __global__ void fillAb (float *res, float *A, float *b) {
// 	size_t idx = threadIdx.x;
	
// 	if (idx<21) {
// 		A[idx] = res[idx];
// 	} else if (idx<26) {
// 		b[idx] = res[idx];
// 	}
// }

void multiplyAndReduce(float *d_J, float *d_res, float *d_redArr, float *d_redArr2, int lvl, int w, int h, Mat6f *A, Vec6f *b) {
		
	// for (int i=0; i<lvl; i++) {
	// 	w = (w+1)/2;
	// 	h = (h+1)/2;
	// }

	dim3 block = dim3(256, 1, 1);
	int blockNum = ((size_t)w*h+block.x-1)/block.x;
	dim3 grid = dim3(blockNum, 1, 1);
	size_t shBytes = block.x*27*sizeof(float);
	
	multiplyAndReducePt1 <<<grid, block, shBytes>>> (d_J, d_res, d_redArr, w, h);
	
	// float *tA = new float[27];
	// cudaMemcpy(tA, d_redArr, 27*sizeof(float), cudaMemcpyDeviceToHost);
	// *A << tA[0 ], tA[1 ], tA[2 ], tA[3 ], tA[4 ], tA[5 ],
	// 	  tA[1 ], tA[6 ], tA[7 ], tA[8 ], tA[9 ], tA[10],
	// 	  tA[2 ], tA[7 ], tA[11], tA[12], tA[13], tA[14],
	// 	  tA[3 ], tA[8 ], tA[12], tA[15], tA[16], tA[17],
	// 	  tA[4 ], tA[9 ], tA[13], tA[16], tA[18], tA[19],
	// 	  tA[5 ], tA[10], tA[14], tA[17], tA[19], tA[20];
	// *b << tA[21], tA[22], tA[23], tA[24], tA[25], tA[26];
	// std::cout << "A = " << *A << std::endl;
	// std::cout << "b = " << *b << std::endl;

	size_t alt = blockNum;
	float *temp = 0;
	while (blockNum>1) {
		blockNum = (alt+block.x-1)/block.x;
		grid = dim3(blockNum, 1, 1);
		reducePt2 <<<grid, block, shBytes>>> (d_redArr, d_redArr2, alt);
		if (blockNum>1) {
			temp = d_redArr;
			d_redArr = d_redArr2;
			d_redArr2 = temp;
		}
		alt = blockNum;
	}

	// fillAb <<<dim3(1,1,1), dim3(32,1,1)>>> (d_redArr2, d_A, d_b);
	float *tA = new float[27];
	cudaMemcpy(tA, d_redArr2, 27*sizeof(float), cudaMemcpyDeviceToHost);
	*A << tA[0], tA[1] , tA[2] , tA[3] , tA[4] , tA[5] ,
		  tA[1], tA[6] , tA[7] , tA[8] , tA[9] , tA[10],
		  tA[2], tA[7] , tA[11], tA[12], tA[13], tA[14],
		  tA[3], tA[8] , tA[12], tA[15], tA[16], tA[17],
		  tA[4], tA[9] , tA[13], tA[16], tA[18], tA[19],
		  tA[5], tA[10], tA[14], tA[17], tA[19], tA[20];
	*b << tA[21], tA[22] , tA[23] , tA[24] , tA[25] , tA[26];
	// std::cout << "A = " << *A << std::endl;
	// std::cout << "b = " << *b << std::endl;

	delete[] tA;

}