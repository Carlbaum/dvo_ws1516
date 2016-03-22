#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <sophus/se3.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


void updateXi(std::vector<Eigen::Matrix3f> &kPyramid, std::vector<cv::Mat> &grayRefPyramid, 
	      std::vector<cv::Mat> &depthRefPyramid, std::vector<cv::Mat> &grayCurPyramid,
	      Eigen::Matrix3f &rot, Eigen::Vector3f &t, Vec6f &xi)
{
	for (int level = LVL; level>0 ; level--)
	{
		cv::Mat grayRef = grayRefPyramid[level-1];
		cv::Mat depthRef = depthRefPyramid[level-1];
		cv::Mat grayCur = grayCurPyramid[level-1];
		Eigen::Matrix3f kLevel = kPyramid[level-1];

		// get image dimensions
		int w = grayRef.cols;         // width
		int h = grayRef.rows;         // height

		// initialize cuda context
		cudaDeviceSynchronize();  CUDA_CHECK;

		// copy data to device
		// Current grayscale image
		float nbytes = w*h*sizeof(float);
		float *d_currImg;
		cudaMalloc(&d_currImg, nbytes);  CUDA_CHECK;
		cudaMemcpy(d_currImg, (void*)grayCur.data, nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;

		// Reference grayscale image
		float *d_refimgIn;
		cudaMalloc(&d_refimgIn, nbytes);  CUDA_CHECK;
		cudaMemcpy(d_refimgIn, (void*)grayRef.data, nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;

		//Reference depth image
		float *d_refdepthImgIn;
		cudaMalloc(&d_refdepthImgIn, nbytes);  CUDA_CHECK;
		cudaMemcpy(d_refdepthImgIn, (void*)depthRef.data,nbytes,cudaMemcpyHostToDevice);  CUDA_CHECK;

		//Residual Image
		float *d_resImg;
		cudaMalloc(&d_resImg,  nbytes);  CUDA_CHECK;
		cudaMemset(d_resImg, 0, nbytes);  CUDA_CHECK;

		float fx = kLevel(0,0);
		float fy = kLevel(1,1);
		float cx = kLevel(0,2);
		float cy = kLevel(1,2); 

		float *d_vx, *d_vy, *d_jacobif,*JtJ_final, *Jtb_final, *d_temp, *d_result, *d_rot, *d_t;
		size_t n_d_vx = (size_t)w*h;
		size_t n_d_vy = (size_t)w*h;
		int N = w*h;

		cudaMalloc(&d_jacobif, N*6*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_temp, N*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_result, N*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&JtJ_final, N*21*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&Jtb_final, N*6*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_vx,  n_d_vx*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_vy,  n_d_vy*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_rot,  9*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_t,  3*sizeof(float));  CUDA_CHECK;

		float *JTJ  = new float[(size_t)N*21];
		float *JTB  = new float[(size_t)N*6];
		float *temp = new float[(size_t)N];
		float *result = new float[(size_t)N];
		float *A_interim = new float[(size_t)21];
		float *b_interim = new float[(size_t)6];
		float errLast = std::numeric_limits<float>::max();
		for(int i = 0; i < ITER ; i++)
		{
			cudaMemcpy(d_currImg, (void*)grayCur.data, nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;
			cudaMemcpy(d_refimgIn, (void*)grayRef.data, nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;
			cudaMemcpy(d_refdepthImgIn, (void*)depthRef.data, nbytes, cudaMemcpyHostToDevice);  				CUDA_CHECK;

			float *rot_data = rot.data();
			float *t_data = t.data();

			cudaMemset(d_jacobif, 0, N*6*sizeof(float));  CUDA_CHECK;
			cudaMemset(d_temp, 0, N*sizeof(float));  CUDA_CHECK;
			cudaMemset(d_result, 0, N*sizeof(float));  CUDA_CHECK;
			cudaMemset(JtJ_final, 0, N*21*sizeof(float));  CUDA_CHECK;
			cudaMemset(Jtb_final, 0, N*6*sizeof(float));  CUDA_CHECK;
			cudaMemset(d_vx, 0, n_d_vx*sizeof(float));  CUDA_CHECK;
			cudaMemset(d_vy, 0, n_d_vy*sizeof(float));  CUDA_CHECK;
			cudaMemcpy(d_rot,rot_data,9*sizeof(float),cudaMemcpyHostToDevice); CUDA_CHECK;
			cudaMemcpy(d_t,t_data,3*sizeof(float),cudaMemcpyHostToDevice); CUDA_CHECK;

			dim3 block = dim3(32, 8, 1);
			dim3 grid = dim3((w+block.x-1)/block.x, (h+block.y-1)/block.y, 1);

			//Texture Memory
			texRef.addressMode[0] = cudaAddressModeClamp;
			texRef.addressMode[1] = cudaAddressModeClamp;
			texRef.filterMode = cudaFilterModeLinear;
			texRef.normalized = false;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cudaBindTexture2D(NULL, &texRef, d_currImg, &desc, w, h, w*sizeof(d_currImg[0]));
			 
			texGradX.addressMode[0] = cudaAddressModeClamp;
			texGradX.addressMode[1] = cudaAddressModeClamp;
			texGradX.filterMode = cudaFilterModeLinear;
			texGradX.normalized = false;
			cudaChannelFormatDesc descX = cudaCreateChannelDesc<float>();
			cudaBindTexture2D(NULL, &texGradX, d_vx, &descX, w, h, w*sizeof(d_vx[0]));

			texGradY.addressMode[0] = cudaAddressModeClamp;
			texGradY.addressMode[1] = cudaAddressModeClamp;
			texGradY.filterMode = cudaFilterModeLinear;
			texGradY.normalized = false;
			cudaChannelFormatDesc descY = cudaCreateChannelDesc<float>();
			cudaBindTexture2D(NULL, &texGradY, d_vy, &descY, w, h, w*sizeof(d_vy[0]));

			//cudaMemset(d_imgOut, 0, nbytes);  CUDA_CHECK;
			cudaMemset(d_resImg, 0, nbytes);  CUDA_CHECK;


			calcErr <<<grid,block>>> (d_refimgIn,d_currImg,d_refdepthImgIn,d_resImg,d_rot,d_t,fx,fy,cx,cy,w,h);
			CUDA_CHECK;
			gradCompute <<<grid,block>>> (d_currImg,d_vx,d_vy,w,h); CUDA_CHECK;

			deriveNumeric <<<grid,block>>>(d_vx,d_vy,d_refdepthImgIn,d_resImg,d_jacobif,w,h,fx,fy,cx,cy,d_rot,d_t,JtJ_final,Jtb_final); //d_imgOut,
			CUDA_CHECK;

			cv::Mat residualGPU(h,w,grayRef.type());
			cudaMemcpy((void *)residualGPU.data,d_resImg,N*sizeof(float),cudaMemcpyDeviceToHost); CUDA_CHECK;
			Eigen::VectorXf residual(N);

			int idx = 0;
			for(int i =0 ;i<w;i++)
			{
				for(int j =0 ;j<h;j++)
				{
					residual[idx] = residualGPU.at<float>(i,j);
					idx++;
				}
			}

			Mat6f A = Mat6f::Zero();
			Vec6f b = Vec6f::Zero();
			cudaMemcpy(JTJ,JtJ_final,N*21*sizeof(float),cudaMemcpyDeviceToHost); CUDA_CHECK;

			dim3 block1 = dim3(128, 1, 1);
			dim3 grid1 = dim3((N + block1.x -1)/block1.x,1,1);
			size_t smBytes = block1.x * block1.y * block1.z * sizeof(float);

			//Reduction for JtJ		
			for(int j = 0; j<21; j++)
			{
				for(int i =0; i<N; i++)
				{			
					temp[i] = JTJ[i+N*j];
				}
				cudaMemcpy(d_temp,temp,N*sizeof(float),cudaMemcpyHostToDevice); CUDA_CHECK;
				block_sum <<<grid1,block1,smBytes>>> (d_temp,d_result,N);
				for(int offset = block1.x / 2;offset > 0;offset >>= 1)
					block_sum <<<grid1,block1,smBytes>>> (d_result,d_result,N);
				cudaMemcpy(result,d_result,N*sizeof(float),cudaMemcpyDeviceToHost); CUDA_CHECK;
				A_interim[j]= result[0];	
			}

			int k = 0;	
			for(int i = 0; i<6; i++)
			{
				for(int j = i; j<6; j++)
				{
					A(i,j) = A(j,i) =A_interim[k];	
					k++;
				}
			}
			cudaMemset(d_result, 0, N*sizeof(float));  CUDA_CHECK;  //reuse for Jtb
			cudaMemset(d_temp, 0, N*sizeof(float));  CUDA_CHECK;
			cudaMemcpy(JTB,Jtb_final,N*6*sizeof(float),cudaMemcpyDeviceToHost); CUDA_CHECK;

			//Reduction for Jtb
			for(int j = 0; j<6; j++)
			{
				for(int i =0; i<N; i++)
				{			
					temp[i] = JTB[i+N*j];
				}
				cudaMemcpy(d_temp,temp,N*sizeof(float),cudaMemcpyHostToDevice); CUDA_CHECK;
				block_sum <<<grid1,block1,smBytes>>> (d_temp,d_result,N);
				for(int offset = block1.x / 2;offset > 0;offset >>= 1)
					block_sum <<<grid1,block1,smBytes>>> (d_result,d_result,N);
				cudaMemcpy(result,d_result,N*sizeof(float),cudaMemcpyDeviceToHost); CUDA_CHECK;
				b_interim[j]= result[0];
			}

			for(int i = 0; i<6; i++)
				b(i) = b_interim[i];

			// solve using Cholesky LDLT decomposition
			Vec6f delta = -(A.ldlt().solve(b));

			// update xi
			xi = Sophus::SE3f::log(Sophus::SE3f::exp(delta)*Sophus::SE3f::exp(xi));

			cudaUnbindTexture(texRef);
			cudaUnbindTexture(texGradX);
			cudaUnbindTexture(texGradY);

			// print out final pose
			convertSE3ToTf(xi, rot, t);		
			float error = (residual.cwiseProduct(residual)).mean();	
			if((error/errLast) > 0.995f)
				break;
			errLast = error;
		}
		delete[] JTJ;
		delete[] JTB;
		delete[] temp;
		delete[] result;
		delete[] A_interim;
		delete[] b_interim;

		cudaFree(d_rot); CUDA_CHECK;
		cudaFree(d_t); CUDA_CHECK;
		cudaFree(d_vx); CUDA_CHECK;
		cudaFree(d_vy); CUDA_CHECK;
		cudaFree(d_jacobif); CUDA_CHECK;
		cudaFree(d_result); CUDA_CHECK;
		cudaFree(d_temp); CUDA_CHECK;
		cudaFree(JtJ_final); CUDA_CHECK;
		cudaFree(Jtb_final); CUDA_CHECK;

		cudaFree(d_currImg); CUDA_CHECK;
		cudaFree(d_refimgIn); CUDA_CHECK;
		cudaFree(d_refdepthImgIn); CUDA_CHECK;
		cudaFree(d_resImg); CUDA_CHECK;
	}
}
