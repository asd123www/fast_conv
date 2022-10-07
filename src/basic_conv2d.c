/*
  Parallel Computing II, Spring 2022.
  Instructor: Prof. Chao Yang @ Peking University.
  This is a naive implement of basic batch 2D convolution.
  Version 0.1 (date: 03/24/2022)
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <immintrin.h>

#define BATCH_SIZE 4
#define CIN 64
#define COUT 64
#define KW 3
#define KH 3
#define H 256
#define W 256
#define SEED 2345
#define SAVE
#define input_offset 4

#define VLEN 32


FILE *inFile, *kFile, *outFile;


double d;
double input[BATCH_SIZE][CIN][H][W];
double my_input[BATCH_SIZE][CIN][H + 10][W + 10];

double kernel[COUT][CIN][KH][KW];
double output[BATCH_SIZE][COUT][H][W];
double time0, time1;


double get_walltime()
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

void init() {

#ifdef SAVE
  inFile = fopen("in.txt", "wb");
  kFile = fopen("kernel.txt", "wb");
  outFile = fopen("out.txt", "wb");
#endif

// generate the data.
  //  initialize data
  srand(SEED);

  for (int b = 0; b < BATCH_SIZE; b++)
  {
    for (int c1 = 0; c1 < CIN; c1++)
    {
      for (int i = 0; i < H + 10; i++)
      {
        for (int j = 0; j < W + 10; j++)
        {
          my_input[b][c1][i][j] =  0;
        }
      }
    }
  }
  for (int b = 0; b < BATCH_SIZE; b++)
  {
    for (int c1 = 0; c1 < CIN; c1++)
    {
      for (int i = 0; i < H; i++)
      {
        for (int j = 0; j < W; j++)
        {
          my_input[b][c1][i + input_offset][j + input_offset] = input[b][c1][i][j] = (double)rand() / RAND_MAX;
        }
      }
    }
  }
#ifdef SAVE
  fwrite(&(input[0][0][0][0]), sizeof(double), BATCH_SIZE * CIN * H * W, inFile);
#endif
  for (int c2 = 0; c2 < COUT; c2++)
  {
    for (int c1 = 0; c1 < CIN; c1++)
    {
      for (int i = 0; i < KH; i++)
      {
        for (int j = 0; j < KW; j++)
        {
          kernel[c2][c1][i][j] = (double)rand() / RAND_MAX;
        }
      }
    }
  }
#ifdef SAVE
  fwrite(&(kernel[0][0][0][0]), sizeof(double), COUT * CIN * KH * KW, kFile);
#endif

for (int b = 0; b < BATCH_SIZE; b++)
  {
    for (int c2 = 0; c2 < COUT; c2++)
    {
      for (int i = 0; i < H; i++)
      {
        for (int j = 0; j < W; j++)
        {
          output[b][c2][i][j] = 0;
        }
      }
    }
  }
}



//  naive implementation
void func0() {
  // 19.408794.
  for (int b = 0; b < BATCH_SIZE; b++)
  {
    for (int c2 = 0; c2 < COUT; c2++)
    {
      for (int c1 = 0; c1 < CIN; c1++)
      {
        for (int i = 0; i < H; i++)
        {
          for (int j = 0; j < W; j++)
          {
            for (int ki = -(KH / 2); ki < 1 + KH / 2; ki++)
            {
              for (int kj = -(KW / 2); kj < 1 + KW / 2; kj++)
              {
                d = (i + ki >= 0 && i + ki < H && j + kj >= 0 && j + kj < W) ? input[b][c1][i + ki][j + kj] : 0.0;
                output[b][c2][i][j] += d * kernel[c2][c1][KH / 2 + ki][KW / 2 + kj];
              }
            }
          }
        }
      }
    }
  }
}

// 去掉计算, 我们本来可以提前计算一些值.
void func1() {
  // 18.642063 in server.
  int half_KH = KH / 2;
  int half_KW = KW / 2;

  for (int b = 0; b < BATCH_SIZE; ++b)
  {
    for (int c2 = 0; c2 < COUT; ++c2)
    {
      for (int c1 = 0; c1 < CIN; ++c1)
      {
        for (int i = 0; i < H; ++i)
        {
          for (int j = 0; j < W; ++j)
          {
            for (int ki = -half_KH; ki <= half_KH; ++ki)
            {
              for (int kj = -half_KW; kj <= half_KW; ++kj)
              {
                d = (i + ki >= 0 && i + ki < H && j + kj >= 0 && j + kj < W) ? input[b][c1][i + ki][j + kj] : 0.0;
                output[b][c2][i][j] += d * kernel[c2][c1][half_KH + ki][half_KW + kj];
              }
            }
          }
        }
      }
    }
  }
}


// 去掉分支判断, 直接加入zero-padding.
void func2() {
  // 14.336702.
  int half_KH = KH / 2;
  int half_KW = KW / 2;

  for (int b = 0; b < BATCH_SIZE; ++b)
  {
    for (int c2 = 0; c2 < COUT; ++c2)
    {
      for (int c1 = 0; c1 < CIN; ++c1)
      {
        for (int i = 0; i < H; ++i)
        {
          for (int j = 0; j < W; ++j)
          {
            int ii = i + input_offset;
            int jj = j + input_offset;
            for (int ki = -half_KH; ki <= half_KH; ++ki)
            {
              for (int kj = -half_KW; kj <= half_KW; ++kj)
              {
                d = my_input[b][c1][ii + ki][jj + kj];
                output[b][c2][i][j] += d * kernel[c2][c1][half_KH + ki][half_KW + kj];
              }
            }
          }
        }
      }
    }
  }
}

// 继续整理结构, 在stack中开启一些变量, 去掉底层循环的数组访问.
void func3() {
	// 15.541136 in server.
	int half_KH = KH / 2;
	int half_KW = KW / 2;
	double t, result;
	int ii, jj;

	for (int b = 0; b < BATCH_SIZE; ++b) {
		for (int c2 = 0; c2 < COUT; ++c2) {
			for (int c1 = 0; c1 < CIN; ++c1) {
				for (int i = 0; i < H; ++i) {
					for (int j = 0; j < W; ++j) {
						ii = i + input_offset;
						jj = j + input_offset;
						result = 0;
						for (int ki = -half_KH; ki <= half_KH; ++ki) {
							for (int kj = -half_KW; kj <= half_KW; ++kj) {
								t = my_input[b][c1][ii + ki][jj + kj];
								result += t * kernel[c2][c1][half_KH + ki][half_KW + kj];
							}
						}
						output[b][c2][i][j] += result;
					}
				}
			}
		}
	}
}

// 观察发现my_kernel可以优化放到stack上.
void func4() {
	// 16.427931 in server.
	int half_KH = KH / 2;
	int half_KW = KW / 2;
	double t, result;
	int ii, jj;
	double my_kernel[KH][KW];

	for (int b = 0; b < BATCH_SIZE; ++b) {
		for (int c2 = 0; c2 < COUT; ++c2) {
			for (int c1 = 0; c1 < CIN; ++c1) {
				memcpy(my_kernel, kernel[c2][c1], sizeof(my_kernel));

				for (int i = 0; i < H; ++i) {
					for (int j = 0; j < W; ++j) {
						ii = i + input_offset;
						jj = j + input_offset;
						result = 0;
						for (int ki = -half_KH; ki <= half_KH; ++ki) {
							for (int kj = -half_KW; kj <= half_KW; ++kj) {
								t = my_input[b][c1][ii + ki][jj + kj];
								result += t * my_kernel[half_KH + ki][half_KW + kj];
							}
						}
						output[b][c2][i][j] += result;
					}
				}
			}
		}
	}
}

// 观察发现my_kernel可以优化放到stack上. 加上循环展开.
void func5() {
	// 4.985553 in server.
	double result;
	int ii, jj;
	double my_kernel[KH][KW];

	for (int b = 0; b < BATCH_SIZE; ++b) {
		for (int c2 = 0; c2 < COUT; ++c2) {
			for (int c1 = 0; c1 < CIN; ++c1) {
				memcpy(my_kernel, kernel[c2][c1], sizeof(my_kernel));

				for (int i = 0; i < H; ++i) {
					for (int j = 0; j < W; ++j) {
						ii = i + input_offset;
						jj = j + input_offset;
						result = 0;
						result += my_kernel[0][0] * my_input[b][c1][ii - 1][jj - 1];
						result += my_kernel[0][1] * my_input[b][c1][ii - 1][jj];
						result += my_kernel[0][2] * my_input[b][c1][ii - 1][jj + 1];
						result += my_kernel[1][0] * my_input[b][c1][ii][jj - 1];
						result += my_kernel[1][1] * my_input[b][c1][ii][jj];
						result += my_kernel[1][2] * my_input[b][c1][ii][jj + 1];
						result += my_kernel[2][0] * my_input[b][c1][ii + 1][jj - 1];
						result += my_kernel[2][1] * my_input[b][c1][ii + 1][jj];
						result += my_kernel[2][2] * my_input[b][c1][ii + 1][jj + 1];
						output[b][c2][i][j] += result;
					}
				}
			}
		}
	}
}


// xjb优化.
void func6() {
	// 5.000144 in server.
	double result;
	double *to, *pt0, *pt1, *pt2;
	double my_kernel[KH * KW];

	for (int b = 0; b < BATCH_SIZE; ++b) {
		for (int c2 = 0; c2 < COUT; ++c2) {
			for (int c1 = 0; c1 < CIN; ++c1) {
				memcpy(my_kernel, kernel[c2][c1], sizeof(my_kernel));
				for (int i = 0; i < H; ++i) {
					to = output[b][c2][i];
					pt0 = my_input[b][c1][i + input_offset - 1] + input_offset - 1;
					pt1 = my_input[b][c1][i + input_offset] + input_offset - 1;
					pt2 = my_input[b][c1][i + input_offset + 1] + input_offset - 1;

					for (int j = 0; j < W; ++j, ++to, ++pt0, ++pt1, ++pt2) {
						result = my_kernel[0] * *pt0;
						result += my_kernel[1] * *(pt0 + 1);
						result += my_kernel[2] * *(pt0 + 2);
						result += my_kernel[3] * *pt1;
						result += my_kernel[4] * *(pt1 + 1);
						result += my_kernel[5] * *(pt1 + 2);
						result += my_kernel[6] * *pt2;
						result += my_kernel[7] * *(pt2 + 1);
						result += my_kernel[8] * *(pt2 + 2);
						*to += result;
					}
				}
			}
		}
	}
}

// 加入SIMD优化?
void func7() {
  // 3.689201 in server.
	double *to, *pt0, *pt1, *pt2;
	double my_kernel[KH * KW];

	for (int b = 0; b < BATCH_SIZE; ++b) {
		for (int c2 = 0; c2 < COUT; ++c2) {
			for (int c1 = 0; c1 < CIN; ++c1) {
				memcpy(my_kernel, kernel[c2][c1], sizeof(my_kernel));
        double kk1[4] = {0}, kk2[4] = {0}, kk3[4] = {0};
        kk1[0] = my_kernel[0], kk1[1] = my_kernel[1], kk1[2] = my_kernel[2]; 
        kk2[0] = my_kernel[3], kk2[1] = my_kernel[4], kk2[2] = my_kernel[5]; 
        kk3[0] = my_kernel[6], kk3[1] = my_kernel[7], kk3[2] = my_kernel[8];
        __m256d k1 = _mm256_load_pd(kk1);
        __m256d k2 = _mm256_load_pd(kk2);
        __m256d k3 = _mm256_load_pd(kk3);

				for (int i = 0; i < H; ++i) {
					to = output[b][c2][i];
					pt0 = my_input[b][c1][i + input_offset - 1] + input_offset - 1;
					pt1 = my_input[b][c1][i + input_offset] + input_offset - 1;
					pt2 = my_input[b][c1][i + input_offset + 1] + input_offset - 1;

					for (int j = 0; j < W; ++j, ++to, ++pt0, ++pt1, ++pt2) {
            __m256d res = _mm256_set_pd(0, 0, 0, 0);
            __m256d a = _mm256_load_pd(pt0);
            __m256d b = _mm256_load_pd(pt1);
            __m256d c = _mm256_load_pd(pt2);
            res = _mm256_fmadd_pd(a, k1, res);
            res = _mm256_fmadd_pd(b, k2, res);
            res = _mm256_fmadd_pd(c, k3, res);
            res = _mm256_hadd_pd(res, res);
            
						*to += ((double*)&res)[0] + ((double*)&res)[2];
					}
				}
			}
		}
	}
}


// improved-SIMD
double I[BATCH_SIZE][CIN/VLEN][H + 10][W + 10][VLEN];
double O[BATCH_SIZE][COUT/VLEN][H][W][VLEN];
double Wei[COUT/VLEN][CIN/VLEN][KH][KW][VLEN][VLEN];
void func8() {
  // 1.62s
  int half_KH = KH / 2;
  int half_KW = KW / 2;

  int Kb = COUT/VLEN;
  int Cb = CIN/VLEN;


  
  for (int n = 0; n < BATCH_SIZE; ++n) {
    for (int cb = 0; cb < Cb; ++cb) {
      for (int i = 0; i < H + input_offset; ++i) {
        for (int j = 0; j < W + input_offset; ++j) {
            for (int c = 0; c < VLEN; ++c) {
              I[n][cb][i][j][c] = my_input[n][cb*VLEN + c][i][j];
            }
        }
      }
    }
  }

  for (int kb = 0; kb < Kb; ++kb) {
    for (int cb = 0; cb < Cb; ++cb) {
      for(int r = 0; r < KH; ++r) for (int s = 0; s < KW; ++s) 
      for (int k = 0; k < VLEN; ++k) {
        for (int c = 0; c < VLEN; ++c) {
          Wei[kb][cb][r][s][k][c] = kernel[kb*VLEN + k][cb*VLEN + c][r][s];
        }
      }
    }
  }

  for (int n = 0; n < BATCH_SIZE; ++n) {
    for (int kb = 0; kb < Kb; ++kb) {
      for (int cb = 0; cb < Cb; ++cb) {
        for (int i = 0; i < H; ++i) {
          for (int j = 0; j < W; ++j) {

            int ii = i + input_offset;
            int jj = j + input_offset;
            // little kernel.

            __m256d val[VLEN];
            for (int k = 0; k < VLEN; ++k) val[k] = _mm256_set_pd(0, 0, 0, 0);
            for (int ki = -half_KH; ki <= half_KH; ++ki) {
              for (int kj = -half_KW; kj <= half_KW; ++kj) {
                double *pt1 = I[n][cb][ii + ki][jj + kj];
                double *pt0 = Wei[kb][cb][half_KH + ki][half_KW + kj][0];
                for (int k = 0; k < VLEN; ++k) {
                  for (int c = 0; c < VLEN; c += 4) {
                    val[k] = _mm256_fmadd_pd(_mm256_load_pd(pt0), _mm256_load_pd(pt1), val[k]);
                    pt0 += 4;
                    pt1 += 4;
                  }
                  pt1 -= VLEN;
                }
              }
            }
            for (int k = 0; k < VLEN; ++k) {
              val[k] = _mm256_hadd_pd(val[k], val[k]);
              O[n][kb][i][j][k] += ((double*)&val[k])[0] + ((double*)&val[k])[2];
            }
          }
        }
      }
    }
  }

  for (int n = 0; n < BATCH_SIZE; ++n) {
    for (int kb = 0; kb < Kb; ++kb) {
      for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
          for (int k = 0; k < VLEN; ++k) {
            output[n][kb*VLEN + k][i][j] = O[n][kb][i][j][k];
          }
        }
      }
    }
  }
}


int main(int argc, char *argv[]) {
  init();

  time0 = get_walltime();
  // func0();
  // func2();
  // func5();
  // func7();
  func8();
  time1 = get_walltime() - time0;


#ifdef SAVE
  fwrite(&(output[0][0][0][0]), sizeof(double), BATCH_SIZE * COUT * H * W, outFile);
#endif
  printf("Batch size: %d, Matrix size: %d x %d, kernel_size: %d, channels:%d->%d\n", BATCH_SIZE, H, W, KH, CIN, COUT);
  printf("Wall time: %f\n", time1);
  printf("The first and last entries: %f, %f\n", output[0][0][0][0], output[BATCH_SIZE - 1][COUT - 1][H - 1][W - 1]);

  return 0;
}