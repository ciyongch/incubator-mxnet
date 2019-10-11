/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2019 by Contributors
 * \file mkl_functions-inl.h
 * \brief Wrapper for MKL VML functions
 * \author Tao Lv, Shufan Wu
*/
#ifndef MXNET_OPERATOR_MKL_FUNCTIONS_INL_H_
#define MXNET_OPERATOR_MKL_FUNCTIONS_INL_H_

#if MSHADOW_USE_MKL == 1
#include "mkl_vml.h"

namespace mxnet {
namespace op {
namespace mkl_func {

MSHADOW_XINLINE
static bool check_size(const size_t n) {
  const size_t MKL_INT_MAX = (sizeof(MKL_INT) == sizeof(int)) ? INT_MAX : LLONG_MAX;
  return (n <= MKL_INT_MAX);
}

MSHADOW_XINLINE
static bool check_type(const int t) {
  return (t == mshadow::kFloat32 || t == mshadow::kFloat64);
}

#define MXNET_MKL_UNARY_MATH_FUNC(name, func)                                               \
struct name {                                                                               \
  MSHADOW_XINLINE static void Vectorize(const index_t n, const float *src, float *dst) {    \
    vs##func(static_cast<MKL_INT>(n), src, dst);                                            \
  }                                                                                         \
  MSHADOW_XINLINE static void Vectorize(const index_t n, const double *src, double *dst) {  \
    vd##func(static_cast<MKL_INT>(n), src, dst);                                            \
  }                                                                                         \
};

#define MXNET_MKL_BINARY_MATH_FUNC(name, func)                                        \
struct name {                                                                         \
  MSHADOW_XINLINE static void Vectorize(const index_t n,                              \
                                        const float *a,                               \
                                        const float *b,                               \
                                        float *c) {                                   \
    vs##func(static_cast<MKL_INT>(n), a, b, c);                                       \
  }                                                                                   \
  MSHADOW_XINLINE static void Vectorize(const index_t n,                              \
                                        const double *a,                              \
                                        const double *b,                              \
                                        double *c) {                                  \
    vd##func(static_cast<MKL_INT>(n), a, b, c);                                       \
  }                                                                                   \
};

MXNET_MKL_UNARY_MATH_FUNC(erf, Erf);
MXNET_MKL_UNARY_MATH_FUNC(exp, Exp);
MXNET_MKL_UNARY_MATH_FUNC(exp2, Exp2);
MXNET_MKL_UNARY_MATH_FUNC(exp10, Exp10);
MXNET_MKL_UNARY_MATH_FUNC(expm1, Expm1);
MXNET_MKL_UNARY_MATH_FUNC(log, Ln);
MXNET_MKL_UNARY_MATH_FUNC(log2, Log2);
MXNET_MKL_UNARY_MATH_FUNC(log10, Log10);
MXNET_MKL_UNARY_MATH_FUNC(log1p, Log1p);

MXNET_MKL_UNARY_MATH_FUNC(sin, Sin);
MXNET_MKL_UNARY_MATH_FUNC(cos, Cos);
MXNET_MKL_UNARY_MATH_FUNC(tan, Tan);
MXNET_MKL_UNARY_MATH_FUNC(asin, Asin);
MXNET_MKL_UNARY_MATH_FUNC(acos, Acos);
MXNET_MKL_UNARY_MATH_FUNC(atan, Atan);

MXNET_MKL_UNARY_MATH_FUNC(sinh, Sinh);
MXNET_MKL_UNARY_MATH_FUNC(cosh, Cosh);
MXNET_MKL_UNARY_MATH_FUNC(tanh, Tanh);
MXNET_MKL_UNARY_MATH_FUNC(asinh, Asinh);
MXNET_MKL_UNARY_MATH_FUNC(acosh, Acosh);
MXNET_MKL_UNARY_MATH_FUNC(atanh, Atanh);

MXNET_MKL_UNARY_MATH_FUNC(sqrt, Sqrt);
MXNET_MKL_UNARY_MATH_FUNC(abs, Abs);
MXNET_MKL_UNARY_MATH_FUNC(cbrt, Cbrt);
MXNET_MKL_UNARY_MATH_FUNC(round, Round);
MXNET_MKL_UNARY_MATH_FUNC(ceil, Ceil);
MXNET_MKL_UNARY_MATH_FUNC(floor, Floor);
MXNET_MKL_UNARY_MATH_FUNC(trunc, Trunc);

MXNET_MKL_UNARY_MATH_FUNC(lgamma, LGamma);
MXNET_MKL_UNARY_MATH_FUNC(tgamma, TGamma);
MXNET_MKL_UNARY_MATH_FUNC(square, Sqr);

MXNET_MKL_BINARY_MATH_FUNC(add, Add);
MXNET_MKL_BINARY_MATH_FUNC(sub, Sub);
MXNET_MKL_BINARY_MATH_FUNC(mul, Mul);
MXNET_MKL_BINARY_MATH_FUNC(pow, Pow);
MXNET_MKL_BINARY_MATH_FUNC(hypot, Hypot);

template <typename DType>
MSHADOW_XINLINE static void sum_(index_t n, DType *in, DType *dst) {
  DType sum = 0.0f;
  for (index_t i = 0; i < n; i++)
    sum += in[i];

  dst[0] = sum;
}


#ifndef CACHELINE_SIZE
# define CACHELINE_SIZE 64
#endif

#ifndef BYTE256_ALIGNED
# ifndef FOURLINE_ALIGNED
#  define MULTIPLE_CACHE_ALIGNED(factor) __attribute__ ((aligned ((factor)*CACHELINE_SIZE))) 
#  define FOURLINE_ALIGNED MULTIPLE_CACHE_ALIGNED(4)
# endif
# define BYTE256_ALIGNED FOURLINE_ALIGNED
#endif

template <typename DType>
struct aligned_dtype {
  public:
  DType volatile BYTE256_ALIGNED item;
};

// LayerNorm on the last dimension
template <typename DType>
MSHADOW_XINLINE static void LayerNormLastDim(index_t m,
                                             index_t n,
                                             DType *a,
                                             DType *b,
                                             DType *gamma,
                                             DType *beta,
                                             DType *mean,
                                             DType *var,
                                             DType eps) {
  auto nthreads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();

  std::vector<aligned_dtype<DType> > aligned_mean(m);
  std::vector<aligned_dtype<DType> > aligned_var(m);
#pragma omp parallel for num_threads(nthreads)
  for (index_t i = 0; i < m; i++) {
    DType* in_offset = a + i * n;
    DType* out_offset = b + i * n;
    DType x_sum = 0.0f;
    DType x_square_sum = 0.0f;

#if !defined(_MSC_VER)
#pragma omp simd
#endif
    for (index_t j = 0; j < n; j++) {
      x_sum += in_offset[j];
      x_square_sum += in_offset[j] * in_offset[j];
    }
    DType mean_ = x_sum / n;
    DType var_ = math::sqrt(x_square_sum / n - mean_ * mean_ + eps);

#if !defined(_MSC_VER)
#pragma omp simd
#endif
    for (index_t j = 0; j < n; j++) {
      out_offset[j] = (in_offset[j] - mean_) * gamma[j] / var_ + beta[j];
    }

    aligned_mean[i].item = mean_;
    aligned_var[i].item = var_;
  }

  for (index_t i = 0; i < m; i++) {
    mean[i] = aligned_mean[i].item;
    var[i] = aligned_var[i].item;
  }
}

}  // namespace mkl_func
}  // namespace op
}  // namespace mxnet
#endif  // MSHADOW_USE_MKL == 1
#endif  // MXNET_OPERATOR_MKL_FUNCTIONS_INL_H_
