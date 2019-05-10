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
 * Copyright (c) 2017 by Contributors
 * \file quantized_fully_connected.cc
 * \brief
 * \author Ziheng Jiang, Jun Wu
*/
#include <vector>
#include "quantization_utils.h"
#include "../nn/fully_connected-inl.h"
#if MXNET_USE_MKLDNN == 1
#include "../nn/mkldnn/mkldnn_fully_connected-inl.h"
#include "mkldnn/mkldnn_quantized_ops-inl.h"
#endif

namespace mxnet {
namespace op {

namespace quantized_fc {
enum QuantizedfcOpResource {kTempSpace};
}

class QuantizedFullyConnectedOp {
 public:
  explicit QuantizedFullyConnectedOp(const nnvm::NodeAttrs &attrs)
    : initialized_(false),
      param_(nnvm::get<FullyConnectedParam>(attrs.parsed)) {}

  template<typename DType>
  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &outputs);

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &outputs) {
    LOG(FATAL) << "Not implemented: QuantizedFullyConnected only supports "
                  "inference computation.";
  }

 private:
  bool initialized_;
  FullyConnectedParam param_;
  float cached_min_data_;
  float cached_max_data_;
  float cached_min_weight_;
  float cached_max_weight_;
  float cached_min_bias_;
  float cached_max_bias_;
  float cached_min_out_;
  float cached_max_out_;
  Tensor<cpu, 1, char> temp_space_;
};

bool QuantizedFullyConnectedShape(const nnvm::NodeAttrs& attrs,
                                  mxnet::ShapeVector *in_shape,
                                  mxnet::ShapeVector *out_shape) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  using namespace mshadow;
  uint32_t num_inputs = param.no_bias ? 2 : 3;
  CHECK_EQ(in_shape->size(), num_inputs * 3);
  CHECK_EQ(out_shape->size(), 3U);

  CHECK(shape_is_known(in_shape->at(0)))
    << "QuantizedFullyConnectedOp input data shape must be given";
  const mxnet::TShape& dshape = in_shape->at(0);
  index_t num_input;

  if (!param.trans_data) {
    if (!param.flatten) {
      num_input = dshape[dshape.ndim() - 1];
    } else {
      num_input = dshape.ProdShape(1, dshape.ndim());
    }
  } else {
    CHECK_EQ(dshape.ndim(), 2) << "trans_data only support 2-d input data.";
    num_input = dshape[0];
  }

  TShape wshape = Shape2(param.num_hidden, num_input);
  SHAPE_ASSIGN_CHECK(*in_shape, fullc::kWeight, wshape);
  if (!param.no_bias) {
    mxnet::TShape bshape = Shape1(param.num_hidden);
    SHAPE_ASSIGN_CHECK(*in_shape, fullc::kBias, bshape);
  }

  for (size_t i = num_inputs; i < 3 * num_inputs; ++i) {
    SHAPE_ASSIGN_CHECK(*in_shape, i, mxnet::TShape{1});
  }

  if (!param.trans_out) {
    if (!param.trans_data) {
      if (!param.flatten) {
        TShape result_shape(dshape);
        result_shape[dshape.ndim() - 1] = param.num_hidden;
        SHAPE_ASSIGN_CHECK(*out_shape, 0, result_shape);
      } else {
        SHAPE_ASSIGN_CHECK(*out_shape, 0, Shape2(dshape[0], param.num_hidden));
      }
    } else {
      SHAPE_ASSIGN_CHECK(*out_shape, 0, Shape2(dshape[1], param.num_hidden));
    }
  } else {
    CHECK_EQ(dshape.ndim(), 2) << "trans_out only support 2-d input data.";
    SHAPE_ASSIGN_CHECK(*out_shape, 0, Shape2(param.num_hidden, dshape[0]));
  }

  SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape(1, 1));
  SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape(1, 1));
  return true;
}

bool QuantizedFullyConnectedType(const nnvm::NodeAttrs& attrs,
                                 std::vector<int> *in_type,
                                 std::vector<int> *out_type) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t num_inputs = param.no_bias ? 2 : 3;
  CHECK_EQ(in_type->size(), num_inputs * 3);
  CHECK_EQ(out_type->size(), 3U);

#if MXNET_USE_MKLDNN == 1
  CHECK(in_type->at(0) == mshadow::kInt8 || in_type->at(0) == mshadow::kUint8)
      << "QuantizedFullyConnected only supports int8/uint8 input, while "
      << in_type->at(0) << " is given.";
#else
  TYPE_ASSIGN_CHECK(*in_type, 0, mshadow::kInt8);
#endif
  for (size_t i = 1; i < num_inputs; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kInt8);
  }
  for (size_t i = num_inputs; i < 3 * num_inputs; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kFloat32);
  }

  TYPE_ASSIGN_CHECK(*out_type, 0, mshadow::kInt32);
  TYPE_ASSIGN_CHECK(*out_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 2, mshadow::kFloat32);
  return true;
}

bool QuantizedFullyConnectedStorageType(const nnvm::NodeAttrs& attrs,
                                        const int dev_mask,
                                        DispatchMode* dispatch_mode,
                                        std::vector<int> *in_attrs,
                                        std::vector<int> *out_attrs) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t num_inputs = param.no_bias ? 2 : 3;
  CHECK_EQ(in_attrs->size(), num_inputs * 3);
  CHECK_EQ(out_attrs->size(), 3U);

#if MXNET_USE_MKLDNN == 1
  bool support_mkldnn = true;
  if (param.trans_data || param.trans_out)
    support_mkldnn = false;
  return MKLDNNStorageType(attrs, dev_mask, support_mkldnn,
                           dispatch_mode, in_attrs, out_attrs);
#else
  *dispatch_mode = DispatchMode::kFCompute;

  for (auto &v : *out_attrs) {
    v = kDefaultStorage;
    if (common::stype_string(v).compare("unknown") == 0) {
      return false;
    }
  }

  for (auto &v : *in_attrs) {
    v = kDefaultStorage;
    if (common::stype_string(v).compare("unknown") == 0) {
      return false;
    }
  }
  return true;
#endif
}

struct QuantizedSumInitKernelWithBias {
  //  init sum data with bias for matrix b (n)
  MSHADOW_XINLINE static void Map(int i, int32_t *out, const int8_t *bias,
                                  const float min_out, const float max_out,
                                  const float min_bias, const float max_bias) {
    typedef int32_t T1;
    typedef int8_t  T2;
    using mshadow::red::limits::MinValue;
    using mshadow::red::limits::MaxValue;
    float float_for_one_out_quant  =
        MaxAbs(min_out, max_out) / static_cast<double>(MaxValue<T1>());
    float float_for_one_bias_quant =
        MaxAbs(min_bias, max_bias) / static_cast<double>(MaxValue<T2>());
    if (float_for_one_out_quant != 0) {
      out[i] = bias[i] * float_for_one_bias_quant /
          float_for_one_out_quant;
    } else {
      LOG(INFO) << "float_for_one_out_quant is 0,"
                << " need to check the why MaxAbs(min_out, max_out) of out_data is 0!";
      out[i] = 0;
    }
  }
};

static inline size_t PadBytes(size_t num_bytes, size_t alignment=64) {
  return (num_bytes + (alignment - num_bytes % alignment) % alignment);
}

template <typename DType>
void QuantizedFullyConnectedOp::Forward(const OpContext &ctx,
                                        const std::vector<TBlob> &in_data,
                                        const std::vector<OpReqType> &req,
                                        const std::vector<TBlob> &out_data) {
#if MSHADOW_USE_MKL == 1
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<cpu> *s = ctx.get_stream<cpu>();
  size_t num_inputs = param_.no_bias ? 2 : 3;
  CHECK_EQ(in_data.size(),  num_inputs * 3);
  CHECK_EQ(out_data.size(), 3U);

  const mxnet::TShape &dshape = in_data[fullc::kData].shape_;
  const mxnet::TShape &wshape = in_data[fullc::kWeight].shape_;
  const mxnet::TShape &oshape = out_data[fullc::kOut].shape_;

  if (dshape.ndim() != 2)
    CHECK(param_.flatten)
        << "QuantizedFullyConnectedForwardCPU only supports flatten=true "
        << "when dshape.ndim() != 2 for now.";

  if (param_.trans_data || param_.trans_out)
    CHECK_EQ(dshape.ndim(), 2U) << "trans_data or trans_out only supports 2D data input so far.";

  bool data_is_int8 = true;
  if(in_data[fullc::kData].type_flag_ == mshadow::kUint8)
    data_is_int8 = false;

  Tensor<cpu, 2, DType> data = in_data[fullc::kData].get_with_shape<cpu, 2, DType>(
    Shape2(dshape[0], dshape.ProdShape(1, dshape.ndim())), s);
  Tensor<cpu, 2, int8_t> weight = in_data[fullc::kWeight].get<cpu, 2, int8_t>(s);
  Tensor<cpu, 2, int32_t> out = out_data[fullc::kOut].get_with_shape<cpu, 2, int32_t>(
    Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);

  DType* matrix_a = data.dptr_;
  int8_t* matrix_b = weight.dptr_;
  int32_t* output_temp = out.dptr_;
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  const float alpha = 1.0f;
  const float beta  = 1.0f;
  const CBLAS_OFFSET offsetc = CblasFixOffset;
  const MKL_INT8 oa = 0;
  const MKL_INT8 ob = 0;
  MKL_INT32 oc = 0;
  int m = dshape[0], n = wshape[0], k = dshape.ProdShape(1, dshape.ndim());

  //  cblas_gemm_s8u8s32 required first matrix must be uint8
  //  shift data from int8(from -128 to 127) to uint8 (from 0 to 255)
  int shift = 128;
  uint8_t* shifted_matrix_a_ptr = nullptr;
  int8_t* shifted_matrix_b_ptr = nullptr;
  int32_t* data_reduction_ptr = nullptr;
  int32_t* int32_bias_ptr = nullptr;
  int32_t* weight_reduction_ptr = nullptr;

  if (param_.trans_data) {
    m = dshape[1];
    k = dshape[0];
    n = wshape[0];
  }
  if (param_.trans_out) {
    m = wshape[0];
    k = dshape[1];
    n = dshape[0];
  }

  Tensor<cpu, 1, float> min_output = out_data[quantized_fullc::kOutMin].get<cpu, 1, float>(s);
  Tensor<cpu, 1, float> max_output = out_data[quantized_fullc::kOutMax].get<cpu, 1, float>(s);
  float min_data = (in_data[num_inputs + quantized_fullc::kDataMin].get<cpu, 1, float>(s)).dptr_[0];
  float max_data = (in_data[num_inputs + quantized_fullc::kDataMax].get<cpu, 1, float>(s)).dptr_[0];
  float min_weight =
    (in_data[num_inputs + quantized_fullc::kWeightMin].get<cpu, 1, float>(s)).dptr_[0];
  float max_weight =
    (in_data[num_inputs + quantized_fullc::kWeightMax].get<cpu, 1, float>(s)).dptr_[0];
  float min_bias = 0.0;
  float max_bias = 0.0;

  if (!param_.no_bias) {
    min_bias =
      (in_data[num_inputs + quantized_fullc::kBiasMin].get<cpu, 1, float>(s)).dptr_[0];
    max_bias =
      (in_data[num_inputs + quantized_fullc::kBiasMax].get<cpu, 1, float>(s)).dptr_[0];
  }

  /*
  Tensor<cpu, 1, float> max_data =
    in_data[num_inputs + quantized_fullc::kDataMax].get<cpu, 1, float>(s);
  Tensor<cpu, 1, float> min_weight =
    in_data[num_inputs + quantized_fullc::kWeightMin].get<cpu, 1, float>(s);
  Tensor<cpu, 1, float> max_weight =
    in_data[num_inputs + quantized_fullc::kWeightMax].get<cpu, 1, float>(s);
    Tensor<cpu, 1, float> min_bias =
      in_data[num_inputs + quantized_fullc::kBiasMin].get<cpu, 1, float>(s);
    Tensor<cpu, 1, float> max_bias =
      in_data[num_inputs + quantized_fullc::kBiasMax].get<cpu, 1, float>(s);
  */

  if (initialized_) {
    if (cached_min_data_ != min_data || cached_max_data_ != max_data ||
        cached_min_weight_ != min_weight || cached_max_weight_ != max_weight ||
        (!param_.no_bias && (cached_min_bias_ != min_bias || cached_max_bias_ != max_bias))) {
          initialized_ = false;
        }
  }

  if (!initialized_) {
    // TODO, DEBUG
    if (data_is_int8)
      std::cout << "s8, ";
    else
      std::cout << "u8, ";

    if (!param_.no_bias)
      std::cout << "with_bias, ";
    if (param_.trans_data)
      std::cout << "trans+fc, ";
    if (param_.trans_out)
      std::cout << "fc+trans, ";

    std::cout << "\n";

    cached_min_data_ = min_data;
    cached_max_data_ = max_data;
    cached_min_weight_ = min_weight;
    cached_max_weight_ = max_weight;
    if (!param_.no_bias) {
      cached_min_bias_ = min_bias;
      cached_max_bias_ = max_bias;
    }

    float data_scale = (data_is_int8 ? kInt8Range : kUint8Range) /
        MaxAbs(cached_min_data_, cached_max_data_);
    float weight_scale = kInt8Range / MaxAbs(cached_min_weight_, cached_max_weight_);
    size_t bias_size = 0;
    size_t temp_space_size = 0;
    Tensor<cpu, 1, int8_t> bias;

    if (!param_.no_bias) {
      bias = in_data[fullc::kBias].get_with_shape<cpu, 1, int8_t>(Shape1(wshape[0]), s);

      bias_size = wshape[0];
      // rescaled int32_bias
      temp_space_size += PadBytes(bias_size * sizeof(int32_t));
    }

    Kernel<QuantizationRangeForMultiplicationStruct, cpu>::Launch(
        s, 1, &cached_min_out_, &cached_max_out_, &cached_min_data_, &cached_max_data_,
        &cached_min_weight_, &cached_max_weight_);

    if (param_.trans_out) {
      // shifted_matrix_a
      temp_space_size += PadBytes(m * k);

      // data(matrix_b) reduction
      temp_space_size += PadBytes(n * sizeof(int32_t));

      if (!data_is_int8) {
        // shifed_matrix_b
        temp_space_size += PadBytes(n * k);

        // weight(matrix_a) reduction
        temp_space_size += PadBytes(m * sizeof(int32_t));
      }
    } else {
      if (data_is_int8) {
        // shifted_matrix_a
        temp_space_size += PadBytes(m * k);

        // weight(matrix_b) reduction
        temp_space_size += PadBytes(n * sizeof(int32_t));
      }
    }

    char* temp_space_curr_ptr = nullptr;
    if (temp_space_size > 0) {
      // allocate enough memory for later use
      temp_space_ = ctx.requested[quantized_fc::kTempSpace].get_space_typed<cpu, 1, char>(
          Shape1(temp_space_size), s);

      temp_space_curr_ptr = temp_space_.dptr_;
    }

    if (param_.trans_out) {
      // weight x data(T)
      // matrix_a is weight, need to convert it from int8_t to uint8_t.
      shifted_matrix_a_ptr = reinterpret_cast<uint8_t*>(temp_space_curr_ptr);
      temp_space_curr_ptr += PadBytes(m * k);

      #pragma omp parallel for num_threads(omp_threads)
      for (index_t i = 0; i < static_cast<index_t>(m * k); ++i) {
        shifted_matrix_a_ptr[i] = weight.dptr_[i] + shift;
      }

      data_reduction_ptr = reinterpret_cast<int32_t*>(temp_space_curr_ptr);
      temp_space_curr_ptr += PadBytes(n * sizeof(int32_t));

      if (!data_is_int8) {
        // matrix_b is data, when data is uint8_t, need to convert to int8_t.
        // Get the buffer for converted data, and do online shifting.
        shifted_matrix_b_ptr = reinterpret_cast<int8_t*>(temp_space_curr_ptr);
        temp_space_curr_ptr += PadBytes(n * k);

        weight_reduction_ptr = reinterpret_cast<int32_t*>(temp_space_curr_ptr);
        temp_space_curr_ptr += PadBytes(m * sizeof(int32_t));

        #pragma omp parallel for num_threads(omp_threads)
        for (index_t i = 0; i < static_cast<index_t>(m); ++i) {
          weight_reduction_ptr[i] = 0;  // TODO, memset?
          for (index_t j = 0; j < static_cast<index_t>(k); ++j) {
            weight_reduction_ptr[i] += weight.dptr_[j];
          }
          weight_reduction_ptr[i] *= shift;
        }
      }
    } else {
      // data x weight(T)
      if (data_is_int8) {
        // matrix_a is data, when data is int8_t, need to convert to uint8_t.
        // Get the buffer for converted data, and do online shifting.
        shifted_matrix_a_ptr = reinterpret_cast<uint8_t*>(temp_space_curr_ptr);
        temp_space_curr_ptr += PadBytes(m * k);

        weight_reduction_ptr = reinterpret_cast<int32_t*>(temp_space_curr_ptr);
        temp_space_curr_ptr += PadBytes(n * sizeof(int32_t));  // size is n here

        #pragma omp parallel for num_threads(omp_threads)
        for (index_t i = 0; i < static_cast<index_t>(n); ++i) { // size is n here
          weight_reduction_ptr[i] = 0;  // TODO, memset?
          for (index_t j = 0; j < static_cast<index_t>(k); ++j) {
            weight_reduction_ptr[i] += weight.dptr_[j];
          }
          weight_reduction_ptr[i] *= shift;
        }
      }
    }

    if (!param_.no_bias) {
      float bias_int32_rescale = data_scale * weight_scale *
          MaxAbs(cached_min_bias_, cached_max_bias_) / kInt8Range;

      CHECK(temp_space_curr_ptr);
      int32_bias_ptr = reinterpret_cast<int32_t*>(temp_space_curr_ptr);
      temp_space_curr_ptr += PadBytes(bias_size * sizeof(int32_t));

      #pragma omp parallel for num_threads(omp_threads)
      for (index_t i = 0; i < static_cast<index_t>(bias_size); ++i) {
        int32_bias_ptr[i] = bias.dptr_[i] * bias_int32_rescale;
      }

      if (weight_reduction_ptr) {
        // valid only: (params_.trans_out && !data_is_int8) || (!params_.trans_out && data_is_int8)
        #pragma omp parallel for num_threads(omp_threads)
        for (index_t i = 0; i < static_cast<index_t>(bias_size); ++i) {
          int32_bias_ptr[i] += weight_reduction_ptr[i];
        }
      }
    }

    initialized_ = true;
  }

  if (param_.no_bias) {
    #pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < m * n; ++i) {
      output_temp[i] = 0;
    }
  }

  if (param_.trans_out) {
    // weight x data(T)
    // out + 128 * weight_reduction - 128 * data_reduction
    #pragma omp parallel for num_threads(omp_threads)
    for (index_t i = 0; i < static_cast<index_t>(n); ++i) {
      data_reduction_ptr[i] = 0; // TODO: memset?
      for (index_t j = 0; j < static_cast<index_t>(k); ++j) {
        data_reduction_ptr[i] += data.dptr_[j];
      }
      data_reduction_ptr[i] *= shift;
    }

    if (!data_is_int8) {
      // matrix_b is data, when data is uint8_t, need to convert to int8_t.
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i < n * k; ++i) {
        shifted_matrix_b_ptr[i] = data.dptr_[i] - shift;
      }
      matrix_b = shifted_matrix_b_ptr;

      auto reduction_ptr = (!param_.no_bias) ? int32_bias_ptr : weight_reduction_ptr;
      #pragma omp parallel for num_threads(omp_threads)
      for (index_t i = 0; i < static_cast<index_t>(m * n); ++i) {
        output_temp[i] = reduction_ptr[i % n] - data_reduction_ptr[i % m];
      }
    } else {
      matrix_b = reinterpret_cast<int8_t *>(data.dptr_);

      #pragma omp parallel for num_threads(omp_threads)
      for (index_t i = 0; i < static_cast<index_t>(m * n); ++i) {
        output_temp[i] -= data_reduction_ptr[i % m];
      }
    }
  } else {
    if (data_is_int8) {
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i < m * k; ++i) {
        shifted_matrix_a_ptr[i] = matrix_a[i] + shift;
      }

      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
          output_temp[i] -= shift * matrix_b[i * k + j];
        }
      }
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = n; i < m * n; ++i) {
        output_temp[i] = output_temp[i % n];
      }
    }
  }

  min_output.dptr_[0] = cached_min_out_;
  max_output.dptr_[0] = cached_max_out_;

  auto trans_a = CblasNoTrans;
  auto trans_b = CblasTrans;
  MKL_INT lda = k;
  MKL_INT ldb = k;
  MKL_INT ldc = n;

  if (param_.trans_data) {
    trans_a = CblasTrans;
    lda = m;
  }

  auto matrix_a_ = (!param_.trans_out && !data_is_int8) ? reinterpret_cast<void *>(matrix_a) :
      reinterpret_cast<void *>(shifted_matrix_a_ptr);

  cblas_gemm_s8u8s32(CblasRowMajor,
                     trans_a,
                     trans_b,
                     offsetc,
                     m,
                     n,
                     k,
                     alpha,
                     matrix_a_, // alwyas uint8_t*;
                     lda,
                     oa,
                     matrix_b,  // always int8_t*
                     ldb,
                     ob,
                     beta,
                     out.dptr_,
                     ldc,
                     &oc);
#else
  LOG(FATAL) << "Quantized fully connected operator relies on cblas_gemm_s8u8s32"
             << " which is only supported by MKL BLAS."
             << " Please build MXNet with USE_BLAS=mkl to leverage this operator.";
#endif
}

static OpStatePtr CreateQuantizedFullyConnectedState(const nnvm::NodeAttrs &attrs,
                                                     Context ctx,
                                                     const mxnet::ShapeVector &in_shapes,
                                                     const std::vector<int> &in_types) {
  return OpStatePtr::Create<QuantizedFullyConnectedOp>(attrs);
}

static void QuantizedFullyConnectedForwardCPU(const OpStatePtr &state,
                                              const OpContext &ctx,
                                              const std::vector<TBlob> &in_data,
                                              const std::vector<OpReqType> &req,
                                              const std::vector<TBlob> &out_data) {
  QuantizedFullyConnectedOp &op = state.get_state<QuantizedFullyConnectedOp>();

  MXNET_INT8_TYPE_SWITCH(in_data[fullc::kData].type_flag_, DType, {
    op.Forward<DType>(ctx, in_data, req, out_data);
  });
}

#if MXNET_USE_MKLDNN == 1
void QuantizedFullyConnectedForwardExCPU(const nnvm::NodeAttrs &attrs,
                                         const OpContext &ctx,
                                         const std::vector<NDArray> &in_data,
                                         const std::vector<OpReqType> &req,
                                         const std::vector<NDArray> &out_data) {
  MKLDNNQuantizedFullyConnectedForward(attrs, ctx, in_data, req, out_data);
}
#endif

NNVM_REGISTER_OP(_contrib_quantized_fully_connected)
.describe(R"code(Fully Connected operator for input, weight and bias data type of int8,
and accumulates in type int32 for the output. For each argument, two more arguments of type
float32 must be provided representing the thresholds of quantizing argument from data
type float32 to int8. The final outputs contain the convolution result in int32, and min
and max thresholds representing the threholds for quantizing the float32 output into int32.

.. Note::
    This operator only supports forward propogation. DO NOT use it in training.)code" ADD_FILELINE)
.set_num_inputs(
  [](const NodeAttrs& attrs) {
    const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
    return param.no_bias? 6 : 9;
  })
.set_num_outputs(3)
.set_attr_parser(ParamParser<FullyConnectedParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
    if (param.no_bias) {
      return std::vector<std::string>{"data", "weight", "min_data", "max_data",
                                      "min_weight", "max_weight"};
    } else {
      return std::vector<std::string>{"data", "weight", "bias", "min_data", "max_data",
                                      "min_weight", "max_weight", "min_bias", "max_bias"};
    }
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "min_output", "max_output"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", QuantizedFullyConnectedShape)
.set_attr<nnvm::FInferType>("FInferType", QuantizedFullyConnectedType)
.set_attr<FInferStorageType>("FInferStorageType", QuantizedFullyConnectedStorageType)
.set_attr<FCreateOpState>("FCreateOpState", CreateQuantizedFullyConnectedState)
.set_attr<FStatefulCompute>("FStatefulComputeEx<cpu>", QuantizedFullyConnectedForwardCPU)
// TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
// will be reverted after the improvement of CachedOP is done.
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
//.set_attr<FCompute>("FCompute<cpu>", QuantizedFullyConnectedForwardCPU)
#if MXNET_USE_MKLDNN == 1
//.set_attr<bool>("TIsMKLDNN", true)
//.set_attr<FComputeEx>("FComputeEx<cpu>", QuantizedFullyConnectedForwardExCPU)
#endif
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("weight", "NDArray-or-Symbol", "weight.")
.add_argument("bias", "NDArray-or-Symbol", "bias.")
.add_argument("min_data", "NDArray-or-Symbol", "Minimum value of data.")
.add_argument("max_data", "NDArray-or-Symbol", "Maximum value of data.")
.add_argument("min_weight", "NDArray-or-Symbol", "Minimum value of weight.")
.add_argument("max_weight", "NDArray-or-Symbol", "Maximum value of weight.")
.add_argument("min_bias", "NDArray-or-Symbol", "Minimum value of bias.")
.add_argument("max_bias", "NDArray-or-Symbol", "Maximum value of bias.")
.add_arguments(FullyConnectedParam::__FIELDS__());

NNVM_REGISTER_OP(FullyConnected)
.set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
    nnvm::NodePtr node = nnvm::Node::Create();
    node->attrs.op = Op::Get("_contrib_quantized_fully_connected");
    node->attrs.name = "quantized_" + attrs.name;
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
  });
}  // namespace op
}  // namespace mxnet
