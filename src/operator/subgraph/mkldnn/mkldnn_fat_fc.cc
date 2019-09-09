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
 * \file mkldnn_fat_fc.cc
 * \brief MKLDNN (Quantized) fat FullyConnected operator based on subgraph
*/

#if MXNET_USE_MKLDNN == 1

#include <utility>
#include <vector>
#include <string>
#include "../common.h"
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../../nn/mkldnn/mkldnn_ops-inl.h"
#include "../../nn/mkldnn/mkldnn_fully_connected-inl.h"
#include "../../quantization/quantization_utils.h"

namespace mxnet {
namespace op {

struct MKLDNNFCFatFullParam {
  std::vector<FullyConnectedParam> default_params;
  MKLDNNFCParam mkldnn_param;
  std::vector<float> output_scales = {0.0};
  std::vector<float> requantize_scales = {0.0};
};

class SgMKLDNNFatFCOp {
 public:
  explicit SgMKLDNNFatFCOp(const nnvm::NodeAttrs &attrs)
      : initialized_(false),
        subgraph_sym_(*attrs.subgraphs[0]),
        full_param_(nnvm::get<MKLDNNFCFatFullParam>(attrs.parsed)) {}

  void Forward(const OpContext &ctx, const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req, const std::vector<NDArray> &outputs);

  void Backward(const OpContext &ctx, const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req, const std::vector<NDArray> &outputs) {
    LOG(FATAL) << "Not implemented: subgraph mkldnn fully connected only supports "
                  "inference computation.";
  }

 private:
  bool initialized_;
  nnvm::Symbol subgraph_sym_;
  MKLDNNFCFatFullParam full_param_;
  std::shared_ptr<mkldnn::inner_product_forward> fwd_;
  std::shared_ptr<mkldnn::memory> cached_data_;
  std::shared_ptr<mkldnn::memory> cached_weight_;
  std::shared_ptr<mkldnn::memory> cached_bias_;
  std::shared_ptr<mkldnn::memory> cached_output_;
  float cached_min_output_;
  float cached_max_output_;
};

void SgMKLDNNFatFCOp::Forward(const OpContext &ctx, const std::vector<NDArray> &in_data,
                              const std::vector<OpReqType> &req,
                              const std::vector<NDArray> &out_data) {
  auto &mkldnn_param = full_param_.mkldnn_param;
  auto &default_param = full_param_.default_params[0];
  const auto num_fc = full_param_.default_params.size();
  bool has_bias = !default_param.no_bias;
  size_t base_num_inputs = has_bias ? 3 : 2;
  CHECK_EQ(default_param.flatten, false) << "Doesn't support flatten = true for now.";

  if (!initialized_) {
    initialized_ = true;
    auto data = in_data[fullc::kData];
    const auto ishape = data.shape();
    index_t num_batch = ishape[0];
    if (ishape.ndim() != 2) {
      if (!default_param.flatten) {
        num_batch = ishape.ProdShape(0, ishape.ndim() - 1);
        data = NDArray(Shape2(ishape.ProdShape(0, ishape.ndim() - 1), ishape[ishape.ndim() - 1]),
                       data.ctx(), true, data.dtype());
      } else {
        data = NDArray(Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), data.ctx(), true,
                       data.dtype());
      }
    }

    mkldnn::memory::dims out_dims(2);
    out_dims[0] = num_batch;
    out_dims[1] = 0;
    std::vector<size_t> weight_channels(num_fc, 0);
    for (size_t i = 0; i < num_fc; i++) {
      const auto &weight = in_data[i * base_num_inputs + fullc::kWeight];
      weight_channels[i] = weight.shape()[0];
      out_dims[1] += weight_channels[i];
    }

    float min_data = std::numeric_limits<float>::max();
    float max_data = std::numeric_limits<float>::min();
    std::vector<NDArray> rescaled_bias;
    if (mkldnn_param.quantized) {
      std::vector<float> weight_scales(num_fc, 0.f);
      std::vector<float> bias_scales(num_fc, 0.f);
      min_data =
          in_data[base_num_inputs * num_fc + quantized_fullc::kDataMin].data().dptr<float>()[0];
      max_data =
          in_data[base_num_inputs * num_fc + quantized_fullc::kDataMax].data().dptr<float>()[0];
      int base_min_max_inputs = base_num_inputs * 2;
      CHECK_EQ(in_data.size(), base_num_inputs * num_fc + base_min_max_inputs * num_fc);
      for (size_t i = 0; i < num_fc; i++) {
        const auto this_weight_min = in_data[base_num_inputs * num_fc + base_min_max_inputs * i +
                                             quantized_fullc::kWeightMin]
                                         .data()
                                         .dptr<float>()[0];
        const auto this_weight_max = in_data[base_num_inputs * num_fc + base_min_max_inputs * i +
                                             quantized_fullc::kWeightMax]
                                         .data()
                                         .dptr<float>()[0];
        weight_scales[i] = kInt8Range / MaxAbs(this_weight_min, this_weight_max);
        if (has_bias) {
          const auto this_bias_min = in_data[base_num_inputs * num_fc + base_min_max_inputs * i +
                                             quantized_fullc::kBiasMin]
                                         .data()
                                         .dptr<float>()[0];
          const auto this_bias_max = in_data[base_num_inputs * num_fc + base_min_max_inputs * i +
                                             quantized_fullc::kBiasMax]
                                         .data()
                                         .dptr<float>()[0];
          bias_scales[i] = kInt8Range / MaxAbs(this_bias_min, this_bias_max);
        }
      }
      CHECK(data.dtype() == mshadow::kInt8 || data.dtype() == mshadow::kUint8);
      auto data_range = (data.dtype() == mshadow::kInt8) ? kInt8Range : kUint8Range;
      float data_scale = data_range / MaxAbs(min_data, max_data);
      float quantized_out_range = kInt8Range;
      if (mkldnn_param.enable_float_output) {
        full_param_.output_scales.clear();
        for (size_t i = 0; i < num_fc; i++) {
          for (size_t j = 0; j < weight_channels[i]; j++) {
            full_param_.output_scales.push_back(1.0 / data_scale / weight_scales[i]);
          }
        }
        full_param_.requantize_scales.resize(0);
      } else if (mkldnn_param.min_calib_range.has_value() &&
                 mkldnn_param.max_calib_range.has_value()) {
        full_param_.output_scales.resize(0);
        cached_min_output_ = mkldnn_param.min_calib_range.value();
        cached_max_output_ = mkldnn_param.max_calib_range.value();
        full_param_.requantize_scales.clear();
        for (size_t i = 0; i < num_fc; i++) {
          for (size_t j = 0; j < weight_channels[i]; j++) {
            full_param_.requantize_scales.push_back(quantized_out_range /
                                                    MaxAbs(cached_min_output_, cached_max_output_) /
                                                    data_scale / weight_scales[i]);
          }
        }
      } else {
        LOG(FATAL) << "per channel scale doesn't support non calibration mode.";
        // Stream<cpu> *s = ctx.get_stream<cpu>();
        // mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
        //     s, 1, &cached_min_output_, &cached_max_output_, &min_data, &max_data, &min_weight,
        //     &max_weight);
      }

      if (has_bias) {
        // Rescale all bias
        rescaled_bias.resize(num_fc);
        for (size_t i = 0; i < num_fc; i++) {
          const float bias_rescale = data_scale * weight_scales[i] / bias_scales[i];
          const auto &bias = in_data[i * base_num_inputs + fullc::kBias];
          rescaled_bias[i] =
              NDArray(bias.storage_type(), bias.shape(), bias.ctx(), true, mshadow::kInt32);
          int8_t *bias_ptr = bias.data().dptr<int8_t>();
          int32_t *quantized_bias_ptr = rescaled_bias[i].data().dptr<int32_t>();
          size_t bias_size = bias.shape().Size();
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
          for (index_t j = 0; j < static_cast<index_t>(bias_size); ++j) {
            quantized_bias_ptr[j] = std::round(bias_ptr[j] * bias_rescale);
          }
        }
      }
    }
    // Concat weight
    std::shared_ptr<mkldnn::memory> concatted_weight;
    {
      std::vector<mkldnn::memory::primitive_desc> weight_md;
      std::vector<mkldnn::primitive::at> weight_mem;
      weight_md.reserve(num_fc);
      weight_mem.reserve(num_fc);
      for (size_t i = 0; i < num_fc; i++) {
        const auto &weight = in_data[i * base_num_inputs + fullc::kWeight];
        const mkldnn::memory *mem = weight.GetMKLDNNData();
        const mkldnn::memory::primitive_desc weight_pd = mem->get_primitive_desc();
        weight_md.push_back(weight_pd);
        weight_mem.push_back(*mem);
      }
      const mkldnn::concat::primitive_desc concat_pd(0, weight_md);
      concatted_weight = std::make_shared<mkldnn::memory>(concat_pd.dst_primitive_desc());
      MKLDNNStream::Get()->RegisterPrim(mkldnn::concat(concat_pd, weight_mem, *concatted_weight));
      MKLDNNStream::Get()->Submit();
    }

    // Concat bias
    std::shared_ptr<mkldnn::memory> concatted_bias;
    if (has_bias) {
      std::vector<mkldnn::memory::primitive_desc> bias_md;
      std::vector<mkldnn::primitive::at> bias_mem;
      bias_md.reserve(num_fc);
      bias_mem.reserve(num_fc);
      for (size_t i = 0; i < num_fc; i++) {
        const auto &bias =
            mkldnn_param.quantized ? rescaled_bias[i] : in_data[i * base_num_inputs + fullc::kBias];
        const mkldnn::memory *mem = bias.GetMKLDNNData();
        mkldnn::memory::primitive_desc bias_pd = mem->get_primitive_desc();
        bias_md.push_back(bias_pd);
        bias_mem.push_back(*mem);
      }
      const mkldnn::concat::primitive_desc concat_pd(0, bias_md);
      concatted_bias = std::make_shared<mkldnn::memory>(concat_pd.dst_primitive_desc());
      MKLDNNStream::Get()->RegisterPrim(mkldnn::concat(concat_pd, bias_mem, *concatted_bias));
      MKLDNNStream::Get()->Submit();
    }
    MKLDNNFCFullParam param;
    param.default_param = default_param;
    param.mkldnn_param = mkldnn_param;
    param.output_scales = full_param_.output_scales;
    param.requantize_scales = full_param_.requantize_scales;
    NDArray weight = NDArray(concatted_weight);
    NDArray bias = has_bias ? NDArray(concatted_bias) : NDArray();
    const auto &output = out_data[fullc::kData];
    mkldnn::memory::desc out_md(out_dims, get_mkldnn_type(output.dtype()),
                                mkldnn::memory::format::any);
    mkldnn::inner_product_forward::primitive_desc fc_pd =
        GetFCFwdImpl(param, false, data, weight, has_bias ? &bias : nullptr, out_md);
    cached_data_ = std::make_shared<mkldnn::memory>(fc_pd.src_primitive_desc(), nullptr);
    cached_output_ = std::make_shared<mkldnn::memory>(fc_pd.dst_primitive_desc(), nullptr);
    // convert weight and bias to the format that MKL-DNN requires
    cached_weight_ = std::make_shared<mkldnn::memory>(fc_pd.weights_primitive_desc());
    MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(*concatted_weight, *cached_weight_));
    if (has_bias) {
      cached_bias_ = std::make_shared<mkldnn::memory>(fc_pd.bias_primitive_desc());
      MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(*concatted_bias, *cached_bias_));
    }
    MKLDNNStream::Get()->Submit();
    if (has_bias) {
      fwd_ = std::make_shared<mkldnn::inner_product_forward>(
          fc_pd, mkldnn::primitive::at(*cached_data_), mkldnn::primitive::at(*cached_weight_),
          mkldnn::primitive::at(*cached_bias_), *cached_output_);
    } else {
      fwd_ = std::make_shared<mkldnn::inner_product_forward>(
          fc_pd, mkldnn::primitive::at(*cached_data_), mkldnn::primitive::at(*cached_weight_),
          *cached_output_);
    }
  }
  cached_data_->set_data_handle(in_data[fullc::kData].GetMKLDNNData()->get_data_handle());
  cached_output_->set_data_handle(out_data[fullc::kOut].GetMKLDNNData()->get_data_handle());
  MKLDNNStream::Get()->RegisterPrim(*fwd_);
  MKLDNNStream::Get()->Submit();
  if (mkldnn_param.quantized && !mkldnn_param.enable_float_output) {
    float *min_output_ptr = out_data[quantized_fullc::kOutMin].data().dptr<float>();
    float *max_output_ptr = out_data[quantized_fullc::kOutMax].data().dptr<float>();
    *min_output_ptr = cached_min_output_;
    *max_output_ptr = cached_max_output_;
  }
}

static void SgMKLDNNFatFCParamParser(nnvm::NodeAttrs *attrs) {
  MKLDNNFCFatFullParam full_param;
  try {
    full_param.mkldnn_param.Init(attrs->dict);
  } catch (const dmlc::ParamError &e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto &k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  auto subgraph_sym = attrs->subgraphs[0];
  DFSVisit(subgraph_sym->outputs, [&](const nnvm::NodePtr &node) {
    if (node->is_variable()) return;
    auto &node_name = node->op()->name;
    if (node_name == "FullyConnected") {
      full_param.default_params.push_back(nnvm::get<FullyConnectedParam>(node->attrs.parsed));
    }
  });
  attrs->parsed = std::move(full_param);
}

static bool SgMKLDNNFatFCInferShape(const nnvm::NodeAttrs &attrs, mxnet::ShapeVector *in_shapes,
                                    mxnet::ShapeVector *out_shapes) {
  auto const &full_param = nnvm::get<MKLDNNFCFatFullParam>(attrs.parsed);
  const auto num_fc = full_param.default_params.size();
  if (full_param.mkldnn_param.quantized) {
    const auto base_num_inputs = full_param.default_params[0].no_bias ? 2 : 3;
    mxnet::ShapeVector base_in_shapes;
    mxnet::ShapeVector base_out_shapes;
    for (size_t i = 0; i < num_fc * base_num_inputs; i++) {
      base_in_shapes.push_back(in_shapes->at(i));
    }
    for (size_t i = 0; i < num_fc; i++) {
      base_out_shapes.push_back(out_shapes->at(0));
    }
    bool ret = DefaultSubgraphOpShape(attrs, &base_in_shapes, &base_out_shapes);
    for (size_t i = 0; i < in_shapes->size(); ++i) {
      if (i < base_in_shapes.size())
        in_shapes->at(i) = base_in_shapes[i];
      else
        SHAPE_ASSIGN_CHECK(*in_shapes, i, Shape1(1));
    }
    size_t base_ndim = base_out_shapes[0].ndim();
    mxnet::TShape new_out_shape(base_ndim + 1, -1);
    size_t j = 0;
    for (size_t i = 0; i < base_ndim + 1; i++) {
      if (i != base_ndim - 1) {
        new_out_shape[i] = base_out_shapes[0][j++];
      } else {
        new_out_shape[i] = num_fc;
      }
    }
    out_shapes->at(0) = new_out_shape;
    for (size_t i = 1; i < out_shapes->size(); ++i) {
      SHAPE_ASSIGN_CHECK(*out_shapes, i, Shape1(1));
    }
    return ret;
  } else {
    mxnet::ShapeVector base_out_shapes;
    for (size_t i = 0; i < num_fc; i++) {
      base_out_shapes.push_back(out_shapes->at(0));

    }
    bool ret = DefaultSubgraphOpShape(attrs, in_shapes, &base_out_shapes);
    size_t base_ndim = base_out_shapes[0].ndim();
    mxnet::TShape new_out_shape(base_ndim + 1, -1);
    size_t j = 0;
    for (size_t i = 0; i < base_ndim + 1; i++) {
      if (i != base_ndim - 1) {
        new_out_shape[i] = base_out_shapes[0][j++];
      } else {
        new_out_shape[i] = num_fc;
      }
    }
    out_shapes->at(0) = new_out_shape;
    return ret;
  }
}

static bool SgMKLDNNFatFCInferType(const nnvm::NodeAttrs &attrs, std::vector<int> *in_types,
                                   std::vector<int> *out_types) {
  auto const &full_param = nnvm::get<MKLDNNFCFatFullParam>(attrs.parsed);
  const auto num_fc = full_param.default_params.size();
  if (full_param.mkldnn_param.quantized) {
    const auto base_num_inputs = full_param.default_params[0].no_bias ? 2 : 3;

    CHECK(in_types->at(0) == mshadow::kInt8 || in_types->at(0) == mshadow::kUint8)
        << "Quantized Fat FullyConnected only supports int8/uint8 input, while " << in_types->at(0)
        << " is given.";
    for (size_t i = 1; i < in_types->size(); ++i) {
      if (i < base_num_inputs * num_fc) {
        TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kInt8);
      } else {
        TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
      }
    }

    if (full_param.mkldnn_param.enable_float_output) {
      for (size_t i = 0; i < out_types->size(); ++i) {
        TYPE_ASSIGN_CHECK(*out_types, i, mshadow::kFloat32);
      }
    } else {
      if (full_param.mkldnn_param.min_calib_range.has_value() &&
          full_param.mkldnn_param.max_calib_range.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);
        for (size_t i = 1; i < out_types->size(); ++i) {
          TYPE_ASSIGN_CHECK(*out_types, i, mshadow::kFloat32);
        }
      }
    }
    return true;
  } else {
    std::vector<int> base_out_types;
    for (size_t i = 0; i < num_fc; i++) {
      base_out_types.push_back(out_types->at(0));
    }
    bool ret = DefaultSubgraphOpType(attrs, in_types, &base_out_types);
    out_types->at(0) = base_out_types[0];
    return ret;
  }
}

static bool SgMKLDNNFatFCStorageType(const nnvm::NodeAttrs &attrs, const int dev_mask,
                                     DispatchMode *dispatch_mode, std::vector<int> *in_attrs,
                                     std::vector<int> *out_attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFatFullParam>(attrs.parsed);
    const auto num_fc = full_param.default_params.size();
  if (full_param.mkldnn_param.quantized) {
    const auto base_num_inputs = full_param.default_params[0].no_bias ? 2 : 3;
    std::vector<int> base_in_attrs;
    std::vector<int> base_out_attrs;
    for (size_t i = 0; i < num_fc * base_num_inputs; i++) {
      base_in_attrs.push_back(in_attrs->at(i));
    }
    for (size_t i = 0; i < num_fc; i++) {
      base_out_attrs.push_back(out_attrs->at(0));
    }
    bool ret = DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode, &base_in_attrs,
                                            &base_out_attrs);

    for (size_t i = 0; i < in_attrs->size(); ++i) {
      if (i < base_in_attrs.size())
        in_attrs->at(i) = base_in_attrs[i];
      else
        type_assign(&in_attrs->at(i), mxnet::kDefaultStorage);
    }
    out_attrs->at(0) = base_out_attrs[0];
    for (size_t i = 1; i < out_attrs->size(); ++i) {
      type_assign(&out_attrs->at(i), mxnet::kDefaultStorage);
    }
    return ret;
  } else {
    std::vector<int> base_out_attrs;
    for (size_t i = 0; i < num_fc; i++) {
      base_out_attrs.push_back(out_attrs->at(0));
    }
    bool ret =
        DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode, in_attrs, &base_out_attrs);
    out_attrs->at(0) = base_out_attrs[0];
    return ret;
  }
}

static OpStatePtr CreateSgMKLDNNFatFCState(const nnvm::NodeAttrs &attrs, Context ctx,
                                           const mxnet::ShapeVector &in_shapes,
                                           const std::vector<int> &in_types) {
  return OpStatePtr::Create<SgMKLDNNFatFCOp>(attrs);
}

static void SgMKLDNNFatFCForward(const OpStatePtr &state_pointer, const OpContext &ctx,
                                 const std::vector<NDArray> &inputs,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<NDArray> &outputs) {
  SgMKLDNNFatFCOp &op = state_pointer.get_state<SgMKLDNNFatFCOp>();
  op.Forward(ctx, inputs, req, outputs);
}

nnvm::NodePtr SgMKLDNNFatFCQuantizedOp(const NodeAttrs &attrs) {
  nnvm::NodePtr node = nnvm::Node::Create();
  node->attrs.op = Op::Get("_sg_mkldnn_fat_fully_connected");
  node->attrs.name = "quantized_" + attrs.name;
  node->attrs.dict = attrs.dict;
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

int SgMKLDNNFatFCNumOutputs(const NodeAttrs& attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFatFullParam>(attrs.parsed);
  return (full_param.mkldnn_param.quantized && !full_param.mkldnn_param.enable_float_output) ? 3
                                                                                             : 1;
}

static std::vector<std::string> SgMKLDNNFatFCListOutputNames(const NodeAttrs &attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFatFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    if (full_param.mkldnn_param.enable_float_output)
      return std::vector<std::string>{"output"};
    else
      return std::vector<std::string>{"output", "min_output", "max_output"};
  } else {
    return std::vector<std::string>{"output"};
  }
}

NNVM_REGISTER_OP(_sg_mkldnn_fat_fully_connected)
.describe(R"code(_sg_mkldnn_fat_fully_connected)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFatFullParam>(attrs.parsed);
  auto num_fc = full_param.default_params.size();
  auto num_inputs = (full_param.default_params[0].no_bias ? 2 : 3) * num_fc;
  if (full_param.mkldnn_param.quantized)
    return num_inputs * 3;
  else
    return num_inputs;
})
.set_num_outputs(SgMKLDNNFatFCNumOutputs)
.set_attr_parser(SgMKLDNNFatFCParamParser)
.set_attr<nnvm::FListOutputNames>("FListOutputNames", SgMKLDNNFatFCListOutputNames)
.set_attr<mxnet::FInferShape>("FInferShape", SgMKLDNNFatFCInferShape)
.set_attr<nnvm::FInferType>("FInferType", SgMKLDNNFatFCInferType)
.set_attr<FInferStorageType>("FInferStorageType", SgMKLDNNFatFCStorageType)
.set_attr<FCreateOpState>("FCreateOpState", CreateSgMKLDNNFatFCState)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgMKLDNNFatFCForward)
.set_attr<bool>("TIsMKLDNN", true)
// TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
// will be reverted after the improvement of CachedOP is done.
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNFatFCQuantizedOp)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; });

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
