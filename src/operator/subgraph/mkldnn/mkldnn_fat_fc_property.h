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
 * \file mkldnn_fat_fc_property.h
 * \brief Partition gragph property for fat FullyConnected operator
*/

#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FAT_FC_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FAT_FC_PROPERTY_H_
#if MXNET_USE_MKLDNN == 1

#include <string>
#include <vector>
#include "../../nn/mkldnn/mkldnn_fully_connected-inl.h"
#include "../common.h"
#include "../subgraph_property.h"

namespace nnvm {
NodePtr CreateVariableNode(const std::string& name);
}

namespace mxnet {
namespace op {

class SgMKLDNNFatFCSelector : public SubgraphSelectorV2 {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    kFail = 0,
    kStart,
    kSuccess,
  };

 private:
  bool disable_ = false;
  const BiDirectedNode *data_ = nullptr;
  std::vector<const BiDirectedNode *> matched_fc_;
  std::vector<const BiDirectedNode *> fc_consumer_;

 public:
  explicit SgMKLDNNFatFCSelector(bool disable) : disable_(disable) {}

  bool Select(const BiDirectedNode &sn) override {
    if (disable_) return false;
    const auto &n = *sn.node;
    if (n.op() == Op::Get("FullyConnected")) {
      matched_fc_.clear();
      matched_fc_.push_back(&sn);
      return true;
    }
    return false;
  }

  bool SelectInput(const BiDirectedNode &sn, const BiDirectedNode &snew_node) override {
    if (((&sn) == matched_fc_[0]) && !data_) {
      data_ = &snew_node;
      return true;
    }
    return false;
  }

  bool SelectOutput(const BiDirectedNode &sn, const BiDirectedNode &snew_node) override {
    if (((&sn) == data_) && snew_node.node->op() == Op::Get("FullyConnected")) {
      FullyConnectedParam param = nnvm::get<FullyConnectedParam>(matched_fc_[0]->node->attrs.parsed);
      FullyConnectedParam new_param = nnvm::get<FullyConnectedParam>(snew_node.node->attrs.parsed);
      if (param.flatten == new_param.flatten && param.no_bias == new_param.no_bias) {
        matched_fc_.push_back(&snew_node);
        return true;
      }
    } else if (std::count(matched_fc_.begin(), matched_fc_.end(), &sn)) {
      fc_consumer_.push_back(&snew_node);
      return false;
    }
    return false;
  }

  virtual std::vector<BiDirectedNode *> Filter(const std::vector<BiDirectedNode *> &candidates) {
    if (matched_fc_.size() > 1) {
      std::vector<BiDirectedNode *> ret;
      for (auto i : matched_fc_) {
        auto non_const_i = const_cast<BiDirectedNode *>(i);
        if (std::find(candidates.begin(), candidates.end(), non_const_i) !=
            candidates.end()) {
          ret.push_back(non_const_i);
        }
      }
      return ret;
    }
    return std::vector<BiDirectedNode *>(0);
  }

  void Reset() override {
    CHECK_GE(matched_fc_.size(), 1);
    auto new_selector = SgMKLDNNFatFCSelector(disable_);
    new_selector.Select(*matched_fc_[0]);
    *this = new_selector;
  }

  const std::vector<const BiDirectedNode *>& GetMatchedNode() { return matched_fc_; }
  const std::vector<const BiDirectedNode *>& GetOutputs() { return fc_consumer_; }
};

class SgMKLDNNFatFCProperty : public SubgraphProperty {
 public:
  SgMKLDNNFatFCProperty() : SubgraphProperty(kAdjust) {}

  static SubgraphPropertyPtr Create() {
    static const std::string &name = "MKLDNN Fat FullyConnected optimization pass";
    auto property = std::make_shared<SgMKLDNNFatFCProperty>();
    if (dmlc::GetEnv("MXNET_DISABLE_MKLDNN_FAT_FC_OPT", 0)) {
      LOG(INFO) << name << " is disabled.";
      property->SetAttr<bool>("disable", true);
    }
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    return property;
  }

  void AdjustSubgraphNode(const std::vector<nnvm::Node *> &subgraph_nodes,
                          const SubgraphSelectorV2Ptr &subgraph_selector,
                          const int subgraph_id = 0) const override {
    // Create fat fc
    nnvm::NodePtr n = nnvm::Node::Create();
    const auto &selector_ptr = static_cast<SgMKLDNNFatFCSelector *>(subgraph_selector.get());
    const auto &matched_fc = selector_ptr->GetMatchedNode();
    std::ostringstream node_name;
    node_name << "sg_mkldnn_fat_fc_" << std::to_string(subgraph_id);
    n->attrs.name = node_name.str();
    n->attrs.op = Op::Get("_sg_mkldnn_fat_fully_connected");
    int var_id = 0;
    for (const auto &fc : matched_fc) {
      for (auto &in : fc->node->inputs) {
        n->inputs.push_back(in);
        std::ostringstream var_name;
        var_name << "subgraph_" << std::to_string(subgraph_id) << "_var_"
                 << std::to_string(var_id++);
        nnvm::NodePtr var = nnvm::CreateVariableNode(var_name.str());
        in = nnvm::NodeEntry{var, 0, 0};
      }
    }
    // Create subgraph inner symbol
    nnvm::Symbol new_sym;
    for (const auto &sub_node : matched_fc) {
      new_sym.outputs.emplace_back(std::make_shared<nnvm::Node>(*sub_node->node), 0, 0);
    }
    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
    n->op()->attr_parser(&(n->attrs));
    // Crate split op after fat fc
    nnvm::NodePtr split = nnvm::Node::Create();
    std::ostringstream split_name;
    split_name << "sg_mkldnn_fat_fc_split_" << std::to_string(subgraph_id);
    split->attrs.name = split_name.str();
    split->attrs.op = Op::Get("split");
    split->attrs.dict["num_outputs"] = std::to_string(matched_fc.size());
    split->attrs.dict["axis"] = "-2";
    split->attrs.dict["squeeze_axis"] = "1";
    split->inputs.emplace_back(n, 0, 0);
    const auto &outputs = selector_ptr->GetOutputs();
    for (auto out : outputs) {
      for (size_t i = 0; i < out->node->inputs.size(); i++) {
        for (size_t j = 0; j < matched_fc.size(); j++) {
          if (matched_fc[j]->node == out->node->inputs[i].node.get()) {
            out->node->inputs[i] = {split, static_cast<int>(j), 0};
            break;
          }
        }
      }
    }
    split->op()->attr_parser(&(split->attrs));
  }

  SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const override {
    bool disable = false;
    if (HasAttr("disable")) {
      disable = GetAttr<bool>("disable");
    }
    auto selector = std::make_shared<SgMKLDNNFatFCSelector>(disable);
    return selector;
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_PROPERTY_H_
