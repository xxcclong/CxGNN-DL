#ifndef DATASET_H
#define DATASET_H

#include <assert.h>

#include <memory>
#include <string>

#include "Yaml.hpp"
#include "common.h"
#include "file_op.h"
#include "graph.h"
#include "spdlog/spdlog.h"
#include "split.h"

using std::make_shared;
using std::shared_ptr;
using std::string;
using torch::Tensor;

class Dataset {
 public:
  Index feature_len_ = 0;
  shared_ptr<Graph> graph_;
  shared_ptr<FileHandler> feature_handler_;  // for node feature

  torch::TensorOptions feature_option;

  static shared_ptr<Dataset> create(Yaml::Node &config);
  virtual void fetchData(const std::vector<Index> &index, int64_t feature_len,
                         Tensor &buf);
  virtual void fetchAllData(Tensor &buf);
  virtual Tensor index2D(const Tensor &tensor, const std::vector<Index> &index);
};

template <class T>
class DatasetTemplate : public Dataset {
 public:
  DatasetTemplate(Yaml::Node &config) {
    SPDLOG_WARN("Memory before full graph structure");
    showCpuMemCurrProc();
    graph_ = make_shared<Graph>(config);
    SPDLOG_WARN("Memory after full graph structure");
    showCpuMemCurrProc();
    // basic information
    string dataset_path = config["dataset"]["path"].As<string>();
    string split_type = config["dataset"]["split_type"].As<string>();
    // open feature file
    string feature_path = dataset_path + "/processed/node_features.dat";
    SPDLOG_INFO("Loading node features from {}", feature_path);
    SPDLOG_INFO("Performance mode: {}",
                config["performance"]["mode"].As<int>(2));
    showCpuMemCurrProc();
    SPDLOG_WARN("Read feature file");
    FileType load_file_mode =
        (FileType)(config["performance"]["mode"].As<int>(2));
    if (config["performance"]["uvm"].As<int>(0)) {
      SPDLOG_WARN("Using UVM, no need to load file");
      load_file_mode = (FileType)(4);  // empty
    }
    feature_handler_ = make_shared<FileHandler>(feature_path, load_file_mode);
    feature_handler_->openFile();
    SPDLOG_WARN("Done");
    showCpuMemCurrProc();
    feature_len_ =
        feature_handler_->filesize_ / sizeof(T) / graph_->getNumNode();
    SPDLOG_INFO("v {} e {} feat {}", graph_->getNumNode(), graph_->getNumEdge(),
                feature_len_);
  }

  void fetchData(const std::vector<Index> &index, int64_t feature_len,
                 Tensor &buf) override {
    feature_handler_->fetchData<T>(index, feature_len, buf);
  }

  void fetchAllData(Tensor &buf) override {
    feature_handler_->fetchAllData<T>(buf);
  }

  Tensor index2D(const Tensor &tensor,
                 const std::vector<Index> &index) override {
    ASSERTWITH(tensor.dim() == 2, "tensor should be 2D");
    int xdim = index.size(), ydim = tensor.size(1);
    // SPDLOG_WARN("Allocing xdim {} ydim {} tensor", xdim, ydim);
    Tensor result = torch::empty({xdim, ydim}, tensor.options());
    // SPDLOG_WARN("Successfully alloced");

    // SPDLOG_WARN("Fetching data");
    for (int i = 0; i < xdim; i++)
      memcpy(result.data<T>() + i * ydim, tensor.data<T>() + index[i] * ydim,
             ydim * sizeof(T));
    // SPDLOG_WARN("Done");
    return result;
  }
};

class Stage1Graph {
 public:
  shared_ptr<Graph> graph;
  Tensor x, y;
  shared_ptr<Split> split;
  shared_ptr<Dataset> parent_dataset;
  // flat_hash_map<Index, Index> inv_split;
  Stage1Graph(shared_ptr<Graph> graph, Tensor x, Tensor y,
              shared_ptr<Split> split, shared_ptr<Dataset> parent_dataset)
      : graph(graph),
        x(x),
        y(y),
        split(split),
        parent_dataset(parent_dataset) {}
};

#endif