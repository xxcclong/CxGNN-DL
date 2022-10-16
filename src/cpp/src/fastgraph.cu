#include <string>

#include "fastgraph.h"

shared_ptr<Split> FastGraph::getSplit(const std::string &split_name) {
  if (split_name == "train")
    return train_split;
  else if (split_name == "valid")
    return val_split;
  else if (split_name == "test")
    return test_split;
  else
    ASSERTWITH(false, "unknown split {}", split_name);
  return nullptr;
}

shared_ptr<Dataset> FastGraph::getDataset() { return dataset; }

void FastGraph::initLogLevel(Yaml::Node &config) {
  spdlog::set_pattern("[%H-%M-%S-%e] [%n] [thread %t] [%s:%# %!] %v");
  int log_level = config["performance"]["log_level"].As<int>();
  ASSERTWITH(log_level >= 0, "specify log level {} {}", log_level,
             config["performance"]["mode"].As<int>());
  if (log_level == 0)
    spdlog::set_level(spdlog::level::debug);
  else if (log_level == 1)
    spdlog::set_level(spdlog::level::info);
  else
    spdlog::set_level(spdlog::level::warn);
}

void FastGraph::initSplit(Yaml::Node &config) {
  bool train_shuffle = config["train"]["train_shuffle"].As<bool>(true);
  bool eval_shuffle = config["train"]["eval_shuffle"].As<bool>(false);
  int num_train_samples = config["train"]["num_train_samples"].As<int>(-1);
  train_split = std::make_shared<Split>(config, "train", num_train_samples,
                                        train_shuffle);
  int num_val_samples = config["train"]["num_val_samples"].As<int>(-1);
  int num_test_samples = config["train"]["num_test_samples"].As<int>(-1);
  val_split =
      std::make_shared<Split>(config, "valid", num_val_samples, eval_shuffle);
  test_split =
      std::make_shared<Split>(config, "test", num_test_samples, eval_shuffle);
}

void FastGraph::initDataset(Yaml::Node &config) {
  dataset = Dataset::create(config);
}

void FastGraph::initLoaders(Yaml::Node &config) {
  train_loader = AbstractLoader::create(config, dataset, train_split, "train");
  val_loader = AbstractLoader::create(config, dataset, val_split, "eval");
  test_loader = AbstractLoader::create(config, dataset, test_split, "eval");
}

void FastGraph::start(const std::string &split_name) {
  if (split_name == "train")
    loader = train_loader;
  else if (split_name == "valid")
    loader = val_loader;
  else if (split_name == "test")
    loader = test_loader;
  else
    ASSERTWITH(false, "unknown split {}", split_name);
  loader->start();
}

shared_ptr<Batch> FastGraph::get_batch() { return loader->get_batch(); }

FastGraph::FastGraph(const std::string &yaml_filename) {
  SPDLOG_WARN("init SYS");
  showCpuMemCurrProc();
  Yaml::Node config;
  SPDLOG_INFO("YAML filename: {}", yaml_filename);
  if (fexist(yaml_filename))
    Yaml::Parse(config, yaml_filename.c_str());
  else
    ASSERT(false);
  initLogLevel(config);
  initDataset(config);
  initSplit(config);
  initLoaders(config);
}