#include <memory.h>
#include <pybind11/pybind11.h>

#include "batch.h"
#include "common.h"
#include "fastgraph.h"
#include "loader.h"
namespace py = pybind11;
using namespace pybind11::literals;

void init_batch(py::module &m) {
  py::class_<Batch, std::shared_ptr<Batch>>(m, "Batch")
      .def_readwrite("edge_index", &Batch::edge_index)
      .def_readwrite("x", &Batch::x)
      .def_readwrite("y", &Batch::y)
      .def_readwrite("mask", &Batch::mask)
      .def_readwrite("ptr", &Batch::ptr)
      .def_readwrite("idx", &Batch::idx)
      .def_readwrite("num_node_in_layer", &Batch::num_node_in_layer)
      .def_readwrite("num_edge_in_layer", &Batch::num_edge_in_layer)
      .def_readwrite("sub_to_full", &Batch::sub_to_full)
      .def_readwrite("edge_type", &Batch::edge_type)
      .def_readwrite("edge_indexs", &Batch::edge_indexs)
      .def_readwrite("xs", &Batch::xs)
      .def_readwrite("ys", &Batch::ys)
      .def_readwrite("masks", &Batch::masks)
      .def_readwrite("ptrs", &Batch::ptrs)
      .def_readwrite("idxs", &Batch::idxs)
      .def_readwrite("sub_to_fulls", &Batch::sub_to_fulls)
      .def_readwrite("etype_partition", &Batch::etype_partition)
      .def_readwrite("typed_num_node_in_layer",
                     &Batch::typed_num_node_in_layer);
}

void init_fastgraph(py::module &m) {
  py::class_<FastGraph>(m, "FastGraph")
      .def(py::init<const std::string &>())
      .def("get_batch", &FastGraph::get_batch)
      .def("start", &FastGraph::start)
      .def("num_iters", &FastGraph::num_iters)
      .def_readwrite("train_loader", &FastGraph::train_loader)
      .def_readwrite("val_loader", &FastGraph::val_loader)
      .def_readwrite("test_loader", &FastGraph::test_loader);
}

void init_util(py::module &m) {
  m.def("show_mem_curr_proc", &showCpuMemCurrProc,
        "Show memory usage of current proc");
}

PYBIND11_MODULE(cxgnndl, m) {
  m.doc() = "A Supa Fast Graph Loader";
  init_batch(m);
  init_fastgraph(m);
  init_util(m);
}
