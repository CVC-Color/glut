#include <torch/extension.h>

torch::Tensor glut_forward_cuda(
    torch::Tensor rgb,
    torch::Tensor positions,
    torch::Tensor prec,
    torch::Tensor log_det,
    torch::Tensor opacities,
    torch::Tensor color_mat,
    torch::Tensor color_bias,
    torch::Tensor global_mat,
    torch::Tensor global_bias,
    bool residual,
    bool weight_norm
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &glut_forward_cuda,
          "Gaussian LUT forward (CUDA)",
          py::arg("rgb"),
          py::arg("positions"),
          py::arg("prec"),
          py::arg("log_det"),
          py::arg("opacities"),
          py::arg("color_mat"),
          py::arg("color_bias"),
          py::arg("global_mat"),
          py::arg("global_bias"),
          py::arg("residual"),
          py::arg("weight_norm") = true);

}

