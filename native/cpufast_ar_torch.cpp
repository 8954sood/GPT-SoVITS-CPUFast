// Same FP32 GEMV, registered directly with PyTorch to avoid Python callbacks.
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <torch/library.h>
#include "cpufast_ar_gemv.cpp"

static at::Tensor lossless_linear(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b) {
 TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu() && b.device().is_cpu(), "CPU tensors required");
 TORCH_CHECK(x.scalar_type()==at::kFloat && w.scalar_type()==at::kHalf && b.scalar_type()==at::kFloat, "FP32 token and exact half cache required");
 TORCH_CHECK(w.dim()==3 && w.size(2)==8 && w.size(1)%4==0 && x.dim()>0 && x.numel()==w.size(1) && x.size(-1)==w.size(1) && b.numel()==w.size(0)*8, "Invalid GEMV geometry");
 TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && b.is_contiguous(), "Contiguous tensors required");
 auto shape=x.sizes().vec();shape.back()=w.size(0)*8;
 auto output=at::empty(shape,x.options());
 cpufast_ar_gemv(x.const_data_ptr<float>(),w.const_data_ptr<at::Half>(),b.const_data_ptr<float>(),output.mutable_data_ptr<float>(),shape.back(),w.size(1),1,0);
 return output;
}
TORCH_LIBRARY(cpufast_cpp, m) {
 m.def("lossless_linear(Tensor x, Tensor weight, Tensor bias) -> Tensor");
}
TORCH_LIBRARY_IMPL(cpufast_cpp, CPU, m) {
 m.impl("lossless_linear", TORCH_FN(lossless_linear));
}
