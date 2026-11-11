// Same FP32 GEMV, registered directly with PyTorch to avoid Python callbacks.
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
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
 auto xp=x.const_data_ptr<float>(); auto wp=w.const_data_ptr<at::Half>(); auto bp=b.const_data_ptr<float>(); auto yp=output.mutable_data_ptr<float>(); auto blocks=w.size(0), k=w.size(1);
 at::parallel_for(0,blocks,8,[&](int64_t start,int64_t end){cpufast_ar_gemv(xp,wp+start*k*8,bp+start*8,yp+start*8,(end-start)*8,k,1,0);});
 return output;
}
TORCH_LIBRARY(cpufast_cpp, m) {
 m.def("lossless_linear(Tensor x, Tensor weight, Tensor bias) -> Tensor");
}
TORCH_LIBRARY_IMPL(cpufast_cpp, CPU, m) {
 m.impl("lossless_linear", TORCH_FN(lossless_linear));
}
static at::Tensor lossless_linear16(const at::Tensor& x,const at::Tensor& w,const at::Tensor& b){
 TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu() && b.device().is_cpu() && x.scalar_type()==at::kFloat && w.scalar_type()==at::kHalf && b.scalar_type()==at::kFloat,"CPU FP32 input and exact half cache required");
 TORCH_CHECK(w.dim()==3 && w.size(2)==16 && w.size(1)%4==0 && x.dim()>0 && x.numel()==w.size(1) && x.size(-1)==w.size(1) && b.numel()==w.size(0)*16 && x.is_contiguous() && w.is_contiguous() && b.is_contiguous(),"Invalid GEMV16 geometry");
 auto shape=x.sizes().vec();shape.back()=w.size(0)*16;auto y=at::empty(shape,x.options());
 auto xp=x.const_data_ptr<float>(); auto wp=reinterpret_cast<const float16_t*>(w.const_data_ptr<at::Half>()); auto bp=b.const_data_ptr<float>(); auto yp=y.mutable_data_ptr<float>(); auto blocks=w.size(0),k=w.size(1);
 at::parallel_for(0,blocks,4,[&](int64_t start,int64_t end){gemv16(xp,wp+start*k*16,bp+start*16,yp+start*16,(end-start)*16,k);});return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("lossless_linear16(Tensor input, Tensor weight, Tensor bias) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("lossless_linear16",TORCH_FN(lossless_linear16));}
