// Fuse overlap-save crop, bias, activation and residual without changing FP32 order.
#include <arm_neon.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/ops/empty.h>
#include <torch/library.h>
static at::Tensor fft_finish(const at::Tensor& x,const std::optional<at::Tensor>& bias,int64_t effective,int64_t length,bool activate,const std::optional<at::Tensor>& residual,bool parallel){
 TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kFloat && x.dim()==4 && x.is_contiguous(),"Contiguous FP32 inverse FFT blocks required");
 auto batch=x.size(0),channels=x.size(1),blocks=x.size(2),fft=x.size(3),hop=fft-effective+1;
 TORCH_CHECK(effective>0 && hop>0 && length>0 && effective-1+length<=blocks*hop,"Invalid overlap-save crop");
 if(bias)TORCH_CHECK(bias->device().is_cpu() && bias->scalar_type()==at::kFloat && bias->is_contiguous() && bias->numel()==channels,"Invalid bias");
 if(residual)TORCH_CHECK(residual->device().is_cpu() && residual->scalar_type()==at::kFloat && residual->stride(2)==1 && residual->sizes()==at::IntArrayRef({batch,channels,length}),"Invalid residual");
 auto y=at::empty({batch,channels,length},x.options());auto src=x.const_data_ptr<float>();auto dst=y.mutable_data_ptr<float>();auto b=bias?bias->const_data_ptr<float>():nullptr;auto res=residual?residual->const_data_ptr<float>():nullptr;
 auto compute=[&](int64_t begin,int64_t end){for(int64_t index=begin;index<end;++index){const auto n=index/channels,c=index%channels;
  auto source=src+(n*channels+c)*blocks*fft;auto out=dst+(n*channels+c)*length;auto rr=res?res+n*residual->stride(0)+c*residual->stride(1):nullptr;auto vb=vdupq_n_f32(b?b[c]:0.f);
  for(int64_t p=0;p<length;){auto index=p+effective-1,block=index/hop,offset=index%hop,take=std::min(length-p,hop-offset);auto in=source+block*fft+effective-1+offset;int64_t j=0;
   for(;j+4<=take;j+=4){auto v=vld1q_f32(in+j);if(b)v=vaddq_f32(v,vb);if(activate)v=vbslq_f32(vcgtq_f32(v,vdupq_n_f32(0)),v,vmulq_n_f32(v,0.1f));if(rr)v=vaddq_f32(v,vld1q_f32(rr+p+j));vst1q_f32(out+p+j,v);}
   for(;j<take;++j){float v=in[j];if(b)v+=b[c];if(activate)v=v>0?v:v*0.1f;if(rr)v+=rr[p+j];out[p+j]=v;}p+=take;
  }
 }};
 if(parallel)at::parallel_for(0,batch*channels,1,compute);else compute(0,batch*channels);
 return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("fft_finish(Tensor blocks, Tensor? bias, int effective_kernel, int length, bool activate=False, Tensor? residual=None, bool parallel=False) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("fft_finish",TORCH_FN(fft_finish));}
