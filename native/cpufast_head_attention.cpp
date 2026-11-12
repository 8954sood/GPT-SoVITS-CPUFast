// FP32 head-major attention experiment; no quantized weights or activations.
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
#include <cmath>
#include <vector>
static at::Tensor head_attn_fp32(const at::Tensor& q,const at::Tensor& keys,const at::Tensor& values,int64_t length,int64_t heads,const std::optional<at::Tensor>& mask){
 TORCH_CHECK(q.device().is_cpu() && keys.device().is_cpu() && values.device().is_cpu() && q.scalar_type()==at::kFloat && keys.scalar_type()==at::kFloat && values.scalar_type()==at::kFloat && q.is_contiguous() && q.dim()==3 && q.size(0)==1 && q.size(1)==1 && keys.is_contiguous() && values.is_contiguous() && keys.dim()==4 && keys.size(0)==1 && heads>0 && keys.size(1)==heads && values.sizes()==keys.sizes() && length>0 && length<=keys.size(2),"head layout required");
 int64_t dim=keys.size(3),maxlen=keys.size(2); TORCH_CHECK(q.numel()==heads*dim && dim%4==0,"bad dimensions");const bool* blocked=nullptr;if(mask){TORCH_CHECK(mask->device().is_cpu() && mask->is_contiguous() && mask->scalar_type()==at::kBool && (mask->numel()==length || mask->numel()==heads*length),"bad mask");blocked=mask->const_data_ptr<bool>();}
 auto y=at::empty(q.sizes(),q.options());auto qp=q.const_data_ptr<float>();auto kp=keys.const_data_ptr<float>();auto vp=values.const_data_ptr<float>();auto out=y.mutable_data_ptr<float>();float scale=1.f/std::sqrt(float(dim));
 at::parallel_for(0,heads,1,[&](int64_t first,int64_t last){std::vector<float> scores(length);for(int64_t h=first;h<last;h++){const float* k=kp+h*maxlen*dim;const float* v=vp+h*maxlen*dim;float mx=-INFINITY;for(int64_t t=0;t<length;t++){auto a=vdupq_n_f32(0);for(int64_t j=0;j<dim;j+=4)a=vfmaq_f32(a,vld1q_f32(qp+h*dim+j),vld1q_f32(k+t*dim+j));float z=blocked&&blocked[(mask->numel()==length ? 0 : h*length)+t]?-INFINITY:vaddvq_f32(a)*scale;scores[t]=z;mx=std::max(mx,z);}if(mx==-INFINITY){for(int64_t j=0;j<dim;j++)out[h*dim+j]=0;continue;}float sum=0;for(int64_t t=0;t<length;t++){scores[t]=std::exp(scores[t]-mx);sum+=scores[t];}float inv=1.f/sum;for(int64_t j=0;j<dim;j+=4){auto a=vdupq_n_f32(0);for(int64_t t=0;t<length;t++)a=vfmaq_n_f32(a,vld1q_f32(v+t*dim+j),scores[t]*inv);vst1q_f32(out+h*dim+j,a);}}});return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("head_attn_fp32(Tensor q, Tensor keys, Tensor values, int length, int heads, Tensor? mask) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("head_attn_fp32",TORCH_FN(head_attn_fp32));}
