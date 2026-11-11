// Single-token attention for the immutable CPU AR cache, with FP32 arithmetic.
#include <arm_neon.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/cpu/vec/vec.h>
#include <torch/library.h>
#include <cmath>
#include <vector>
static at::Tensor token_attention(const at::Tensor& q,const at::Tensor& keys,const at::Tensor& values,int64_t length,int64_t heads,const std::optional<at::Tensor>& mask) {
 TORCH_CHECK(q.device().is_cpu() && q.scalar_type()==at::kFloat && q.dim()==3 && q.size(0)==1 && q.size(1)==1,"Single CPU FP32 query required");
 const auto width=q.size(2),dim=width/heads;
 TORCH_CHECK(heads>0 && width%heads==0 && dim%4==0 && keys.is_contiguous() && values.is_contiguous() && keys.scalar_type()==at::kFloat && values.scalar_type()==at::kFloat && keys.device().is_cpu() && values.device().is_cpu() && keys.dim()==3 && values.sizes()==keys.sizes() && keys.size(0)==1 && keys.size(2)==width && length>0 && length<=keys.size(1),"Invalid attention cache");
 const bool* blocked=nullptr;
 if(mask){TORCH_CHECK(mask->device().is_cpu() && mask->scalar_type()==at::kBool && mask->is_contiguous() && mask->numel()==length,"Invalid blocked-position mask");blocked=mask->const_data_ptr<bool>();}
 auto output=at::empty(q.sizes(),q.options());auto qp=q.const_data_ptr<float>(),kp=keys.const_data_ptr<float>(),vp=values.const_data_ptr<float>();auto out=output.mutable_data_ptr<float>();
 std::vector<float> scores(length);float scale=1.f/std::sqrt(float(dim));
 for(int64_t h=0;h<heads;++h){
  float maxscore=-INFINITY;
  for(int64_t t=0;t<length;++t){auto acc=vdupq_n_f32(0);for(int64_t j=0;j<dim;j+=4)acc=vfmaq_f32(acc,vld1q_f32(qp+h*dim+j),vld1q_f32(kp+t*width+h*dim+j));float s=blocked&&blocked[t]?-INFINITY:vaddvq_f32(acc)*scale;scores[t]=s;maxscore=std::max(maxscore,s);}
  float sum=0;int64_t t=0;
  using Vec=at::vec::Vectorized<float>;
  for(;t+Vec::size()<=length;t+=Vec::size()){auto e=(Vec::loadu(scores.data()+t)-Vec(maxscore)).exp();e.store(scores.data()+t);}
  for(;t<length;++t)scores[t]=std::exp(scores[t]-maxscore);
  for(auto s:scores)sum+=s;
  const float inverse=1.f/sum;for(auto& s:scores)s*=inverse;
  for(int64_t j=0;j<dim;j+=4){auto acc=vdupq_n_f32(0);for(int64_t t=0;t<length;++t)acc=vfmaq_n_f32(acc,vld1q_f32(vp+t*width+h*dim+j),scores[t]);vst1q_f32(out+h*dim+j,acc);}
 }
 return output;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("token_attention(Tensor query, Tensor keys, Tensor values, int length, int heads, Tensor? mask) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("token_attention",TORCH_FN(token_attention));}
