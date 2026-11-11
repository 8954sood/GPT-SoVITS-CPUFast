// Fused FP32 spectral rearrangement and Gauss-product combination.
#include <arm_neon.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <torch/library.h>
static at::Tensor fft_split(const at::Tensor& x) {
 TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kComplexFloat && x.dim()==4 && x.is_contiguous(),"Contiguous complex64 FFT blocks required");
 const auto batch=x.size(0),channels=x.size(1),blocks=x.size(2),freq=x.size(3),rows=batch*blocks,plane=freq*channels*rows;
 auto y=at::empty({3*freq,channels,rows},x.options().dtype(at::kFloat));
 auto src=x.const_data_ptr<c10::complex<float>>();auto out=y.mutable_data_ptr<float>();
 for(int64_t b=0;b<batch;++b)for(int64_t c=0;c<channels;++c)for(int64_t f0=0;f0<freq;f0+=16)for(int64_t m0=0;m0<blocks;m0+=16) {
  const auto fstop=std::min<int64_t>(freq,f0+16),mstop=std::min<int64_t>(blocks,m0+16);
  for(int64_t f=f0;f<fstop;++f)for(int64_t m=m0;m<mstop;++m) {
   auto v=src[((b*channels+c)*blocks+m)*freq+f];auto i=(f*channels+c)*rows+b*blocks+m;
   out[i]=v.real();out[plane+i]=v.imag();out[2*plane+i]=v.real()+v.imag();
  }
 }
 return y;
}
static at::Tensor fft_combine(const at::Tensor& x) {
 TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kFloat && x.dim()==3 && x.is_contiguous() && x.size(0)%3==0,"Contiguous triple FP32 products required");
 auto freq=x.size(0)/3,count=freq*x.size(1)*x.size(2);
 auto y=at::empty({freq,x.size(1),x.size(2)},x.options().dtype(at::kComplexFloat));
 const float* a=x.const_data_ptr<float>();const float* b=a+count;const float* c=b+count;auto out=reinterpret_cast<float*>(y.mutable_data_ptr<c10::complex<float>>());
 int64_t i=0;for(;i+4<=count;i+=4) {
  auto va=vld1q_f32(a+i),vb=vld1q_f32(b+i),vc=vld1q_f32(c+i);float32x4x2_t v;
  v.val[0]=vsubq_f32(va,vb);v.val[1]=vsubq_f32(vsubq_f32(vc,va),vb);vst2q_f32(out+2*i,v);
 }
 for(;i<count;++i){out[2*i]=a[i]-b[i];out[2*i+1]=(c[i]-a[i])-b[i];}
 return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("fft_split(Tensor input) -> Tensor");m.def("fft_combine(Tensor products) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("fft_split",TORCH_FN(fft_split));m.impl("fft_combine",TORCH_FN(fft_combine));}
static at::Tensor fft_combine_layout(const at::Tensor& x,int64_t batch) {
 TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kFloat && x.dim()==3 && x.is_contiguous() && x.size(0)%3==0 && batch>0 && x.size(2)%batch==0,"Invalid FFT products");
 const auto freq=x.size(0)/3,channels=x.size(1),rows=x.size(2),blocks=rows/batch,count=freq*channels*rows;
 auto y=at::empty({batch,channels,blocks,freq},x.options().dtype(at::kComplexFloat));auto a=x.const_data_ptr<float>();auto b=a+count;auto c=b+count;auto out=y.mutable_data_ptr<c10::complex<float>>();
 for(int64_t n=0;n<batch;++n)for(int64_t ch=0;ch<channels;++ch)for(int64_t m0=0;m0<blocks;m0+=16)for(int64_t f0=0;f0<freq;f0+=16)
  for(int64_t m=m0;m<std::min(blocks,m0+16);++m)for(int64_t f=f0;f<std::min(freq,f0+16);++f){auto i=(f*channels+ch)*rows+n*blocks+m;out[((n*channels+ch)*blocks+m)*freq+f]={a[i]-b[i],(c[i]-a[i])-b[i]};}
 return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("fft_combine_layout(Tensor products, int batch) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("fft_combine_layout",TORCH_FN(fft_combine_layout));}
