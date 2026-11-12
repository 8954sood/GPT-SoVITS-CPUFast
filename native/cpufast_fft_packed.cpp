// Prepacked FP32 spectral GEMM. The included ARM kernel retains its MIT notice.
#include <arm_neon.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/ops/empty.h>
#include <torch/library.h>
#include "acl_sgemm_8x12.cpp"
static void transpose4(float32x4_t* v);
static at::Tensor fft_pack(const at::Tensor& x){
 TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kComplexFloat && x.dim()==4 && x.is_contiguous(),"Contiguous complex64 FFT required");
 const auto batch=x.size(0),channels=x.size(1),blocks=x.size(2),freq=x.size(3),rows=batch*blocks,tiles=(rows+7)/8,plane=freq*tiles*channels*8;
 auto y=at::empty({3*freq,tiles,channels,8},x.options().dtype(at::kFloat));auto src=x.const_data_ptr<c10::complex<float>>();auto dst=y.mutable_data_ptr<float>();
 at::parallel_for(0,(freq+15)/16,1,[&](int64_t begin,int64_t end){for(int64_t fi=begin;fi<end;++fi){int64_t f0=fi*16;for(int64_t t=0;t<tiles;++t)for(int64_t c=0;c<channels;++c){
  int64_t f=f0,stop=std::min(freq,f0+16);
  for(;f+4<=stop;f+=4)for(int half=0;half<2;++half){float32x4_t re[4],im[4];for(int j=0;j<4;++j){auto row=t*8+half*4+j;if(row<rows){auto index=(((row/blocks)*channels+c)*blocks+row%blocks)*freq+f;auto z=vld2q_f32(reinterpret_cast<const float*>(src+index));re[j]=z.val[0];im[j]=z.val[1];}else{re[j]=vdupq_n_f32(0);im[j]=vdupq_n_f32(0);}}
   transpose4(re);transpose4(im);for(int j=0;j<4;++j){auto index=(((f+j)*tiles+t)*channels+c)*8+half*4;vst1q_f32(dst+index,re[j]);vst1q_f32(dst+plane+index,im[j]);vst1q_f32(dst+2*plane+index,vaddq_f32(re[j],im[j]));}
  }
  for(;f<stop;++f)for(int j=0;j<8;++j){auto row=t*8+j;float re=0,im=0;if(row<rows){auto v=src[(((row/blocks)*channels+c)*blocks+row%blocks)*freq+f];re=v.real();im=v.imag();}auto index=((f*tiles+t)*channels+c)*8+j;dst[index]=re;dst[plane+index]=im;dst[2*plane+index]=re+im;}
 }}});
 return y;
}
static at::Tensor fft_packed_gemm(const at::Tensor& x,const at::Tensor& w){
 TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu() && x.scalar_type()==at::kFloat && w.scalar_type()==at::kFloat && x.dim()==4 && w.dim()==4 && x.is_contiguous() && w.is_contiguous() && x.size(0)==w.size(0) && x.size(2)==w.size(2) && x.size(3)==8 && w.size(3)==12,"Invalid prepacked spectral matrices");
 auto batch=x.size(0),m=x.size(1),n=w.size(1),k=x.size(2);auto y=at::empty({batch,m,n,8,12},x.options());auto a=x.const_data_ptr<float>(),b=w.const_data_ptr<float>();auto out=y.mutable_data_ptr<float>();
 at::parallel_for(0,batch,1,[&](int64_t begin,int64_t end){for(int64_t i=begin;i<end;++i)arm_gemm::a64_sgemm_asimd_8x12(a+i*m*k*8,b+i*n*k*12,out+i*m*n*96,int(m),int(n),int(k));});return y;
}
static void transpose4(float32x4_t* v){
 auto t0=vtrn1q_f32(v[0],v[1]),t1=vtrn2q_f32(v[0],v[1]),t2=vtrn1q_f32(v[2],v[3]),t3=vtrn2q_f32(v[2],v[3]);
 v[0]=vcombine_f32(vget_low_f32(t0),vget_low_f32(t2));v[1]=vcombine_f32(vget_low_f32(t1),vget_low_f32(t3));v[2]=vcombine_f32(vget_high_f32(t0),vget_high_f32(t2));v[3]=vcombine_f32(vget_high_f32(t1),vget_high_f32(t3));
}
static at::Tensor fft_unpack(const at::Tensor& x,int64_t rows){
 TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kFloat && x.dim()==5 && x.is_contiguous() && x.size(0)%3==0 && x.size(3)==8 && x.size(4)==12 && rows>0 && rows<=x.size(1)*8,"Invalid packed spectral products");
 auto freq=x.size(0)/3,tiles=x.size(1),groups=x.size(2),channels=groups*12,plane=freq*tiles*groups*96;
 auto y=at::empty({freq,channels,rows},x.options().dtype(at::kComplexFloat));auto a=x.const_data_ptr<float>(),b=a+plane,c=b+plane;auto out=reinterpret_cast<float*>(y.mutable_data_ptr<c10::complex<float>>());
 for(int64_t f=0;f<freq;++f)for(int64_t t=0;t<tiles;++t)for(int64_t g=0;g<groups;++g)for(int r=0;r<8;r+=4)for(int z=0;z<12;z+=4){
  auto base=((f*tiles+t)*groups+g)*96+r*12+z;float32x4_t real[4],imag[4];for(int j=0;j<4;++j){auto va=vld1q_f32(a+base+j*12),vb=vld1q_f32(b+base+j*12),vc=vld1q_f32(c+base+j*12);real[j]=vsubq_f32(va,vb);imag[j]=vsubq_f32(vsubq_f32(vc,va),vb);}transpose4(real);transpose4(imag);
  for(int j=0;j<4;++j){auto index=((f*channels+g*12+z+j)*rows+t*8+r)*2;float32x4x2_t val;val.val[0]=real[j];val.val[1]=imag[j];if(t*8+r+4<=rows)vst2q_f32(out+index,val);else{float re[4],im[4];vst1q_f32(re,real[j]);vst1q_f32(im,imag[j]);for(int s=0;s<4 && t*8+r+s<rows;++s){out[index+2*s]=re[s];out[index+2*s+1]=im[s];}}}
 }
 return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("fft_pack(Tensor input) -> Tensor");m.def("fft_packed_gemm(Tensor input, Tensor weight) -> Tensor");m.def("fft_unpack(Tensor products, int rows) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("fft_pack",TORCH_FN(fft_pack));m.impl("fft_packed_gemm",TORCH_FN(fft_packed_gemm));m.impl("fft_unpack",TORCH_FN(fft_unpack));}
static at::Tensor fft_unpack_layout(const at::Tensor& x,int64_t batch,int64_t blocks){
 TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kFloat && x.dim()==5 && x.is_contiguous() && x.size(0)%3==0 && x.size(3)==8 && x.size(4)==12 && batch>0 && blocks>0 && batch*blocks<=x.size(1)*8,"Invalid packed products");
 auto freq=x.size(0)/3,tiles=x.size(1),groups=x.size(2),channels=groups*12,rows=batch*blocks,plane=freq*tiles*groups*96;
 auto y=at::empty({batch,channels,blocks,freq},x.options().dtype(at::kComplexFloat));auto a=x.const_data_ptr<float>(),b=a+plane,c=b+plane;auto out=reinterpret_cast<float*>(y.mutable_data_ptr<c10::complex<float>>());
 at::parallel_for(0,(freq+15)/16,1,[&](int64_t begin,int64_t end){for(int64_t fi=begin;fi<end;++fi){int64_t f0=fi*16;for(int64_t t=0;t<tiles;++t)for(int64_t g=0;g<groups;++g)for(int r=0;r<8 && t*8+r<rows;++r)for(int z=0;z<12;z+=4){
  const auto row=t*8+r;int64_t f=f0,stop=std::min(freq,f0+16);
  for(;f+4<=stop;f+=4){float32x4_t real[4],imag[4];for(int j=0;j<4;++j){auto index=(((f+j)*tiles+t)*groups+g)*96+r*12+z;auto va=vld1q_f32(a+index),vb=vld1q_f32(b+index),vc=vld1q_f32(c+index);real[j]=vsubq_f32(va,vb);imag[j]=vsubq_f32(vsubq_f32(vc,va),vb);}transpose4(real);transpose4(imag);
   for(int j=0;j<4;++j){auto index=(((row/blocks)*channels+g*12+z+j)*blocks+row%blocks)*freq+f;float32x4x2_t v;v.val[0]=real[j];v.val[1]=imag[j];vst2q_f32(out+2*index,v);}
  }
  for(;f<stop;++f){auto index=((f*tiles+t)*groups+g)*96+r*12+z;auto va=vld1q_f32(a+index),vb=vld1q_f32(b+index),vc=vld1q_f32(c+index);float re[4],im[4];vst1q_f32(re,vsubq_f32(va,vb));vst1q_f32(im,vsubq_f32(vsubq_f32(vc,va),vb));for(int j=0;j<4;++j){auto dest=(((row/blocks)*channels+g*12+z+j)*blocks+row%blocks)*freq+f;out[2*dest]=re[j];out[2*dest+1]=im[j];}}
 }}});
 return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("fft_unpack_layout(Tensor products, int batch, int blocks) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("fft_unpack_layout",TORCH_FN(fft_unpack_layout));}

// Fuse each four-frequency GEMM group with complex/layout reconstruction.
// The existing kernel and (c-a)-b combination order are retained verbatim.
static at::Tensor fft_packed_stream(const at::Tensor& x,const at::Tensor& w,int64_t batch,int64_t blocks){
 TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu() && x.scalar_type()==at::kFloat && w.scalar_type()==at::kFloat && x.dim()==4 && w.dim()==4 && x.is_contiguous() && w.is_contiguous() && x.size(0)==w.size(0) && x.size(0)%3==0 && x.size(2)==w.size(2) && x.size(3)==8 && w.size(3)==12 && batch>0 && blocks>0 && batch*blocks<=x.size(1)*8,"Invalid streaming spectral matrices");
 const auto freq=x.size(0)/3,tiles=x.size(1),groups=w.size(1),k=x.size(2),channels=groups*12,rows=batch*blocks,product=tiles*groups*96;
 auto y=at::empty({batch,channels,blocks,freq},x.options().dtype(at::kComplexFloat));
 auto a=x.const_data_ptr<float>(),b=w.const_data_ptr<float>();auto out=reinterpret_cast<float*>(y.mutable_data_ptr<c10::complex<float>>());
 at::parallel_for(0,(freq+3)/4,1,[&](int64_t begin,int64_t end){
  auto scratch=at::empty({3,4,product},x.options());auto work=scratch.mutable_data_ptr<float>();
  for(int64_t fi=begin;fi<end;++fi){const auto f0=fi*4,count=std::min<int64_t>(4,freq-f0);
   for(int part=0;part<3;++part)for(int64_t f=0;f<count;++f){auto plane=part*freq+f0+f;
    arm_gemm::a64_sgemm_asimd_8x12(a+plane*tiles*k*8,b+plane*groups*k*12,work+(part*4+f)*product,int(tiles),int(groups),int(k));
   }
   for(int64_t t=0;t<tiles;++t)for(int64_t g=0;g<groups;++g)for(int64_t r=0;r<8 && t*8+r<rows;++r)for(int64_t z=0;z<12;z+=4){
    const auto row=t*8+r,index=(t*groups+g)*96+r*12+z;
    float32x4_t real[4],imag[4];
    for(int64_t f=0;f<count;++f){auto va=vld1q_f32(work+f*product+index),vb=vld1q_f32(work+(4+f)*product+index),vc=vld1q_f32(work+(8+f)*product+index);real[f]=vsubq_f32(va,vb);imag[f]=vsubq_f32(vsubq_f32(vc,va),vb);}
    if(count==4){transpose4(real);transpose4(imag);for(int j=0;j<4;++j){auto dest=(((row/blocks)*channels+g*12+z+j)*blocks+row%blocks)*freq+f0;float32x4x2_t v;v.val[0]=real[j];v.val[1]=imag[j];vst2q_f32(out+2*dest,v);}}
    else for(int64_t f=0;f<count;++f){float re[4],im[4];vst1q_f32(re,real[f]);vst1q_f32(im,imag[f]);for(int j=0;j<4;++j){auto dest=(((row/blocks)*channels+g*12+z+j)*blocks+row%blocks)*freq+f0+f;out[2*dest]=re[j];out[2*dest+1]=im[j];}}
   }
  }
 });return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("fft_packed_stream(Tensor input, Tensor weight, int batch, int blocks) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("fft_packed_stream",TORCH_FN(fft_packed_stream));}
