// Isolated FP32 cache-locality probe. ACL retains its included MIT notice.
// Keep each 8x12 kernel's K accumulation and complex combine order unchanged.
#include <arm_neon.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/ops/empty.h>
#include <torch/library.h>
#include "acl_sgemm_8x12.cpp"

static void transpose4(float32x4_t* v) {
  auto t0=vtrn1q_f32(v[0],v[1]),t1=vtrn2q_f32(v[0],v[1]);
  auto t2=vtrn1q_f32(v[2],v[3]),t3=vtrn2q_f32(v[2],v[3]);
  v[0]=vcombine_f32(vget_low_f32(t0),vget_low_f32(t2));
  v[1]=vcombine_f32(vget_low_f32(t1),vget_low_f32(t3));
  v[2]=vcombine_f32(vget_high_f32(t0),vget_high_f32(t2));
  v[3]=vcombine_f32(vget_high_f32(t1),vget_high_f32(t3));
}

static at::Tensor fft_packed_output_tiles(
    const at::Tensor& x,const at::Tensor& w,
    int64_t batch,int64_t blocks,int64_t groups_per_tile) {
  TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu()
      && x.scalar_type()==at::kFloat && w.scalar_type()==at::kFloat
      && x.dim()==4 && w.dim()==4 && x.is_contiguous() && w.is_contiguous(),
      "Contiguous CPU FP32 packed spectral matrices required");
  TORCH_CHECK(x.size(0)>0 && x.size(0)==w.size(0) && x.size(0)%3==0
      && x.size(1)>0 && w.size(1)>0 && x.size(2)>0
      && x.size(2)==w.size(2) && x.size(3)==8 && w.size(3)==12
      && batch>0 && blocks>0 && batch*blocks<=x.size(1)*8
      && groups_per_tile>0 && groups_per_tile<=w.size(1),
      "Invalid output-tiled spectral geometry");
  const auto freq=x.size(0)/3,tiles=x.size(1),groups=w.size(1),k=x.size(2);
  const auto channels=groups*12,rows=batch*blocks;
  const auto chunks=(groups+groups_per_tile-1)/groups_per_tile;
  const auto product=tiles*groups_per_tile*96;
  auto y=at::empty({batch,channels,blocks,freq},x.options().dtype(at::kComplexFloat));
  auto a=x.const_data_ptr<float>(),b=w.const_data_ptr<float>();
  auto out=reinterpret_cast<float*>(y.mutable_data_ptr<c10::complex<float>>());
  at::parallel_for(0,((freq+3)/4)*chunks,1,[&](int64_t begin,int64_t end) {
    auto scratch=at::empty({3,4,product},x.options());
    auto work=scratch.mutable_data_ptr<float>();
    for(int64_t task=begin;task<end;++task) {
      const auto f0=(task/chunks)*4,go=(task%chunks)*groups_per_tile;
      const auto count=std::min<int64_t>(4,freq-f0);
      const auto gc=std::min<int64_t>(groups_per_tile,groups-go);
      for(int part=0;part<3;++part)for(int64_t f=0;f<count;++f) {
        auto plane=part*freq+f0+f;
        arm_gemm::a64_sgemm_asimd_8x12(
            a+plane*tiles*k*8,b+(plane*groups+go)*k*12,
            work+(part*4+f)*product,int(tiles),int(gc),int(k));
      }
      for(int64_t t=0;t<tiles;++t)for(int64_t g=0;g<gc;++g)
      for(int64_t r=0;r<8 && t*8+r<rows;++r)for(int64_t z=0;z<12;z+=4) {
        const auto row=t*8+r,index=(t*gc+g)*96+r*12+z;
        float32x4_t real[4],imag[4];
        for(int64_t f=0;f<count;++f) {
          auto va=vld1q_f32(work+f*product+index);
          auto vb=vld1q_f32(work+(4+f)*product+index);
          auto vc=vld1q_f32(work+(8+f)*product+index);
          real[f]=vsubq_f32(va,vb);imag[f]=vsubq_f32(vsubq_f32(vc,va),vb);
        }
        if(count==4) {
          transpose4(real);transpose4(imag);
          for(int j=0;j<4;++j) {
            auto dest=(((row/blocks)*channels+(go+g)*12+z+j)*blocks+row%blocks)*freq+f0;
            float32x4x2_t v;v.val[0]=real[j];v.val[1]=imag[j];
            vst2q_f32(out+2*dest,v);
          }
        } else for(int64_t f=0;f<count;++f) {
          float re[4],im[4];vst1q_f32(re,real[f]);vst1q_f32(im,imag[f]);
          for(int j=0;j<4;++j) {
            auto dest=(((row/blocks)*channels+(go+g)*12+z+j)*blocks+row%blocks)*freq+f0+f;
            out[2*dest]=re[j];out[2*dest+1]=im[j];
          }
        }
      }
    }
  });
  return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m) {
  m.def("fft_packed_output_tiles(Tensor input, Tensor weight, int batch, int blocks, int groups_per_tile) -> Tensor");
}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m) {
  m.impl("fft_packed_output_tiles",TORCH_FN(fft_packed_output_tiles));
}
