// Unlinked research candidate: preserve fft_hgemm FP16 accumulation and unpack
// arithmetic while bounding intermediate products to four frequency bins.
// Deliberately absent from build_cpufast_fast.py until the P profile is reviewed.
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
#include <algorithm>

static void half_plane(const float16_t* a, const float16_t* b, float* out,
                       int64_t tiles, int64_t groups, int64_t k) {
  for (int64_t t=0; t<tiles; ++t) for (int64_t g=0; g<groups; ++g) {
    const auto x=a+t*k*8, w=b+g*k*12;
    float16x8_t acc[12];
    for (int c=0; c<12; ++c) acc[c]=vdupq_n_f16(0);
    for (int64_t j=0; j<k; ++j) {
      auto z=vld1q_f16(x+j*8);
      for (int c=0; c<12; ++c) acc[c]=vfmaq_n_f16(acc[c],z,w[j*12+c]);
    }
    alignas(16) float16_t tmp[12][8];
    for (int c=0; c<12; ++c) vst1q_f16(tmp[c],acc[c]);
    auto p=out+(t*groups+g)*96;
    for (int r=0; r<8; ++r) for (int c=0; c<12; ++c) p[r*12+c]=float(tmp[c][r]);
  }
}

static void transpose4(float32x4_t* v) {
  auto t0=vtrn1q_f32(v[0],v[1]), t1=vtrn2q_f32(v[0],v[1]);
  auto t2=vtrn1q_f32(v[2],v[3]), t3=vtrn2q_f32(v[2],v[3]);
  v[0]=vcombine_f32(vget_low_f32(t0),vget_low_f32(t2));
  v[1]=vcombine_f32(vget_low_f32(t1),vget_low_f32(t3));
  v[2]=vcombine_f32(vget_high_f32(t0),vget_high_f32(t2));
  v[3]=vcombine_f32(vget_high_f32(t1),vget_high_f32(t3));
}

static at::Tensor half_stream(const at::Tensor& x, const at::Tensor& w,
                              int64_t batch, int64_t blocks) {
  TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu() &&
      x.scalar_type()==at::kHalf && w.scalar_type()==at::kHalf &&
      x.is_contiguous() && w.is_contiguous() && x.dim()==4 && w.dim()==4 &&
      x.size(0)==w.size(0) && x.size(0)>0 && x.size(0)%3==0 &&
      x.size(2)==w.size(2) && x.size(2)>0 && x.size(3)==8 && w.size(3)==12 &&
      w.size(1)>0 && batch>0 && blocks>0 && batch*blocks<=x.size(1)*8,
      "Invalid FP16 streaming spectral matrices");
  const auto freq=x.size(0)/3, tiles=x.size(1), groups=w.size(1), k=x.size(2);
  const auto channels=groups*12, rows=batch*blocks, product=tiles*groups*96;
  auto y=at::empty({batch,channels,blocks,freq},x.options().dtype(at::kComplexFloat));
  auto a=reinterpret_cast<const float16_t*>(x.const_data_ptr<at::Half>());
  auto b=reinterpret_cast<const float16_t*>(w.const_data_ptr<at::Half>());
  auto out=reinterpret_cast<float*>(y.mutable_data_ptr<c10::complex<float>>());
  at::parallel_for(0,(freq+3)/4,1,[&](int64_t begin,int64_t end) {
    auto scratch=at::empty({3,4,product},x.options().dtype(at::kFloat));
    auto work=scratch.mutable_data_ptr<float>();
    for (int64_t fi=begin; fi<end; ++fi) {
      const auto f0=fi*4, count=std::min<int64_t>(4,freq-f0);
      for (int part=0; part<3; ++part) for (int64_t f=0; f<count; ++f) {
        auto plane=part*freq+f0+f;
        half_plane(a+plane*tiles*k*8,b+plane*groups*k*12,
                   work+(part*4+f)*product,tiles,groups,k);
      }
      for (int64_t t=0; t<tiles; ++t) for (int64_t g=0; g<groups; ++g)
      for (int r=0; r<8 && t*8+r<rows; ++r) for (int z=0; z<12; z+=4) {
        const auto row=t*8+r, index=(t*groups+g)*96+r*12+z;
        float32x4_t real[4], imag[4];
        for (int64_t f=0; f<count; ++f) {
          auto va=vld1q_f32(work+f*product+index);
          auto vb=vld1q_f32(work+(4+f)*product+index);
          auto vc=vld1q_f32(work+(8+f)*product+index);
          real[f]=vsubq_f32(va,vb);
          imag[f]=vsubq_f32(vsubq_f32(vc,va),vb);
        }
        if (count==4) {
          transpose4(real); transpose4(imag);
          for (int j=0; j<4; ++j) {
            auto dest=(((row/blocks)*channels+g*12+z+j)*blocks+row%blocks)*freq+f0;
            float32x4x2_t value; value.val[0]=real[j]; value.val[1]=imag[j];
            vst2q_f32(out+2*dest,value);
          }
        } else for (int64_t f=0; f<count; ++f) {
          float re[4], im[4]; vst1q_f32(re,real[f]); vst1q_f32(im,imag[f]);
          for (int j=0; j<4; ++j) {
            auto dest=(((row/blocks)*channels+g*12+z+j)*blocks+row%blocks)*freq+f0+f;
            out[2*dest]=re[j]; out[2*dest+1]=im[j];
          }
        }
      }
    }
  });
  return y;
}

TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m) {
  m.def("fft_half_stream(Tensor input, Tensor weight, int batch, int blocks) -> Tensor");
}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m) { m.impl("fft_half_stream",TORCH_FN(half_stream)); }
