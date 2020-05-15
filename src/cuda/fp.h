#ifndef _FP_H_
#define _FP_H_


extern "C" void Fp_rawAdd(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fp_rawSub(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fp_rawMMul(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fp_rawMSquare(uint32_t *r, const uint32_t *x);
extern "C" void Fp_fromMont(uint32_t *z, const uint32_t *x);
extern "C" void Fp_toMont(uint32_t *z, const uint32_t *x);

#endif
