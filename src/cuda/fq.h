#ifndef _FQ_H_
#define _FQ_H_

extern "C" void Fq_rawAdd(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fq_rawSub(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fq_rawMMul(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fq_rawMSquare(uint32_t *r, const uint32_t *x);
extern "C" void Fq_fromMont(uint32_t *z, const uint32_t *x);
extern "C" void Fq_toMont(uint32_t *z, const uint32_t *x);

#endif
