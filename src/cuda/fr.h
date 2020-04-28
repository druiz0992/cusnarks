#ifndef _FR_H_
#define _FR_H_


extern "C" void Fr_rawAdd(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fr_rawSub(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fr_rawMMul(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fr_rawMSquare(uint32_t *r, const uint32_t *x);
extern "C" void Fr_fromMont(uint32_t *z, const uint32_t *x);
extern "C" void Fr_toMont(uint32_t *z, const uint32_t *x);

#endif
