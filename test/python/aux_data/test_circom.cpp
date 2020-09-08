#include "circom.hpp"
#include "calcwit.hpp"
#define NSignals 2041
#define NComponents 1
#define NOutputs 1
#define NInputs 1
#define NVars 2039
#define __P__ "21888242871839275222246405745257275088548364400416034343698204186575808495617"

/*
A
n=2038
*/
void A_dc1fe12a3b41e0da(Circom_CalcWit *ctx, int __cIdx) {
    FrElement _sigValue[1];
    FrElement _sigValue_1[1];
    FrElement _sigValue_2[1];
    FrElement _tmp_3[1];
    FrElement _tmp_4[1];
    FrElement _tmp_6[1];
    FrElement i[1];
    FrElement _tmp_7[1];
    FrElement _sigValue_3[1];
    FrElement _tmp_8[1];
    FrElement _sigValue_4[1];
    FrElement _tmp_9[1];
    FrElement _tmp_10[1];
    FrElement _tmp_11[1];
    FrElement _tmp_12[1];
    FrElement _sigValue_5[1];
    int _in_sigIdx_;
    int _intermediate_sigIdx_;
    int _offset;
    int _offset_5;
    int _offset_7;
    int _offset_10;
    int _offset_16;
    int _offset_18;
    int _offset_21;
    int _offset_27;
    int _out_sigIdx_;
    Circom_Sizes _sigSizes_intermediate;
    PFrElement _loopCond;
    Fr_copy(&(_tmp_6[0]), ctx->circuit->constants +1);
    Fr_copy(&(i[0]), ctx->circuit->constants +2);
    _in_sigIdx_ = ctx->getSignalOffset(__cIdx, 0x08b73807b55c4bbeLL /* in */);
    _intermediate_sigIdx_ = ctx->getSignalOffset(__cIdx, 0x9cec2b21f5b61d84LL /* intermediate */);
    _out_sigIdx_ = ctx->getSignalOffset(__cIdx, 0x19f79b1921bbcfffLL /* out */);
    _sigSizes_intermediate = ctx->getSignalSizes(__cIdx, 0x9cec2b21f5b61d84LL /* intermediate */);
    /* signal input in */
    /* signal output out */
    /* signal intermediate[n] */
    /* intermediate[0] <== in */
    ctx->getSignal(__cIdx, __cIdx, _in_sigIdx_, _sigValue);
    _offset = _intermediate_sigIdx_;
    ctx->setSignal(__cIdx, __cIdx, _offset, _sigValue);
    /* for (var i=1;i<n;i++) */
    /* intermediate[i] <== intermediate[i-1] * intermediate[i-1] + i */
    _offset_5 = _intermediate_sigIdx_;
    ctx->getSignal(__cIdx, __cIdx, _offset_5, _sigValue_1);
    _offset_7 = _intermediate_sigIdx_;
    ctx->getSignal(__cIdx, __cIdx, _offset_7, _sigValue_2);
    Fr_mul(_tmp_3, _sigValue_1, _sigValue_2);
    Fr_add(_tmp_4, _tmp_3, (ctx->circuit->constants + 1));
    _offset_10 = _intermediate_sigIdx_ + 1*_sigSizes_intermediate[1];
    ctx->setSignal(__cIdx, __cIdx, _offset_10, _tmp_4);
    _loopCond = _tmp_6;
    while (Fr_isTrue(_loopCond)) {
        /* intermediate[i] <== intermediate[i-1] * intermediate[i-1] + i */
        Fr_sub(_tmp_7, i, (ctx->circuit->constants + 1));
        _offset_16 = _intermediate_sigIdx_ + Fr_toInt(_tmp_7)*_sigSizes_intermediate[1];
        ctx->getSignal(__cIdx, __cIdx, _offset_16, _sigValue_3);
        Fr_sub(_tmp_8, i, (ctx->circuit->constants + 1));
        _offset_18 = _intermediate_sigIdx_ + Fr_toInt(_tmp_8)*_sigSizes_intermediate[1];
        ctx->getSignal(__cIdx, __cIdx, _offset_18, _sigValue_4);
        Fr_mul(_tmp_9, _sigValue_3, _sigValue_4);
        Fr_add(_tmp_10, _tmp_9, i);
        _offset_21 = _intermediate_sigIdx_ + Fr_toInt(i)*_sigSizes_intermediate[1];
        ctx->setSignal(__cIdx, __cIdx, _offset_21, _tmp_10);
        Fr_add(_tmp_11, i, (ctx->circuit->constants + 1));
        Fr_copyn(i, _tmp_11, 1);
        Fr_lt(_tmp_12, i, (ctx->circuit->constants + 3));
        _loopCond = _tmp_12;
    }
    /* out <== intermediate[n-1] */
    _offset_27 = _intermediate_sigIdx_ + 2037*_sigSizes_intermediate[1];
    ctx->getSignal(__cIdx, __cIdx, _offset_27, _sigValue_5);
    ctx->setSignal(__cIdx, __cIdx, _out_sigIdx_, _sigValue_5);
    ctx->finished(__cIdx);
}
// Function Table
Circom_ComponentFunction _functionTable[1] = {
     A_dc1fe12a3b41e0da
};
