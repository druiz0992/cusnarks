import numpy as np

def _32_to_64(A):
    _A = []
    for idx in range(int(len(A)/2)):
        _A.append((A[2*idx+1]<<32) + A[2*idx])

    return _A

def montmult(A,B, t=32):
    NPrime = [4026531839, 3269588371, 1281954227, 1703315019, 2567316369, 3818559528,  226705842, 1945644829]
    N = [4026531841, 1138881939, 2042196113,  674490440, 2172737629, 3092268470, 3778125865,  811880050]
    T = [0,0,0,0,0,0,0,0,0,0,0,0]

    if t==32:
        mask = 0xffffffff
        sh   = 32
        NWORDS = 8
        _NPrime = NPrime.copy()
        _N = N.copy()
        _A = A.copy()
        _B = B.copy()
        _T = T.copy()
    else:
        mask = 0xffffffffffffffff
        sh = 64
        NWORDS = 4
        _NPrime = _32_to_64(NPrime)
        _N = _32_to_64(N)
        _A = _32_to_64(A)
        _B = _32_to_64(B)
        _T = _32_to_64(T)

    print([hex(t) for t in _A[:NWORDS]])       
    print([hex(t) for t in _B[:NWORDS]])       

    for i in range(NWORDS):
        X = _A[0] * _B[i] + _T[0]
        S = X & mask
        C = (X >> sh) & mask
        print("1["+str(i)+"]: C: " + str(hex(C)) + " S: " + str(hex(S)))

        X = _T[1] + C
        _T[1] = X & mask
        carry = (X >> sh) & mask
        print("a["+str(i)+"]: C: " + str(hex(carry)) + " T[1]: " + str(hex(_T[1])))

        X = S * _NPrime[0]
        M = X & mask
        print("b["+str(i)+"]: M: " + str(hex(M)) + " N: "+str(hex(_N[0])) )

        X = M * _N[0] + S
        S = X & mask
        C = (X >> sh) & mask
        print("2["+str(i)+"]: C: " + str(hex(C)) + " S: " + str(hex(S)) + " carry : "+str(hex(carry)))

        for j in range(1,NWORDS):
           X = _T[j] + _A[j]*_B[i] + C
           S = X & mask
           X = (X >> sh)& mask
           C = X & mask
           C3 = (X >> sh) & mask
           print("3["+str(i)+"-"+str(j)+"]: C1: " + str(hex(C3)) + " C: " +str(hex(C))+" S: " + str(hex(S)))
           
           X = C +carry
           C = X & mask
           C3 = ((X >> sh) & mask) + C3

           print("c["+str(i)+"-"+str(j)+"]: C1: " + str(hex(C3)) +" C: "+str(hex(C))+ " T[j+1]: " + str(hex(_T[j+1])))
           X = _T[j+1] + C
           _T[j+1] = X & mask
           carry = ((X >> sh) & mask) + C3
           print("4["+str(i)+"-"+str(j)+"]: C1: " + str(hex(C3)) +" C: "+str(hex(C))+ " S: " + str(hex(_T[j+1])) + " carry : "+str(hex(carry)))

           X = M * _N[j] + S
           S = X & mask
           C = (X >> sh) & mask

           _T[j-1] = S
           print([hex(t) for t in _T[:NWORDS]])       


        X = _T[NWORDS] + C   
        S = X & mask
        C = (X >> sh) & mask

        _T[NWORDS-1] = S

        _T[NWORDS] = T[NWORDS+1] + C
        _T[NWORDS+1] = 0
        print([hex(t) for t in _T[:NWORDS]])       

    print([hex(t) for t in _T[:NWORDS]])       
    return _T[:NWORDS]

        
if __name__ == '__main__':

    A = [int(0x6b8b4567),
                    int(0x327b23c6),
                    int(0x643c9869),
                    int(0x66334873),
                    int(0x74b0dc51),
                    int(0x19495cff),
                    int(0x2ae8944a),
                    int(0x5558ec)]

    B = [int(0x238e1f29),
                    int(0x46e87ccd),
                    int(0x3d1b58ba),
                    int(0x507ed7ab),
                    int(0x2eb141f2),
                    int(0x41b71efb),
                    int(0x79e2a9e3),
                    int(0x45e146)]
    A = [1256258315, 601837617 ,1069080936 ,2391979847 ,2958730709, 3679080337, 1518004222 ,102069457 ]
    B = [ 548181926, 1225716380 ,468126608 ,4005204904 ,2388513804, 583156214 ,67710866, 156399213 ]
    #C1 = montmult(A,B)
    C2 = montmult(A,B,t=64)
