template Multiplier(N) {
 signal private input a;
 signal private input b;
 signal output c;
 var i;
 signal j[N];
 j[0]=a;
 for (i=0; i < N-1; i++){
   j[i+1] <== j[i]*b
 } 
 c <== j[N-1];
}

component main = Multiplier(2);
