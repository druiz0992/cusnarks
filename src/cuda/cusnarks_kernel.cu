/*
    Copyright 2018 0kims association.

    This file is part of cusnarks.

    cusnarks is a free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    cusnarks is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
    more details.

    You should have received a copy of the GNU General Public License along with
    cusnarks. If not, see <https://www.gnu.org/licenses/>.

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : cusnarks_kernel.cu
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of CUDA resources management. CUSnarks is the base class
// for all CUDA modules. Class provides functionality for GPU memory allocation
//  and deallocation, kernel launching, time measurement, random number generation
//
// ------------------------------------------------------------------

*/

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "types.h"
#include "cuda.h"
#include "log.h"
#include "rng.h"
#include "cusnarks_kernel.h"


using namespace std;


// Prime information for finitie fields. Includes 3 numbers : p. p_ and r_ that 
// follow p x p_ - r * r_ = 1 whrere r is 1^256. This is used for Montgomery reduction
//
// There are two different set of primes (MOD_N)
__constant__ mod_info_t mod_info_ct[MOD_N];


// Group
//  p1 = 21888242871839275222246405745257275088696311157297823662689037894645226208583L
//  res = 21888242871839275222246405745257275088696311157297823662689037894645226208582L
// 'Pp': 111032442853175714102588374283752698368366046808579839647964533820976443843465L,
// 'R': 115792089237316195423570985008687907853269984665640564039457584007913129639936L,
// 'R3modP': 14921786541159648185948152738563080959093619838510245177710943249661917737183L,
// 'Rbitlen': 256,
// 'Rmask': 115792089237316195423570985008687907853269984665640564039457584007913129639935L,
// 'RmodP': 6350874878119819312338956282401532409788428879151445726012394534686998597021L,
// 'Rp': 20988524275117001072002809824448087578619730785600314334253784976379291040311


// Field
//p2 = 21888242871839275222246405745257275088548364400416034343698204186575808495617L
//red = 21888242871839275222246405745257275088548364400416034343698204186575808495616L
// 'Pp': 52454480824480482120356829342366457550537710351690908576382634413609933864959L,
// 'R': 115792089237316195423570985008687907853269984665640564039457584007913129639936L,
// 'R3modP': 5866548545943845227489894872040244720403868105578784105281690076696998248512L,
// 'Rbitlen': 256,
// 'Rmask': 115792089237316195423570985008687907853269984665640564039457584007913129639935L,
// 'RmodP': 6350874878119819312338956282401532410528162663560392320966563075034087161851L,
// 'Rp': 9915499612839321149637521777990102151350674507940716049588462388200839649614L}

static uint32_t mod_info_init[] = {
        3632069959, 1008765974, 1752287885, 2541841041, 2172737629, 3092268470, 3778125865,  811880050, // p_group
        3834012553, 2278688642,  516582089, 2665381221,  406051456, 3635399632, 2441645163, 4118422199, // pp_group
        21690935,   3984885834,   41479672, 3944751749, 3074724569, 3479431631, 1508230713,  778507633, // rp_group
        3632069958, 1008765974, 1752287885, 2541841041, 2172737629, 3092268470, 3778125865,  811880050, // nonres_group

        4026531841, 1138881939, 2042196113,  674490440, 2172737629, 3092268470, 3778125865,  811880050, // p_field
        4026531839, 3269588371, 1281954227, 1703315019, 2567316369, 3818559528,  226705842, 1945644829, // pp_field
        1840322894, 3696992261, 3776048263,  151975337, 2931318109, 3357937124, 2193970460,  367786321, // rp_field
        4026531840, 1138881939, 2042196113,  674490440, 2172737629, 3092268470, 3778125865,  811880050, // nonres_field
};



// EC BN128 curve and params definition
// Y^2 = X^3 + b
// b = 3
// Generator point G =(Gx, Gy) = (1,2)
// Generator point G2 ([Gx1, Gx2], [Gy1, Gy2])
//       'G1x' : 10857046999023057135944570762232829481370756359578518086990519993285655852781L,
//       'G2x' : 8495653923123431417604973247489272438418190587263600148770280649306958101930L,
//       'G1y' : 11559732032986387107991004021392285783925812861821192530917403151452391805634L,
//       'G2y' : 4082367875863433681332203403145435568316851327593401208105741076214120093531L
//
// Assumption is that we are woking in Mongtgomery domain, so I need to transforms all these parameters
// Also, these parameters will vary depending on prime number used. 

// There are two different set of primes (MOD_N)
__constant__ ecbn128_t ecbn128_params_ct[MOD_N];


// Group prime
// b = 19052624634359457937016868847204597229365286637454337178037183604060995791063L
// Gx = 6350874878119819312338956282401532409788428879151445726012394534686998597021L
// Gy = 12701749756239638624677912564803064819576857758302891452024789069373997194042L
// G2x[0] = 11461925177900819176832270005713103520318409907105193817603008068482420711462L
// G2x[1] = 18540402224736191443939503902445128293982106376239432540843647066670759668214L
// G2y[0] = 9496696083199853777875401760424613833161720860855390556979200160215841136960L
// G2y[1] = 6170940445994484564222204938066213705353407449799250191249554538140978927342L

// Field prime
// b = 19052624634359457937016868847204597231584487990681176962899689225102261485553L
// Gx = 6350874878119819312338956282401532410528162663560392320966563075034087161851L
// Gy = 12701749756239638624677912564803064821056325327120784641933126150068174323702L
// G2x[0] = 3440318644824060289325407041038137137632482455953552081609447686580196514077L
// G2x[1] = 15555376658169732961166172612384867299105908138835914639331977638675822381717L
// G2y[0] = 1734729704421626988316384622007148076088981578411341419187802115428207738199L
// G2y[1] = 7947406416180328355183476715314321281876370016171195924215639649670494139363L 

static uint32_t ecbn128_params_init [] = {
    1353525463, 2048379561, 3780452793,  527090042, 1768673924,  860613198, 3457654158,  706701124,   // b_group
    3314486685, 3546104717, 4123462461,  175696680, 2021213740, 1718526831, 2584207151,  235567041,   // Gx_group
    2334006074, 2797242139, 3951957627,  351393361, 4042427480, 3437053662,  873447006,  471134083,   // Gy group
      45883430, 2390996433, 1232798066, 3706394933, 2541820639, 4223149639, 2945863739,  425146433,   // G2x[0] group
    2288773622, 1637743261, 4120812408, 4269789847,  589004286, 4288551522, 2929607174,  687701739,   // G2x[1] group
    2823577920, 2947838845, 1476581572, 1615060314, 1386229638,  166285564,  988445547,  352252035,   // G2y[0] group
    3340261102, 1678334806,  847068347, 3696752930,  859115638, 1442395582, 2482857090,  228892902,   // G2y[1] group

    4026531825,   96640084, 3726796669, 2767545280, 1768673930,  860613198, 3457654158,  706701124,  // b field
    1342177275, 2895524892, 2673921321,  922515093, 2021213742, 1718526831, 2584207151,  235567041,  // Gx field
    2684354550, 1496082488, 1052875347, 1845030187, 4042427484, 3437053662,  873447006,  471134083,  // Gy field
    3570833693,  985424601, 3020216734, 2567113431,  703417746, 1422701227, 3337448090,  127608510,  // G2x[0] field
    1354730133, 2060109890, 3374016652, 3251713708,  786468672, 1666612222, 3296074718,  576980987,  // G2x[1] field
    2182790487,  762510808, 2006819228, 3200553925, 2281110735, 3404365023, 3840597178,   64344700,  // G2y[0] field
     939242467, 1534311190, 2907306748, 1573550191,  646343074, 2690260169, 2616010917,  294785687   // G2y[1] field
};


// Additional constants
__constant__ misc_const_t misc_const_ct[MOD_N];

// group
// 1 => 6350874878119819312338956282401532409788428879151445726012394534686998597021L
// 2 => 12701749756239638624677912564803064819576857758302891452024789069373997194042L
// 3 => 19052624634359457937016868847204597229365286637454337178037183604060995791063L
// 4 => 3515256640640002027109419384348854550457404359307959241360540244102768179501L
// 8 => 7030513281280004054218838768697709100914808718615918482721080488205536359002L
// 12 (4b) => 10545769921920006081328258153046563651372213077923877724081620732308304538503L
// 24 (8b)=> 21091539843840012162656516306093127302744426155847755448163241464616609077006L
// field
// 1 => 6350874878119819312338956282401532410528162663560392320966563075034087161851L
// 2 => 12701749756239638624677912564803064821056325327120784641933126150068174323702L
// 3 => 19052624634359457937016868847204597231584487990681176962899689225102261485553L
// 4 =>  3515256640640002027109419384348854553564286253825534940168048113560540151787L
// 8 => 7030513281280004054218838768697709107128572507651069880336096227121080303574L
// 12 (4b) => 10545769921920006081328258153046563660692858761476604820504144340681620455361L
// 24 (8b) => 21091539843840012162656516306093127321385717522953209641008288681363240910722L

static uint32_t misc_const_init[] = {
         3314486685, 3546104717, 4123462461,  175696680, 2021213740, 1718526831, 2584207151,  235567041,    // 1 group
         2334006074, 2797242139, 3951957627,  351393361, 4042427480, 3437053662,  873447006,  471134083,    // 2 group
         1353525463, 2048379561, 3780452793,  527090042, 1768673924,  860613198, 3457654158,  706701124,   // 3_group
         1035942189,  290751008, 1856660074, 2455912978, 1617150034, 3781838855, 2263735443,  130388115,    // 4 group
         2071884378,  581502016, 3713320148,  616858660, 3234300069, 3268710414,  232503591,  260776231,   // 8 group
         3107826567,  872253024, 1275012926, 3072771639,  556482807, 2755581974, 2496239035,  391164346,    // 4b group
	 1920685838, 1744506049, 2550025852, 1850575982, 1112965615, 1216196652,  697510775,  782328693,    // 8b group 
         3314486685, 3546104717, 4123462461,  175696680, 2021213740, 1718526831, 2584207151,  235567041,    // inf group
                  0,          0,          0,          0,          0,          0,          0,          0,
         3314486685, 3546104717, 4123462461,  175696680, 2021213740, 1718526831, 2584207151,  235567041,    // 

         1342177275, 2895524892, 2673921321,  922515093, 2021213742, 1718526831, 2584207151,  235567041,   // 1 field
         2684354550, 1496082488, 1052875347, 1845030187, 4042427484, 3437053662,  873447006,  471134083,   // 2 field
         4026531825,   96640084, 3726796669, 2767545280, 1768673930,  860613198, 3457654158,  706701124,  // 3 field
         1342177259, 1853283037,   63554581, 3015569934, 1617150043, 3781838855, 2263735443,  130388115,    // 4 field
         2684354518, 3706566074,  127109162, 1736172572, 3234300087, 3268710414,  232503591,  260776231,  // 8 field
         4026531777, 1264881815,  190663744,  456775210,  556482835, 2755581974, 2496239035,  391164346,    // 4b field
	 3758096258, 2529763631,  381327488,  913550420, 1112965670, 1216196652,  697510775,  782328693,     // 8b field
         1342177275, 2895524892, 2673921321,  922515093, 2021213742, 1718526831, 2584207151,  235567041,   // inf
                  0,          0,          0,          0,          0,          0,          0,          0,    
         1342177275, 2895524892, 2673921321,  922515093, 2021213742, 1718526831, 2584207151,  235567041,   // 
};

// 32 roots of unitity of field prime (only first 16)

//0:  6350874878119819312338956282401532410528162663560392320966563075034087161851L,
//1:  9269790970655856939060861028170300436291862458449419110576697760057602964270L,
//2:  19842606027486572848039611947872621678304286801077156294824909905662646061134L,
//3:  13759120526037196500129128819789394413689414465328022089822484987151763408268L,
//4:  5491040059702810429711982390307830649966987922105626750643517491307525342146L,
//5:  19679110521383505085478399228223544330787991294964066996443436559375109934180L,
//6:  16978050423172483659693846109059477509086323706103642533077561519595429120499L,
//7:  9746772031407045348032214952967177820231224727604262011979369064780488775588L,
//8:  19547479427403304814847004989093697233993870177734472828559976221244791779211L,
//9:  8129309600626375419283237767199111367911583567608984561693754702616141988553L,
//10: 10170256882410828375274983558986772540753495154277897142144890932984137355011L,
//11: 151965293740925585589094366864719293687797514907895125087738304882507001933L,
//12: 8440431145628724787403704960401007518256413004629773987513885884223374663323L,
//13: 17425778421222095042759364376586862489004360793394175340784999844505696826413L,
//14: 1884313618740866241758905828412908890553358798399578056779203882743340049521L,
//15: 9841396785537381856564969674971451141489211418471679732638463192434711953212L,

__constant__ uint32_t W32_ct[NWORDS_256BIT * 16];

static uint32_t W32_roots[] = {
   1342177275, 2895524892, 2673921321,  922515093, 2021213742, 1718526831, 2584207151,  235567041,   // 0
   2292284206, 1436927493, 1863072836, 2310988525, 3284102420, 2592528728, 3605342721,  343835656,   // 1
   2999093326, 2160059935, 3700525642, 2935736508,  729567453, 2513295254,  980132873,  736003163,   // 2
   3962865036,  849939859,  659744097, 3665818920,  887683753, 1314972491, 3574560323,  510354144,   // 3
    978486210, 1261742769, 4126010066, 4208140616,  602746469, 3738781835, 2711012991,  203673995,   // 4
   3210169444,  571799473, 4147215858,  687416111, 2964157901, 3849684136, 3692522755,  729938777,   // 5
   1244482035,  330452492, 3296279002, 2808567544, 2940955937, 1760199938, 2418605771,  629750890,   // 6
   4268758948,  417762385, 4236814568, 2248217265, 3374594792,  687869391, 2931020772,  361527867,   // 7
   2665279371, 2138389911, 2974257440, 1597970220, 1946774638, 2377212537,  968952015,  725056309,   // 8
   2022146761, 2150658060, 2105772189, 2218376580, 1786403205, 423080585,  3070285582,  301532851,   // 9
   1949273859, 4105600380, 1674602700, 1746611676,  469456746, 2363866351, 3535158526,  377235793,   // 10
   4024447053,  463933058, 2063891815,  575215092, 2272269203, 3968666890,   95986586,    5636706,   // 11
   3718422171,  839371741,  640663292, 2009740710, 2958679608, 1031669727,  851251543,  313072991,   // 12
   2121233453, 4279475682, 4226397677, 1393044800, 1535131808, 3429418656,   22039117,  646358045,   // 13
    115975281, 3281997936, 4285124589, 2634917807, 2727787619, 1854498425, 2768436868,   69893076,   // 14
   3512696636, 3662204595, 2047957525, 3694366805, 1895475809, 3092782664, 3791196599,  365037694    // 15
}; 

// 32 inverse roots of unitity of field prime (only first 16)

//0:  6350874878119819312338956282401532410528162663560392320966563075034087161851L,
//1:  12046846086301893365681436070285823947059152981944354611059740994141096542405L,
//2:  20003929253098408980487499916844366197995005602016456286919000303832468446096L,
//3:  4462464450617180179487041368670412599544003607021859002913204342070111669204L,
//4:  13447811726210550434842700784856267570291951395786260356184318302352433832294L,
//5:  21736277578098349636657311378392555794860566885508139218610465881693301493684L,
//6:  11717985989428446846971422186270502547794869246138137201553313253591671140606L,
//7:  13758933271212899802963167978058163720636780832807049782004449483959666507064L,
//8:  2340763444435970407399400756163577854554494222681561515138227965331016716406L,
//9:  12141470840432229874214190792290097268317139672811772331718835121795319720029L,
//10:  4910192448666791562552559636197797579462040694312391810620642666980379375118L,
//11:  2209132350455770136768006517033730757760373105451967347254767627200698561437L,
//12:  16397202812136464792534423354949444438581376478310407593054686695268283153471L,
//13:  8129122345802078722117276925467880674858949935088012253875719199424045087349L,
//14:  2045636844352702374206793797384653410244077599338878048873294280913162434483L,
//15:  12618451901183418283185544717086974652256501941966615233121506426518205531347L,

__constant__ uint32_t IW32_ct[NWORDS_256BIT * 16];

static uint32_t IW32_roots[] = {
        1342177275, 2895524892, 2673921321,  922515093, 2021213742, 1718526831, 2584207151,  235567041,
         513835205, 1771644640, 4289205883, 1275090930,  277261819, 4294453102, 4281896561,  446842355,
        3910556560, 2151851299, 2052038819, 2334539928, 3739917305, 1237770044, 1009688997,  741986974,
        1905298388, 1154373553, 2110765731, 3576412935,  637605820, 3957817110, 3756086747,  165522005,
         308109670,  299510198, 1401532821, 2959717026, 3509025316, 2060598742, 2926874322,  498807059,
           2084788,  674948881, 4273271594,   99275347, 4195435722, 3418568875, 3682139278,  806243344,
        2077257982, 1328248855,  367593412, 3222846060, 1703280882,  728402119,  242967339,  434644257,
        2004385080, 3283191175, 4231391219, 2751081155,  386334423, 2669187885,  707840283,  510347199,
        1361252470, 3295459324, 3362905968, 3371487515,  225962990, 715055933,  2809173850,   86823741,
        4052740189,  721119553, 2100348841, 2721240470, 3093110132, 2404399078,  847105093,  450352183,
        2782049806,  808429447, 3040884407, 2160890191, 3526748987, 1332068531, 1359520094,  182129160,
         816362397,  567082466, 2189947551, 4282041624, 3503547023, 3537551629,   85603109,   81941273,
        3048045631, 4172106466, 2211153342,  761317119, 1569991159, 3648453931, 1067112873,  608206055,
          63666805,  288942080, 1382452016, 1303638816, 1285053875, 1777295979,  203565542,  301525906,
        1027438515, 3273789300, 2636637766, 2033721227, 1443170175, 578973216,  2797992992,   75876887,
        1734247635, 3996921742,  179123276, 2658469211, 3183602504, 499739741,   172783144,  468044394
};

// During IFFT, I need to scale by inv(32). Below is the representation of 32 in Mongtgomery
//32 : 3618502788666131106986593281521497120414687020801267626233049500247285301248L => in 256 is (1 << 251 )
__constant__ uint32_t IW32_nroots_ct[NWORDS_256BIT];

static uint32_t IW32_nroots[] = { 0,0,0,0,0,0,0,0x8000000};

/*
    Constructor : Reserves global (vector) and constant (prime info) memory 

    Arguments :
      in_length : Maximum number of elements in Kernel input data
      in_size   : Maximum size of Kernel input data (in Bytes)
      out_length: Maximum number of elements in Kernel output  data
      out_size  : Mamimum size of Kernel output data (in Bytes)
      kcb       : Pointer to kernel functions (indexed by XXX_callback_t enum)
      
*/
CUSnarks::CUSnarks (uint32_t in_len, uint32_t in_size, 
                    uint32_t out_len, uint32_t out_size, kernel_cb *kcb) : 
                          kernel_callbacks(kcb)
{
  CUSnarks(in_len, in_size, out_len, out_size, 0);
}

CUSnarks::CUSnarks (uint32_t in_len, uint32_t in_size,
                    uint32_t out_len, uint32_t out_size,
                    kernel_cb *kcb,uint32_t seed) : 
                       kernel_callbacks(kcb)
{
  in_vector_device.data = NULL;
  in_vector_device.length = in_len;
  in_vector_device.size = in_size;
  out_vector_device.data = NULL;
  out_vector_device.length = out_len;
  out_vector_device.size = out_size;

  allocateCudaResources(in_size, out_size);
  initRNG(seed);
}

/*
   Reserve GPU memory for input and output vectors and input kernel params (global memory),
   as well as some constant info (constant memory)
 */
void CUSnarks::allocateCudaResources(uint32_t in_size, uint32_t out_size)
{
  mod_info_t mod_h[MOD_N];
  ecbn128_t ecbn128_h[MOD_N];
  misc_const_t misc_h[MOD_N]; 

  // Allocate kernel input and putput data vectors in global memory 
  CCHECK(cudaMalloc((void**) &this->in_vector_device.data, in_size));
  CCHECK(cudaMalloc((void**) &this->out_vector_device.data, out_size));

  // Allocate kernel params in global memory 
  CCHECK(cudaMalloc((void**) &this->params_device, sizeof(kernel_params_t)));

  // constants ->  Initialize data and copy to constant memory
  memcpy(mod_h,     mod_info_init,           sizeof(mod_info_t)    * MOD_N); // prime info
  memcpy(ecbn128_h, ecbn128_params_init,     sizeof(ecbn128_t)     * MOD_N); // ecbn128
  memcpy(misc_h,    misc_const_init,         sizeof(misc_const_t)  * MOD_N); // misc

  // Copy modulo info to device constant
  CCHECK(cudaMemcpyToSymbol(mod_info_ct,       mod_h,     MOD_N * sizeof(mod_info_t)));  // prime info
  CCHECK(cudaMemcpyToSymbol(ecbn128_params_ct, ecbn128_h, MOD_N * sizeof(ecbn128_t)));   // ecbn128
  CCHECK(cudaMemcpyToSymbol(misc_const_ct,    misc_h,    MOD_N * sizeof(misc_const_t)));// misc
  CCHECK(cudaMemcpyToSymbol(W32_ct,           W32_roots, sizeof(uint32_t) * NWORDS_256BIT * 16));// W32roots
  CCHECK(cudaMemcpyToSymbol(IW32_ct,          IW32_roots, sizeof(uint32_t) * NWORDS_256BIT * 16));// IW32roots
  CCHECK(cudaMemcpyToSymbol(IW32_nroots_ct,   IW32_nroots, sizeof(uint32_t) * NWORDS_256BIT ));// inverse 32
}

/*
   Initialize PCG random number generator  
   http://www.pcg-random.org/

   IF seed is 0, random seed is taken from urand.

   NOTE : when seed is 0, generator breaks
   TODO :  Fix seed = 0
*/
void CUSnarks::initRNG(uint32_t seed)
{
  if (seed == 0){ rng =  _RNG::get_instance(); }
  else { rng = _RNG::get_instance(seed); }
}
/*
   Generate N 32 bit random samples
*/
void CUSnarks::rand(uint32_t *samples, uint32_t n_samples)
{
    uint32_t size_sample = in_vector_device.size / (in_vector_device.length * sizeof(uint32_t));
    rng->randu32(samples, n_samples * size_sample);
}

/*
   Free memory allocated in GPU:
*/
CUSnarks::~CUSnarks()
{
  cudaFree(in_vector_device.data);
  cudaFree(out_vector_device.data);
  cudaFree(params_device);
}

/*
   Kernel launcher. This function is an attempt to hide the complexity of launching a kernel. When called,
   the input vector is copied to global GPU memory, the kernel is launched, and when finished, kernel
   output vector data is copied back from GPU to host.

   Arguments:
    kernel_idx : kernel number to be launched. Defined by XXX_callback_t enum types
    out_vector_host : kernel ouput data vector (Host size)
    in_vector_host  : kernel input data vector (host size)
    config          : kernel configuration info (grid, block, smem,...)
    params          : Kernel input parameters

*/
double CUSnarks::kernelLaunch(
                uint32_t kernel_idx,
		vector_t *out_vector_host,
	       	vector_t *in_vector_host,
                kernel_config_t *config,
                kernel_params_t *params)
{
  // check input lengths do not exceed reserved amount
  if (in_vector_host->length > in_vector_device.length) { return 0.0; }
  if (out_vector_host->length > out_vector_device.length) { return 0.0; }

  in_vector_host->size = in_vector_host->length * (in_vector_device.size / in_vector_device.length  );
  out_vector_host->size = out_vector_host->length * (out_vector_device.size / out_vector_device.length );

  double start, end_copy_in, end_kernel, end_copy_out;
  int blockD, gridD;

  // measure data xfer time Host -> Device
  start = elapsedTime();
  CCHECK(cudaMemcpy(in_vector_device.data, in_vector_host->data, in_vector_host->size, cudaMemcpyHostToDevice));
  CCHECK(cudaMemcpy(params_device, params, sizeof(kernel_params_t), cudaMemcpyHostToDevice));
  end_copy_in = elapsedTime() - start;
 
  // configure kernel. Input parameter invludes block size. Grid is calculated 
  // depending on input data length and stride (how many samples of input data are 
  // used per thread
  blockD = config->blockD;
  if (config->gridD == 0){
     config->gridD = (blockD + in_vector_host->length/params->stride - 1) / blockD;
  }
  gridD = config->gridD;


  // launch kernel
  start = elapsedTime();
  kernel_callbacks[kernel_idx]<<<gridD, blockD, config->smemS>>>(out_vector_device.data, in_vector_device.data, params_device);
  CCHECK(cudaGetLastError());
  CCHECK(cudaDeviceSynchronize());
  end_kernel = elapsedTime() - start;

  // retrieve kernel output data from GPU to host
  start = elapsedTime();
  CCHECK(cudaMemcpy(out_vector_host->data, out_vector_device.data, out_vector_host->size, cudaMemcpyDeviceToHost));
  end_copy_out = elapsedTime() - start;

  logInfo("----- Info -------\n");
  logInfo("IVHS : %d, IVHL : %d, IVDS : %d, IVDL : %d\n",in_vector_host->size, 
		                                        in_vector_host->length,
						       	in_vector_device.size,
						       	in_vector_device.length);

  logInfo("OVHS : %d, OVHL : %d, OVDS : %d, OVDL : %d\n",out_vector_host->size,
		                                        out_vector_host->length, 
							out_vector_device.size,
						       	out_vector_device.length);

  logInfo("Params : premod : %d, midx : %d, In Length : %d, Out Length : %d, Stride : %d\n",params->premod, params->midx, params->in_length, params->out_length, params->stride);
  logInfo("Kernel IDX :%d <<<%d, %d, %d>>> Time Elapsed Kernel : %f.sec\n", 
          kernel_idx, gridD, blockD, config->smemS,end_kernel);
  logInfo("Time Elapsed Xfering in %d bytes : %f sec\n",
          in_vector_host->size, end_copy_in);
  logInfo("Time Elapsed Xfering out %d bytes : %f sec\n",
          out_vector_host->size, end_copy_out);

  return end_kernel;
}

/*
  Professional CUDA C Programming by John Cheng, Max Grossman, Ty McKercher
*/
double CUSnarks::elapsedTime(void)
{
  struct timeval tp;
  gettimeofday (&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

/*
  Professional CUDA C Programming by John Cheng, Max Grossman, Ty McKercher
*/
void CUSnarks::getDeviceInfo(void)
{
   int deviceCount = 0;
   CCHECK(cudaGetDeviceCount(&deviceCount));
  
   if (deviceCount == 0) {
      logInfo("There are no available device(s) that support CUDA\n");
   } else {
      logInfo("Detected %d CUDA Capable device(s)\n", deviceCount);
   }
   int dev, driverVersion = 0, runtimeVersion = 0;

   dev =0;
   cudaSetDevice(dev);
   cudaDeviceProp deviceProp;
   cudaGetDeviceProperties(&deviceProp, dev);
   logInfo("Device %d: \"%s\"\n", dev, deviceProp.name);

   cudaDriverGetVersion(&driverVersion);
   cudaRuntimeGetVersion(&runtimeVersion);
   logInfo(" CUDA Driver Version / Runtime Version                     %d.%d / %d.%d\n",
          driverVersion/1000, (driverVersion%100)/10,
          runtimeVersion/1000, (runtimeVersion%100)/10);
   logInfo(" CUDA Capability Major/Minor version number:               %d.%d\n",
         deviceProp.major, deviceProp.minor);
   logInfo(" Total amount of global memory:                            %.2f MBytes (%llu bytes)\n",
         (float)deviceProp.totalGlobalMem/(pow(1024.0,3)),
         (unsigned long long) deviceProp.totalGlobalMem);
   logInfo(" GPU Clock rate:                                           %.0f MHz (%0.2f GHz)\n",
         deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
   logInfo(" Memory Clock rate:                                        %.0f Mhz\n",
         deviceProp.memoryClockRate * 1e-3f);
   logInfo(" Memory Bus Width:                                         %d-bit\n",
         deviceProp.memoryBusWidth);
   if (deviceProp.l2CacheSize) {
         logInfo(" L2 Cache Size:                                      %d bytes\n",
           deviceProp.l2CacheSize);
   }
   logInfo(" Max Texture Dimension Size (x,y,z)             "
     "    1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
            deviceProp.maxTexture1D , deviceProp.maxTexture2D[0],
            deviceProp.maxTexture2D[1], 
            deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
            deviceProp.maxTexture3D[2]);
   logInfo(" Max Layered Texture Size (dim) x layers         "
     "   1D=(%d) x %d, 2D=(%d,%d) x %d\n",
       deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
       deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
       deviceProp.maxTexture2DLayered[2]);

   logInfo(" Total amount of constant memory:                         %lu bytes\n",
      deviceProp.totalConstMem);
   logInfo(" Total amount of shared memory per block:                 %lu bytes\n",
      deviceProp.sharedMemPerBlock);
   logInfo(" Total number of registers available per block:           %d\n",
      deviceProp.regsPerBlock);
   logInfo(" Warp size:                                               %d\n", deviceProp.warpSize);
   logInfo(" Maximum number of threads per multiprocessor:            %d\n",
      deviceProp.maxThreadsPerMultiProcessor);
   logInfo(" Maximum number of threads per block:                     %d\n",
      deviceProp.maxThreadsPerBlock);
   logInfo(" Maximum sizes of each dimension of a block:              %d x %d x %d\n",
      deviceProp.maxThreadsDim[0],
      deviceProp.maxThreadsDim[1],
      deviceProp.maxThreadsDim[2]);
   logInfo(" Maximum sizes of each dimension of a grid:               %d x %d x %d\n",
      deviceProp.maxGridSize[0],
      deviceProp.maxGridSize[1],
      deviceProp.maxGridSize[2]);
   logInfo(" Maximum memory pitch:                                    %lu bytes\n", 
      deviceProp.memPitch);
}
