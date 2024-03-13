/*
 * File: Quadrotor.c
 *
 * Code generated for Simulink model 'Quadrotor'.
 *
 * Model version                  : 1.42
 * Simulink Coder version         : 9.7 (R2022a) 13-Nov-2021
 * C/C++ source code generated on : Fri Apr 14 14:37:36 2023
 *
 * Target selection: ert.tlc
 * Embedded hardware selection: STMicroelectronics->ST10/Super10
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#include "Quadrotor.h"
#include <math.h>
#include "rtwtypes.h"
#include <stddef.h>
#define NumBitsPerChar                 8U

/* Private macros used by the generated code to access rtModel */
#ifndef rtmIsMajorTimeStep
#define rtmIsMajorTimeStep(rtm)        (((rtm)->Timing.simTimeStep) == MAJOR_TIME_STEP)
#endif

#ifndef rtmIsMinorTimeStep
#define rtmIsMinorTimeStep(rtm)        (((rtm)->Timing.simTimeStep) == MINOR_TIME_STEP)
#endif

#ifndef rtmSetTPtr
#define rtmSetTPtr(rtm, val)           ((rtm)->Timing.t = (val))
#endif

/* Exported block signals */
real_T n1;                             /* '<Root>/n1' */
real_T n2;                             /* '<Root>/n2' */
real_T n3;                             /* '<Root>/n3' */
real_T n4;                             /* '<Root>/n4' */
real_T L1;                             /* '<Root>/L1' */
real_T L2;                             /* '<Root>/L2' */
real_T L3;                             /* '<Root>/L3' */
real_T L4;                             /* '<Root>/L4' */
real_T mass;
real_T Ixx;                            /* '<Root>/Ixx' */
real_T Iyy;                            /* '<Root>/Iyy' */
real_T Izz;                            /* '<Root>/Izz' */
real_T x;                              /* '<S6>/Integrator1' */
real_T y;                              /* '<S6>/Integrator3' */
real_T z;                              /* '<S6>/Integrator5' */
real_T dx;                             /* '<S6>/Integrator' */
real_T dy;                             /* '<S6>/Integrator2' */
real_T dz;                             /* '<S6>/Integrator4' */
real_T theta;                          /* '<S9>/Integrator1' */
real_T dtheta;                         /* '<S9>/Integrator' */
real_T Psi;                            /* '<S8>/Integrator1' */
real_T dPsi;                           /* '<S8>/Integrator' */
real_T ddPhi_n;                        /* '<S7>/Step' */
real_T Phi;                            /* '<S7>/Integrator1' */
real_T dPhi;                           /* '<S7>/Integrator' */
real_T ddtheta_n;                      /* '<S9>/Step' */
real_T ddx_n;                          /* '<S6>/Step' */
real_T ddy_n;                          /* '<S6>/Step1' */
real_T ddz_n;                          /* '<S6>/Step2' */
real_T ddPsi_n;                        /* '<S8>/Step' */

/*add*/
float x_o;
float y_o;
float z_o;
float Phi_o;
float theta_o;
float Psi_o;
float dx_o;
float dy_o;
float dz_o;
float dPhi_o;
float dtheta_o;
float dPsi_o;

/* Exported block parameters */
real_T Phi_0 = 0.0;                    /* Variable: Phi_0
                                        * Referenced by: '<S7>/Integrator1'
                                        */
real_T Psi_0 = 0.0;                    /* Variable: Psi_0
                                        * Referenced by: '<S8>/Integrator1'
                                        */
real_T dPhi_0 = 0.0;                   /* Variable: dPhi_0
                                        * Referenced by: '<S7>/Integrator'
                                        */
real_T dPsi_0 = 0.0;                   /* Variable: dPsi_0
                                        * Referenced by: '<S8>/Integrator'
                                        */
real_T dtheta_0 = 0.0;                 /* Variable: dtheta_0
                                        * Referenced by: '<S9>/Integrator'
                                        */
real_T dx_0 = 0.0;                     /* Variable: dx_0
                                        * Referenced by: '<S6>/Integrator'
                                        */
real_T dy_0 = 0.0;                     /* Variable: dy_0
                                        * Referenced by: '<S6>/Integrator2'
                                        */
real_T dz_0 = 0.0;                     /* Variable: dz_0
                                        * Referenced by: '<S6>/Integrator4'
                                        */
real_T theta_0 = 0.0;                  /* Variable: theta_0
                                        * Referenced by: '<S9>/Integrator1'
                                        */
real_T x_0 = 0.0;                      /* Variable: x_0
                                        * Referenced by: '<S6>/Integrator1'
                                        */
real_T y_0 = 0.0;                      /* Variable: y_0
                                        * Referenced by: '<S6>/Integrator3'
                                        */
real_T z_0 = 0.0;                      /* Variable: z_0
                                        * Referenced by: '<S6>/Integrator5'
                                        */

/* Block signals (default storage) */
B_Quadrotor_T Quadrotor_B;

/* Continuous states */
X_Quadrotor_T Quadrotor_X;

/* External outputs (root outports fed by signals with default storage) */
ExtY_Quadrotor_T Quadrotor_Y;

/* Real-time model */
static RT_MODEL_Quadrotor_T Quadrotor_M_;
RT_MODEL_Quadrotor_T *const Quadrotor_M = &Quadrotor_M_;

/* private model entry point functions */
extern void Quadrotor_derivatives(void);
extern real_T rtInf;
extern real_T rtMinusInf;
extern real_T rtNaN;
extern real32_T rtInfF;
extern real32_T rtMinusInfF;
extern real32_T rtNaNF;
static void rt_InitInfAndNaN(size_t realSize);
static boolean_T rtIsInf(real_T value);
static boolean_T rtIsInfF(real32_T value);
static boolean_T rtIsNaN(real_T value);
static boolean_T rtIsNaNF(real32_T value);
typedef struct {
  struct {
    uint32_T wordH;
    uint32_T wordL;
  } words;
} BigEndianIEEEDouble;

typedef struct {
  struct {
    uint32_T wordL;
    uint32_T wordH;
  } words;
} LittleEndianIEEEDouble;

typedef struct {
  union {
    real32_T wordLreal;
    uint32_T wordLuint;
  } wordL;
} IEEESingle;

real_T rtInf;
real_T rtMinusInf;
real_T rtNaN;
real32_T rtInfF;
real32_T rtMinusInfF;
real32_T rtNaNF;
static real_T rtGetInf(void);
static real32_T rtGetInfF(void);
static real_T rtGetMinusInf(void);
static real32_T rtGetMinusInfF(void);
static real_T rtGetNaN(void);
static real32_T rtGetNaNF(void);

/*
 * Initialize the rtInf, rtMinusInf, and rtNaN needed by the
 * generated code. NaN is initialized as non-signaling. Assumes IEEE.
 */
static void rt_InitInfAndNaN(size_t realSize)
{
  (void) (realSize);
  rtNaN = rtGetNaN();
  rtNaNF = rtGetNaNF();
  rtInf = rtGetInf();
  rtInfF = rtGetInfF();
  rtMinusInf = rtGetMinusInf();
  rtMinusInfF = rtGetMinusInfF();
}

/* Test if value is infinite */
static boolean_T rtIsInf(real_T value)
{
  return (boolean_T)((value==rtInf || value==rtMinusInf) ? 1U : 0U);
}

/* Test if single-precision value is infinite */
static boolean_T rtIsInfF(real32_T value)
{
  return (boolean_T)(((value)==rtInfF || (value)==rtMinusInfF) ? 1U : 0U);
}

/* Test if value is not a number */
static boolean_T rtIsNaN(real_T value)
{
  boolean_T result = (boolean_T) 0;
  size_t bitsPerReal = sizeof(real_T) * (NumBitsPerChar);
  if (bitsPerReal == 32U) {
    result = rtIsNaNF((real32_T)value);
  } else {
    union {
      LittleEndianIEEEDouble bitVal;
      real_T fltVal;
    } tmpVal;

    tmpVal.fltVal = value;
    result = (boolean_T)((tmpVal.bitVal.words.wordH & 0x7FF00000) == 0x7FF00000 &&
                         ( (tmpVal.bitVal.words.wordH & 0x000FFFFF) != 0 ||
                          (tmpVal.bitVal.words.wordL != 0) ));
  }

  return result;
}

/* Test if single-precision value is not a number */
static boolean_T rtIsNaNF(real32_T value)
{
  IEEESingle tmp;
  tmp.wordL.wordLreal = value;
  return (boolean_T)( (tmp.wordL.wordLuint & 0x7F800000) == 0x7F800000 &&
                     (tmp.wordL.wordLuint & 0x007FFFFF) != 0 );
}

/*
 * Initialize rtInf needed by the generated code.
 * Inf is initialized as non-signaling. Assumes IEEE.
 */
static real_T rtGetInf(void)
{
  size_t bitsPerReal = sizeof(real_T) * (NumBitsPerChar);
  real_T inf = 0.0;
  if (bitsPerReal == 32U) {
    inf = rtGetInfF();
  } else {
    union {
      LittleEndianIEEEDouble bitVal;
      real_T fltVal;
    } tmpVal;

    tmpVal.bitVal.words.wordH = 0x7FF00000U;
    tmpVal.bitVal.words.wordL = 0x00000000U;
    inf = tmpVal.fltVal;
  }

  return inf;
}

/*
 * Initialize rtInfF needed by the generated code.
 * Inf is initialized as non-signaling. Assumes IEEE.
 */
static real32_T rtGetInfF(void)
{
  IEEESingle infF;
  infF.wordL.wordLuint = 0x7F800000U;
  return infF.wordL.wordLreal;
}

/*
 * Initialize rtMinusInf needed by the generated code.
 * Inf is initialized as non-signaling. Assumes IEEE.
 */
static real_T rtGetMinusInf(void)
{
  size_t bitsPerReal = sizeof(real_T) * (NumBitsPerChar);
  real_T minf = 0.0;
  if (bitsPerReal == 32U) {
    minf = rtGetMinusInfF();
  } else {
    union {
      LittleEndianIEEEDouble bitVal;
      real_T fltVal;
    } tmpVal;

    tmpVal.bitVal.words.wordH = 0xFFF00000U;
    tmpVal.bitVal.words.wordL = 0x00000000U;
    minf = tmpVal.fltVal;
  }

  return minf;
}

/*
 * Initialize rtMinusInfF needed by the generated code.
 * Inf is initialized as non-signaling. Assumes IEEE.
 */
static real32_T rtGetMinusInfF(void)
{
  IEEESingle minfF;
  minfF.wordL.wordLuint = 0xFF800000U;
  return minfF.wordL.wordLreal;
}

/*
 * Initialize rtNaN needed by the generated code.
 * NaN is initialized as non-signaling. Assumes IEEE.
 */
static real_T rtGetNaN(void)
{
  size_t bitsPerReal = sizeof(real_T) * (NumBitsPerChar);
  real_T nan = 0.0;
  if (bitsPerReal == 32U) {
    nan = rtGetNaNF();
  } else {
    union {
      LittleEndianIEEEDouble bitVal;
      real_T fltVal;
    } tmpVal;

    tmpVal.bitVal.words.wordH = 0xFFF80000U;
    tmpVal.bitVal.words.wordL = 0x00000000U;
    nan = tmpVal.fltVal;
  }

  return nan;
}

/*
 * Initialize rtNaNF needed by the generated code.
 * NaN is initialized as non-signaling. Assumes IEEE.
 */
static real32_T rtGetNaNF(void)
{
  IEEESingle nanF = { { 0.0F } };

  nanF.wordL.wordLuint = 0xFFC00000U;
  return nanF.wordL.wordLreal;
}

/*
 * This function updates continuous states using the ODE3 fixed-step
 * solver algorithm
 */
static void rt_ertODEUpdateContinuousStates(RTWSolverInfo *si )
{
  /* Solver Matrices */
  static const real_T rt_ODE3_A[3] = {
    1.0/2.0, 3.0/4.0, 1.0
  };

  static const real_T rt_ODE3_B[3][3] = {
    { 1.0/2.0, 0.0, 0.0 },

    { 0.0, 3.0/4.0, 0.0 },

    { 2.0/9.0, 1.0/3.0, 4.0/9.0 }
  };

  time_T t = rtsiGetT(si);
  time_T tnew = rtsiGetSolverStopTime(si);
  time_T h = rtsiGetStepSize(si);
  real_T *x = rtsiGetContStates(si);
  ODE3_IntgData *id = (ODE3_IntgData *)rtsiGetSolverData(si);
  real_T *y = id->y;
  real_T *f0 = id->f[0];
  real_T *f1 = id->f[1];
  real_T *f2 = id->f[2];
  real_T hB[3];
  int_T i;
  int_T nXc = 12;
  rtsiSetSimTimeStep(si,MINOR_TIME_STEP);

  /* Save the state values at time t in y, we'll use x as ynew. */
  (void) memcpy(y, x,
                (uint_T)nXc*sizeof(real_T));

  /* Assumes that rtsiSetT and ModelOutputs are up-to-date */
  /* f0 = f(t,y) */
  rtsiSetdX(si, f0);
  Quadrotor_derivatives();

  /* f(:,2) = feval(odefile, t + hA(1), y + f*hB(:,1), args(:)(*)); */
  hB[0] = h * rt_ODE3_B[0][0];
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[0]);
  rtsiSetdX(si, f1);
  Quadrotor_step();
  Quadrotor_derivatives();

  /* f(:,3) = feval(odefile, t + hA(2), y + f*hB(:,2), args(:)(*)); */
  for (i = 0; i <= 1; i++) {
    hB[i] = h * rt_ODE3_B[1][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[1]);
  rtsiSetdX(si, f2);
  Quadrotor_step();
  Quadrotor_derivatives();

  /* tnew = t + hA(3);
     ynew = y + f*hB(:,3); */
  for (i = 0; i <= 2; i++) {
    hB[i] = h * rt_ODE3_B[2][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1] + f2[i]*hB[2]);
  }

  rtsiSetT(si, tnew);
  rtsiSetSimTimeStep(si,MAJOR_TIME_STEP);
}

/* Model step function */
void Quadrotor_step(void)
{
  real_T ddx_tmp;
  real_T rtb_Cos;
  real_T rtb_Cos2;
  real_T rtb_Gain_l;
  real_T rtb_Sin;
  real_T rtb_Sin2;
  real_T rtb_Sum1_o;
  real_T rtb_Sum_h;
  if (rtmIsMajorTimeStep(Quadrotor_M)) {
    /* set solver stop time */
    rtsiSetSolverStopTime(&Quadrotor_M->solverInfo,
                          ((Quadrotor_M->Timing.clockTick0+1)*
      Quadrotor_M->Timing.stepSize0));
  }                                    /* end MajorTimeStep */

  /* Update absolute time of base rate at minor time step */
  if (rtmIsMinorTimeStep(Quadrotor_M)) {
    Quadrotor_M->Timing.t[0] = rtsiGetT(&Quadrotor_M->solverInfo);
  }

  /* Integrator: '<S6>/Integrator1' */
  x = Quadrotor_X.Integrator1_CSTATE;

  /* Outport: '<Root>/x' */
  Quadrotor_Y.x_j = x;

  /* Integrator: '<S6>/Integrator3' */
  y = Quadrotor_X.Integrator3_CSTATE;

  /* Outport: '<Root>/y' */
  Quadrotor_Y.y_l = y;

  /* Integrator: '<S6>/Integrator5' */
  z = Quadrotor_X.Integrator5_CSTATE;

  /* Outport: '<Root>/z' */
  Quadrotor_Y.z_f = z;

  /* Integrator: '<S6>/Integrator' */
  dx = Quadrotor_X.Integrator_CSTATE;

  /* Outport: '<Root>/dx' */
  Quadrotor_Y.dx_m = dx;

  /* Integrator: '<S6>/Integrator2' */
  dy = Quadrotor_X.Integrator2_CSTATE;

  /* Outport: '<Root>/dy' */
  Quadrotor_Y.dy_h = dy;

  /* Integrator: '<S6>/Integrator4' */
  dz = Quadrotor_X.Integrator4_CSTATE;

  /* Outport: '<Root>/dz' */
  Quadrotor_Y.dz_h = dz;

  /* Signum: '<S4>/Sign1' incorporates:
   *  Inport: '<Root>/n2'
   */
  if (rtIsNaN(n2)) {
    rtb_Gain_l = n2;
  } else if (n2 < 0.0) {
    rtb_Gain_l = -1.0;
  } else {
    rtb_Gain_l = (n2 > 0.0);
  }

  /* End of Signum: '<S4>/Sign1' */

  /* Product: '<S4>/Product5' incorporates:
   *  Inport: '<Root>/n2'
   *  Math: '<S4>/Square4'
   */
  rtb_Sum1_o = n2 * n2 * rtb_Gain_l;

  /* Signum: '<S4>/Sign3' incorporates:
   *  Inport: '<Root>/n4'
   */
  if (rtIsNaN(n4)) {
    rtb_Gain_l = n4;
  } else if (n4 < 0.0) {
    rtb_Gain_l = -1.0;
  } else {
    rtb_Gain_l = (n4 > 0.0);
  }

  /* End of Signum: '<S4>/Sign3' */

  /* Product: '<S4>/Product7' incorporates:
   *  Inport: '<Root>/n4'
   *  Math: '<S4>/Square2'
   */
  rtb_Sin = n4 * n4 * rtb_Gain_l;

  /* Integrator: '<S9>/Integrator1' */
  theta = Quadrotor_X.Integrator1_CSTATE_f;

  /* Integrator: '<S9>/Integrator' */
  dtheta = Quadrotor_X.Integrator_CSTATE_j;

  /* Integrator: '<S8>/Integrator1' */
  Psi = Quadrotor_X.Integrator1_CSTATE_m;

  /* Integrator: '<S8>/Integrator' */
  dPsi = Quadrotor_X.Integrator_CSTATE_k;


  /* Sum: '<S7>/Sum3' incorporates:
   *  Constant: '<S2>/sqrt(2)*lbody+rrot1'
   *  Gain: '<S4>/Gain2'
   *  Inport: '<Root>/Ixx'
   *  Inport: '<Root>/Iyy'
   *  Inport: '<Root>/Izz'
   *  Inport: '<Root>/L2'
   *  Inport: '<Root>/L4'
   *  Product: '<S4>/Product'
   *  Product: '<S4>/Product1'
   *  Product: '<S5>/Divide'
   *  Product: '<S7>/Divide'
   *  Product: '<S7>/Product'
   *  Product: '<S7>/Product1'
   *  Sum: '<S2>/Sum5'
   *  Sum: '<S2>/Sum7'
   *  Sum: '<S4>/Sum4'
   *  Sum: '<S5>/Sum'
   *  Sum: '<S7>/Sum'
   */
  Quadrotor_B.ddPhi = ((L4 + 0.21384776310850237) * rtb_Sin - (L2 +
    0.21384776310850237) * rtb_Sum1_o) * 3.03E-5 / Ixx + (Iyy - Izz) / Ixx *
    (dtheta * dPsi) + ddPhi_n;

  /* Outport: '<Root>/dPsi' */
  Quadrotor_Y.dPsi_m = dPsi;

  /* Integrator: '<S7>/Integrator1' */
  Phi = Quadrotor_X.Integrator1_CSTATE_a;

  /* Integrator: '<S7>/Integrator' */
  dPhi = Quadrotor_X.Integrator_CSTATE_i;

  /* Signum: '<S4>/Sign' incorporates:
   *  Inport: '<Root>/n1'
   */
  if (rtIsNaN(n1)) {
    rtb_Gain_l = n1;
  } else if (n1 < 0.0) {
    rtb_Gain_l = -1.0;
  } else {
    rtb_Gain_l = (n1 > 0.0);
  }

  /* End of Signum: '<S4>/Sign' */

  /* Product: '<S4>/Product4' incorporates:
   *  Inport: '<Root>/n1'
   *  Math: '<S4>/Square'
   */
  rtb_Sum_h = n1 * n1 * rtb_Gain_l;

  /* Signum: '<S4>/Sign2' incorporates:
   *  Inport: '<Root>/n3'
   */
  if (rtIsNaN(n3)) {
    rtb_Gain_l = n3;
  } else if (n3 < 0.0) {
    rtb_Gain_l = -1.0;
  } else {
    rtb_Gain_l = (n3 > 0.0);
  }

  /* End of Signum: '<S4>/Sign2' */

  /* Product: '<S4>/Product6' incorporates:
   *  Inport: '<Root>/n3'
   *  Math: '<S4>/Square1'
   */
  rtb_Gain_l *= n3 * n3;


  /* Sum: '<S9>/Sum2' incorporates:
   *  Constant: '<S2>/sqrt(2)*lbody+rrot1'
   *  Gain: '<S4>/Gain3'
   *  Inport: '<Root>/Ixx'
   *  Inport: '<Root>/Iyy'
   *  Inport: '<Root>/Izz'
   *  Inport: '<Root>/L1'
   *  Inport: '<Root>/L3'
   *  Product: '<S4>/Product2'
   *  Product: '<S4>/Product3'
   *  Product: '<S5>/Divide1'
   *  Product: '<S9>/Divide'
   *  Product: '<S9>/Product'
   *  Product: '<S9>/Product1'
   *  Sum: '<S2>/Sum4'
   *  Sum: '<S2>/Sum6'
   *  Sum: '<S4>/Sum5'
   *  Sum: '<S5>/Sum1'
   *  Sum: '<S9>/Sum'
   */
  Quadrotor_B.ddtheta = ((L1 + 0.21384776310850237) * rtb_Sum_h - (L3 +
    0.21384776310850237) * rtb_Gain_l) * 3.03E-5 / Iyy + (Izz - Ixx) / Iyy *
    (dPhi * dPsi) + ddtheta_n;

  /* Sum: '<S4>/Sum' */
  rtb_Sum_h += rtb_Gain_l;

  /* Sum: '<S4>/Sum1' */
  rtb_Sum1_o += rtb_Sin;

  /* Gain: '<S6>/Gain' incorporates:
   *  Gain: '<S4>/Gain'
   *  Sum: '<S4>/Sum2'
   */
  rtb_Gain_l = (rtb_Sum_h + rtb_Sum1_o) * 3.03E-5 / mass;

  /* Trigonometry: '<S6>/Sin' */
  rtb_Sin = sin(Phi);

  /* Trigonometry: '<S6>/Sin2' */
  rtb_Sin2 = sin(Psi);

  /* Trigonometry: '<S6>/Cos' */
  rtb_Cos = cos(Phi);

  /* Trigonometry: '<S6>/Cos2' */
  rtb_Cos2 = cos(Psi);


  /* Product: '<S6>/Product1' incorporates:
   *  Product: '<S6>/Product2'
   *  Trigonometry: '<S6>/Sin1'
   */
  ddx_tmp = rtb_Cos * sin(theta);

  /* Sum: '<S6>/Sum3' incorporates:
   *  Product: '<S6>/Product'
   *  Product: '<S6>/Product1'
   *  Product: '<S6>/Product5'
   *  Sum: '<S6>/Sum'
   */
  Quadrotor_B.ddx = (ddx_tmp * rtb_Cos2 + rtb_Sin * rtb_Sin2) * rtb_Gain_l + ddx_n;


  /* Sum: '<S6>/Sum6' incorporates:
   *  Product: '<S6>/Product2'
   *  Product: '<S6>/Product3'
   *  Product: '<S6>/Product6'
   *  Sum: '<S6>/Sum1'
   */
  Quadrotor_B.ddy = (ddx_tmp * rtb_Sin2 - rtb_Sin * rtb_Cos2) * rtb_Gain_l + ddy_n;


  /* Sum: '<S6>/Sum8' incorporates:
   *  Constant: '<S6>/Constant2'
   *  Product: '<S6>/Product4'
   *  Product: '<S6>/Product7'
   *  Sum: '<S6>/Sum2'
   *  Trigonometry: '<S6>/Cos1'
   */
  Quadrotor_B.ddz = cos(theta) * rtb_Cos * rtb_Gain_l - 9.8 + ddz_n;


  /* Sum: '<S8>/Sum1' incorporates:
   *  Gain: '<S4>/Gain1'
   *  Inport: '<Root>/Ixx'
   *  Inport: '<Root>/Iyy'
   *  Inport: '<Root>/Izz'
   *  Product: '<S5>/Divide2'
   *  Product: '<S8>/Divide'
   *  Product: '<S8>/Product'
   *  Product: '<S8>/Product1'
   *  Sum: '<S4>/Sum3'
   *  Sum: '<S5>/Sum2'
   *  Sum: '<S8>/Sum'
   */
  Quadrotor_B.ddPsi = (rtb_Sum_h - rtb_Sum1_o) * 5.5E-7 / Izz + (Ixx - Iyy) /
    Izz * (dPhi * dtheta) + ddPsi_n;

  /* Outport: '<Root>/dPhi' */
  Quadrotor_Y.dPhi_p = dPhi;

  /* Outport: '<Root>/Phi' */
  Quadrotor_Y.Phi_n = Phi;

  /* Outport: '<Root>/Psi' */
  Quadrotor_Y.Psi_f = Psi;

  /* Outport: '<Root>/dtheta' */
  Quadrotor_Y.dtheta_j = dtheta;

  /* Outport: '<Root>/theta' */
  Quadrotor_Y.theta_p = theta;
  if (rtmIsMajorTimeStep(Quadrotor_M)) {
    rt_ertODEUpdateContinuousStates(&Quadrotor_M->solverInfo);

    /* Update absolute time for base rate */
    /* The "clockTick0" counts the number of times the code of this task has
     * been executed. The absolute time is the multiplication of "clockTick0"
     * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
     * overflow during the application lifespan selected.
     */
    ++Quadrotor_M->Timing.clockTick0;
    Quadrotor_M->Timing.t[0] = rtsiGetSolverStopTime(&Quadrotor_M->solverInfo);

    {
      /* Update absolute timer for sample time: [0.001s, 0.0s] */
      /* The "clockTick1" counts the number of times the code of this task has
       * been executed. The resolution of this integer timer is 0.001, which is the step size
       * of the task. Size of "clockTick1" ensures timer will not overflow during the
       * application lifespan selected.
       */
      Quadrotor_M->Timing.clockTick1++;
    }
  }                                    /* end MajorTimeStep */
  /*add*/
  x_o = (float)x;
  y_o = (float)y;
  z_o = (float)z;
  Phi_o = (float)Phi;
  theta_o = (float)theta;
  Psi_o = (float)Psi;
  dx_o = (float)dx;
  dy_o = (float)dy;
  dz_o = (float)dz;
  dPhi_o = (float)dPhi;
  dtheta_o = (float)dtheta;
  dPsi_o = (float)dPsi;
}

/* Derivatives for root system: '<Root>' */
void Quadrotor_derivatives(void)
{
  XDot_Quadrotor_T *_rtXdot;
  _rtXdot = ((XDot_Quadrotor_T *) Quadrotor_M->derivs);

  /* Derivatives for Integrator: '<S6>/Integrator1' */
  _rtXdot->Integrator1_CSTATE = dx;

  /* Derivatives for Integrator: '<S6>/Integrator3' */
  _rtXdot->Integrator3_CSTATE = dy;

  /* Derivatives for Integrator: '<S6>/Integrator5' */
  _rtXdot->Integrator5_CSTATE = dz;

  /* Derivatives for Integrator: '<S6>/Integrator' */
  _rtXdot->Integrator_CSTATE = Quadrotor_B.ddx;

  /* Derivatives for Integrator: '<S6>/Integrator2' */
  _rtXdot->Integrator2_CSTATE = Quadrotor_B.ddy;

  /* Derivatives for Integrator: '<S6>/Integrator4' */
  _rtXdot->Integrator4_CSTATE = Quadrotor_B.ddz;

  /* Derivatives for Integrator: '<S9>/Integrator1' */
  _rtXdot->Integrator1_CSTATE_f = dtheta;

  /* Derivatives for Integrator: '<S9>/Integrator' */
  _rtXdot->Integrator_CSTATE_j = Quadrotor_B.ddtheta;

  /* Derivatives for Integrator: '<S8>/Integrator1' */
  _rtXdot->Integrator1_CSTATE_m = dPsi;

  /* Derivatives for Integrator: '<S8>/Integrator' */
  _rtXdot->Integrator_CSTATE_k = Quadrotor_B.ddPsi;

  /* Derivatives for Integrator: '<S7>/Integrator1' */
  _rtXdot->Integrator1_CSTATE_a = dPhi;

  /* Derivatives for Integrator: '<S7>/Integrator' */
  _rtXdot->Integrator_CSTATE_i = Quadrotor_B.ddPhi;
}

/* Model initialize function */
void Quadrotor_initialize(void)
{
  /* Registration code */

  /* initialize non-finites */
  rt_InitInfAndNaN(sizeof(real_T));

  {
    /* Setup solver object */
    rtsiSetSimTimeStepPtr(&Quadrotor_M->solverInfo,
                          &Quadrotor_M->Timing.simTimeStep);
    rtsiSetTPtr(&Quadrotor_M->solverInfo, &rtmGetTPtr(Quadrotor_M));
    rtsiSetStepSizePtr(&Quadrotor_M->solverInfo, &Quadrotor_M->Timing.stepSize0);
    rtsiSetdXPtr(&Quadrotor_M->solverInfo, &Quadrotor_M->derivs);
    rtsiSetContStatesPtr(&Quadrotor_M->solverInfo, (real_T **)
                         &Quadrotor_M->contStates);
    rtsiSetNumContStatesPtr(&Quadrotor_M->solverInfo,
      &Quadrotor_M->Sizes.numContStates);
    rtsiSetNumPeriodicContStatesPtr(&Quadrotor_M->solverInfo,
      &Quadrotor_M->Sizes.numPeriodicContStates);
    rtsiSetPeriodicContStateIndicesPtr(&Quadrotor_M->solverInfo,
      &Quadrotor_M->periodicContStateIndices);
    rtsiSetPeriodicContStateRangesPtr(&Quadrotor_M->solverInfo,
      &Quadrotor_M->periodicContStateRanges);
    rtsiSetErrorStatusPtr(&Quadrotor_M->solverInfo, (&rtmGetErrorStatus
      (Quadrotor_M)));
    rtsiSetRTModelPtr(&Quadrotor_M->solverInfo, Quadrotor_M);
  }

  rtsiSetSimTimeStep(&Quadrotor_M->solverInfo, MAJOR_TIME_STEP);
  Quadrotor_M->intgData.y = Quadrotor_M->odeY;
  Quadrotor_M->intgData.f[0] = Quadrotor_M->odeF[0];
  Quadrotor_M->intgData.f[1] = Quadrotor_M->odeF[1];
  Quadrotor_M->intgData.f[2] = Quadrotor_M->odeF[2];
  Quadrotor_M->contStates = ((X_Quadrotor_T *) &Quadrotor_X);
  rtsiSetSolverData(&Quadrotor_M->solverInfo, (void *)&Quadrotor_M->intgData);
  rtsiSetIsMinorTimeStepWithModeChange(&Quadrotor_M->solverInfo, false);
  rtsiSetSolverName(&Quadrotor_M->solverInfo,"ode3");
  rtmSetTPtr(Quadrotor_M, &Quadrotor_M->Timing.tArray[0]);
  Quadrotor_M->Timing.stepSize0 = 0.001;

  /* external inputs */
  L1 = 0.15;
  L2 = 0.15;
  L3 = 0.15;
  L4 = 0.15;
  mass = 1.732;
  Ixx = 0.037483483311225634;
  Iyy = 0.037483483311225634;
  Izz = 0.074859433289117927;

  /* InitializeConditions for Integrator: '<S6>/Integrator1' */
  Quadrotor_X.Integrator1_CSTATE = x_0;

  /* InitializeConditions for Integrator: '<S6>/Integrator3' */
  Quadrotor_X.Integrator3_CSTATE = y_0;

  /* InitializeConditions for Integrator: '<S6>/Integrator5' */
  Quadrotor_X.Integrator5_CSTATE = z_0;

  /* InitializeConditions for Integrator: '<S6>/Integrator' */
  Quadrotor_X.Integrator_CSTATE = dx_0;

  /* InitializeConditions for Integrator: '<S6>/Integrator2' */
  Quadrotor_X.Integrator2_CSTATE = dy_0;

  /* InitializeConditions for Integrator: '<S6>/Integrator4' */
  Quadrotor_X.Integrator4_CSTATE = dz_0;

  /* InitializeConditions for Integrator: '<S9>/Integrator1' */
  Quadrotor_X.Integrator1_CSTATE_f = theta_0;

  /* InitializeConditions for Integrator: '<S9>/Integrator' */
  Quadrotor_X.Integrator_CSTATE_j = dtheta_0;

  /* InitializeConditions for Integrator: '<S8>/Integrator1' */
  Quadrotor_X.Integrator1_CSTATE_m = Psi_0;

  /* InitializeConditions for Integrator: '<S8>/Integrator' */
  Quadrotor_X.Integrator_CSTATE_k = dPsi_0;

  /* InitializeConditions for Integrator: '<S7>/Integrator1' */
  Quadrotor_X.Integrator1_CSTATE_a = Phi_0;

  /* InitializeConditions for Integrator: '<S7>/Integrator' */
  Quadrotor_X.Integrator_CSTATE_i = dPhi_0;
}

/* Model terminate function */
void Quadrotor_terminate(void)
{
  /* (no terminate code required) */
}

/*add*/
void motor(float n1i, float n2i, float n3i, float n4i){
  n1 = (real_T)n1i;
  n2 = (real_T)n2i;
  n3 = (real_T)n3i;
  n4 = (real_T)n4i;
}

void arm(float L1i, float L2i, float L3i, float L4i){
  L1 = (real_T)L1i;
  L2 = (real_T)L2i;
  L3 = (real_T)L3i;
  L4 = (real_T)L4i;
}

void inertia(float massi, float Ixxi, float Iyyi, float Izzi){
  mass = (real_T)massi;
  Ixx = (real_T)Ixxi;
  Iyy = (real_T)Iyyi;
  Izz = (real_T)Izzi;
}

void initializestates(float x0i, float y0i, float z0i, float Phi0i, float theta0i, float Psi0i,\
                      float dx0i, float dy0i, float dz0i, float dPhi0i, float dtheta0i, float dPsi0i){
                        x_0 = (real_T)x0i;
                        y_0 = (real_T)y0i;
                        z_0 = (real_T)z0i;
                        Phi_0 = (real_T)Phi0i;
                        theta_0 = (real_T)theta0i;
                        Psi_0 = (real_T)Psi0i;
                        dx_0 = (real_T)dx0i;
                        dy_0 = (real_T)dy0i;
                        dz_0 = (real_T)dz0i;
                        dPhi_0 = (real_T)dPhi0i;
                        dtheta_0 = (real_T)dtheta0i;
                        dPsi_0 = (real_T)dPsi0i;
                      }

void noiseinputs(float ddx_ni, float ddy_ni, float ddz_ni, float ddPhi_ni, float ddtheta_ni, float ddPsi_ni){
  ddx_n = (real_T)ddx_ni;
  ddy_n = (real_T)ddy_ni;
  ddz_n = (real_T)ddz_ni;
  ddPhi_n = (real_T)ddPhi_ni;
  ddtheta_n = (real_T)ddtheta_ni;
  ddPsi_n = (real_T)ddPsi_ni;
}

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
