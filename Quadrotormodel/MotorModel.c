/*
 * File: MotorModel.c
 *
 * Code generated for Simulink model 'MotorModel'.
 *
 * Model version                  : 1.3
 * Simulink Coder version         : 9.7 (R2022a) 13-Nov-2021
 * C/C++ source code generated on : Mon Jun  5 12:18:01 2023
 *
 * Target selection: ert.tlc
 * Embedded hardware selection: STMicroelectronics->ST10/Super10
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#include "MotorModel.h"
#include "rtwtypes.h"

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
real_T n_y1;                           /* '<S1>/Transfer Fcn' */
real_T n_y2;                           /* '<S1>/Transfer Fcn1' */
real_T n_y3;                           /* '<S1>/Transfer Fcn2' */
real_T n_y4;                           /* '<S1>/Transfer Fcn3' */
real_T n_r1;                           /* '<Root>/Step' */
real_T n_r2;                           /* '<Root>/Step1' */
real_T n_r3;                           /* '<Root>/Step2' */
real_T n_r4;                           /* '<Root>/Step3' */
float n1_o;
float n2_o;
float n3_o;
float n4_o;


/* Continuous states */
X_MotorModel_T MotorModel_X;

/* Real-time model */
static RT_MODEL_MotorModel_T MotorModel_M_;
RT_MODEL_MotorModel_T *const MotorModel_M = &MotorModel_M_;

/* private model entry point functions */
extern void MotorModel_derivatives(void);

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
  int_T nXc = 8;
  rtsiSetSimTimeStep(si,MINOR_TIME_STEP);

  /* Save the state values at time t in y, we'll use x as ynew. */
  (void) memcpy(y, x,
                (uint_T)nXc*sizeof(real_T));

  /* Assumes that rtsiSetT and ModelOutputs are up-to-date */
  /* f0 = f(t,y) */
  rtsiSetdX(si, f0);
  MotorModel_derivatives();

  /* f(:,2) = feval(odefile, t + hA(1), y + f*hB(:,1), args(:)(*)); */
  hB[0] = h * rt_ODE3_B[0][0];
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[0]);
  rtsiSetdX(si, f1);
  MotorModel_step();
  MotorModel_derivatives();

  /* f(:,3) = feval(odefile, t + hA(2), y + f*hB(:,2), args(:)(*)); */
  for (i = 0; i <= 1; i++) {
    hB[i] = h * rt_ODE3_B[1][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[1]);
  rtsiSetdX(si, f2);
  MotorModel_step();
  MotorModel_derivatives();

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
void MotorModel_step(void)
{
  real_T n_r1_tmp;
  if (rtmIsMajorTimeStep(MotorModel_M)) {
    /* set solver stop time */
    rtsiSetSolverStopTime(&MotorModel_M->solverInfo,
                          ((MotorModel_M->Timing.clockTick0+1)*
      MotorModel_M->Timing.stepSize0));
  }                                    /* end MajorTimeStep */

  /* Update absolute time of base rate at minor time step */
  if (rtmIsMinorTimeStep(MotorModel_M)) {
    MotorModel_M->Timing.t[0] = rtsiGetT(&MotorModel_M->solverInfo);
  }

  /* TransferFcn: '<S1>/Transfer Fcn' */
  n_y1 = 0.0 * MotorModel_X.TransferFcn_CSTATE[0];

  /* TransferFcn: '<S1>/Transfer Fcn1' */
  n_y2 = 0.0 * MotorModel_X.TransferFcn1_CSTATE[0];

  /* TransferFcn: '<S1>/Transfer Fcn2' */
  n_y3 = 0.0 * MotorModel_X.TransferFcn2_CSTATE[0];

  /* TransferFcn: '<S1>/Transfer Fcn3' */
  n_y4 = 0.0 * MotorModel_X.TransferFcn3_CSTATE[0];

  /* TransferFcn: '<S1>/Transfer Fcn' */
  n_y1 += 14400.0 * MotorModel_X.TransferFcn_CSTATE[1];

  /* TransferFcn: '<S1>/Transfer Fcn1' */
  n_y2 += 14400.0 * MotorModel_X.TransferFcn1_CSTATE[1];

  /* TransferFcn: '<S1>/Transfer Fcn2' */
  n_y3 += 14400.0 * MotorModel_X.TransferFcn2_CSTATE[1];

  /* TransferFcn: '<S1>/Transfer Fcn3' */
  n_y4 += 14400.0 * MotorModel_X.TransferFcn3_CSTATE[1];

  /* Step: '<Root>/Step' incorporates:
   *  Step: '<Root>/Step1'
   *  Step: '<Root>/Step2'
   *  Step: '<Root>/Step3'
   */
  n_r1_tmp = MotorModel_M->Timing.t[0];

  if (rtmIsMajorTimeStep(MotorModel_M)) {
    rt_ertODEUpdateContinuousStates(&MotorModel_M->solverInfo);

    /* Update absolute time for base rate */
    /* The "clockTick0" counts the number of times the code of this task has
     * been executed. The absolute time is the multiplication of "clockTick0"
     * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
     * overflow during the application lifespan selected.
     */
    ++MotorModel_M->Timing.clockTick0;
    MotorModel_M->Timing.t[0] = rtsiGetSolverStopTime(&MotorModel_M->solverInfo);

    {
      /* Update absolute timer for sample time: [0.001s, 0.0s] */
      /* The "clockTick1" counts the number of times the code of this task has
       * been executed. The resolution of this integer timer is 0.001, which is the step size
       * of the task. Size of "clockTick1" ensures timer will not overflow during the
       * application lifespan selected.
       */
      MotorModel_M->Timing.clockTick1++;
    }
  }                                    /* end MajorTimeStep */
  n1_o = (float)n_y1;
  n2_o = (float)n_y2;
  n3_o = (float)n_y3;
  n4_o = (float)n_y4;
}

/* Derivatives for root system: '<Root>' */
void MotorModel_derivatives(void)
{
  XDot_MotorModel_T *_rtXdot;
  _rtXdot = ((XDot_MotorModel_T *) MotorModel_M->derivs);

  /* Derivatives for TransferFcn: '<S1>/Transfer Fcn' */
  _rtXdot->TransferFcn_CSTATE[0] = -168.0 * MotorModel_X.TransferFcn_CSTATE[0];

  /* Derivatives for TransferFcn: '<S1>/Transfer Fcn1' */
  _rtXdot->TransferFcn1_CSTATE[0] = -168.0 * MotorModel_X.TransferFcn1_CSTATE[0];

  /* Derivatives for TransferFcn: '<S1>/Transfer Fcn2' */
  _rtXdot->TransferFcn2_CSTATE[0] = -168.0 * MotorModel_X.TransferFcn2_CSTATE[0];

  /* Derivatives for TransferFcn: '<S1>/Transfer Fcn3' */
  _rtXdot->TransferFcn3_CSTATE[0] = -168.0 * MotorModel_X.TransferFcn3_CSTATE[0];

  /* Derivatives for TransferFcn: '<S1>/Transfer Fcn' */
  _rtXdot->TransferFcn_CSTATE[0] += -14400.0 * MotorModel_X.TransferFcn_CSTATE[1];

  /* Derivatives for TransferFcn: '<S1>/Transfer Fcn1' */
  _rtXdot->TransferFcn1_CSTATE[0] += -14400.0 *
    MotorModel_X.TransferFcn1_CSTATE[1];

  /* Derivatives for TransferFcn: '<S1>/Transfer Fcn2' */
  _rtXdot->TransferFcn2_CSTATE[0] += -14400.0 *
    MotorModel_X.TransferFcn2_CSTATE[1];

  /* Derivatives for TransferFcn: '<S1>/Transfer Fcn3' */
  _rtXdot->TransferFcn3_CSTATE[0] += -14400.0 *
    MotorModel_X.TransferFcn3_CSTATE[1];

  /* Derivatives for TransferFcn: '<S1>/Transfer Fcn' */
  _rtXdot->TransferFcn_CSTATE[1] = MotorModel_X.TransferFcn_CSTATE[0];
  _rtXdot->TransferFcn_CSTATE[0] += n_r1;

  /* Derivatives for TransferFcn: '<S1>/Transfer Fcn1' */
  _rtXdot->TransferFcn1_CSTATE[1] = MotorModel_X.TransferFcn1_CSTATE[0];
  _rtXdot->TransferFcn1_CSTATE[0] += n_r2;

  /* Derivatives for TransferFcn: '<S1>/Transfer Fcn2' */
  _rtXdot->TransferFcn2_CSTATE[1] = MotorModel_X.TransferFcn2_CSTATE[0];
  _rtXdot->TransferFcn2_CSTATE[0] += n_r3;

  /* Derivatives for TransferFcn: '<S1>/Transfer Fcn3' */
  _rtXdot->TransferFcn3_CSTATE[1] = MotorModel_X.TransferFcn3_CSTATE[0];
  _rtXdot->TransferFcn3_CSTATE[0] += n_r4;
}

/* Model initialize function */
void MotorModel_initialize(void)
{
  /* Registration code */
  {
    /* Setup solver object */
    rtsiSetSimTimeStepPtr(&MotorModel_M->solverInfo,
                          &MotorModel_M->Timing.simTimeStep);
    rtsiSetTPtr(&MotorModel_M->solverInfo, &rtmGetTPtr(MotorModel_M));
    rtsiSetStepSizePtr(&MotorModel_M->solverInfo,
                       &MotorModel_M->Timing.stepSize0);
    rtsiSetdXPtr(&MotorModel_M->solverInfo, &MotorModel_M->derivs);
    rtsiSetContStatesPtr(&MotorModel_M->solverInfo, (real_T **)
                         &MotorModel_M->contStates);
    rtsiSetNumContStatesPtr(&MotorModel_M->solverInfo,
      &MotorModel_M->Sizes.numContStates);
    rtsiSetNumPeriodicContStatesPtr(&MotorModel_M->solverInfo,
      &MotorModel_M->Sizes.numPeriodicContStates);
    rtsiSetPeriodicContStateIndicesPtr(&MotorModel_M->solverInfo,
      &MotorModel_M->periodicContStateIndices);
    rtsiSetPeriodicContStateRangesPtr(&MotorModel_M->solverInfo,
      &MotorModel_M->periodicContStateRanges);
    rtsiSetErrorStatusPtr(&MotorModel_M->solverInfo, (&rtmGetErrorStatus
      (MotorModel_M)));
    rtsiSetRTModelPtr(&MotorModel_M->solverInfo, MotorModel_M);
  }

  rtsiSetSimTimeStep(&MotorModel_M->solverInfo, MAJOR_TIME_STEP);
  MotorModel_M->intgData.y = MotorModel_M->odeY;
  MotorModel_M->intgData.f[0] = MotorModel_M->odeF[0];
  MotorModel_M->intgData.f[1] = MotorModel_M->odeF[1];
  MotorModel_M->intgData.f[2] = MotorModel_M->odeF[2];
  MotorModel_M->contStates = ((X_MotorModel_T *) &MotorModel_X);
  rtsiSetSolverData(&MotorModel_M->solverInfo, (void *)&MotorModel_M->intgData);
  rtsiSetIsMinorTimeStepWithModeChange(&MotorModel_M->solverInfo, false);
  rtsiSetSolverName(&MotorModel_M->solverInfo,"ode3");
  rtmSetTPtr(MotorModel_M, &MotorModel_M->Timing.tArray[0]);
  MotorModel_M->Timing.stepSize0 = 0.001;

  /* InitializeConditions for TransferFcn: '<S1>/Transfer Fcn' */
  MotorModel_X.TransferFcn_CSTATE[0] = 0.0;

  /* InitializeConditions for TransferFcn: '<S1>/Transfer Fcn1' */
  MotorModel_X.TransferFcn1_CSTATE[0] = 0.0;

  /* InitializeConditions for TransferFcn: '<S1>/Transfer Fcn2' */
  MotorModel_X.TransferFcn2_CSTATE[0] = 0.0;

  /* InitializeConditions for TransferFcn: '<S1>/Transfer Fcn3' */
  MotorModel_X.TransferFcn3_CSTATE[0] = 0.0;

  /* InitializeConditions for TransferFcn: '<S1>/Transfer Fcn' */
  MotorModel_X.TransferFcn_CSTATE[1] = 0.0;

  /* InitializeConditions for TransferFcn: '<S1>/Transfer Fcn1' */
  MotorModel_X.TransferFcn1_CSTATE[1] = 0.0;

  /* InitializeConditions for TransferFcn: '<S1>/Transfer Fcn2' */
  MotorModel_X.TransferFcn2_CSTATE[1] = 0.0;

  /* InitializeConditions for TransferFcn: '<S1>/Transfer Fcn3' */
  MotorModel_X.TransferFcn3_CSTATE[1] = 0.0;
}

/* Model terminate function */
void MotorModel_terminate(void)
{
  /* (no terminate code required) */
}

void Motorinputs(float n_ri1, float n_ri2, float n_ri3, float n_ri4){
  n_r1 = (real_T)n_ri1;
  n_r2 = (real_T)n_ri2;
  n_r3 = (real_T)n_ri3;
  n_r4 = (real_T)n_ri4;
}


/*
 * File trailer for generated code.
 *
 * [EOF]
 */
