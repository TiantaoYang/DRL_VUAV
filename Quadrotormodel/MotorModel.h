/*
 * File: MotorModel.h
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

#ifndef RTW_HEADER_MotorModel_h_
#define RTW_HEADER_MotorModel_h_
#ifndef MotorModel_COMMON_INCLUDES_
#define MotorModel_COMMON_INCLUDES_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#endif                                 /* MotorModel_COMMON_INCLUDES_ */

#include <string.h>

/* Model Code Variants */

/* Macros for accessing real-time model data structure */
#ifndef rtmGetErrorStatus
#define rtmGetErrorStatus(rtm)         ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
#define rtmSetErrorStatus(rtm, val)    ((rtm)->errorStatus = (val))
#endif

#ifndef rtmGetStopRequested
#define rtmGetStopRequested(rtm)       ((rtm)->Timing.stopRequestedFlag)
#endif

#ifndef rtmSetStopRequested
#define rtmSetStopRequested(rtm, val)  ((rtm)->Timing.stopRequestedFlag = (val))
#endif

#ifndef rtmGetStopRequestedPtr
#define rtmGetStopRequestedPtr(rtm)    (&((rtm)->Timing.stopRequestedFlag))
#endif

#ifndef rtmGetT
#define rtmGetT(rtm)                   (rtmGetTPtr((rtm))[0])
#endif

#ifndef rtmGetTPtr
#define rtmGetTPtr(rtm)                ((rtm)->Timing.t)
#endif

/* Forward declaration for rtModel */
typedef struct tag_RTM_MotorModel_T RT_MODEL_MotorModel_T;

/* Continuous states (default storage) */
typedef struct {
  real_T TransferFcn_CSTATE[2];        /* '<S1>/Transfer Fcn' */
  real_T TransferFcn1_CSTATE[2];       /* '<S1>/Transfer Fcn1' */
  real_T TransferFcn2_CSTATE[2];       /* '<S1>/Transfer Fcn2' */
  real_T TransferFcn3_CSTATE[2];       /* '<S1>/Transfer Fcn3' */
} X_MotorModel_T;

/* State derivatives (default storage) */
typedef struct {
  real_T TransferFcn_CSTATE[2];        /* '<S1>/Transfer Fcn' */
  real_T TransferFcn1_CSTATE[2];       /* '<S1>/Transfer Fcn1' */
  real_T TransferFcn2_CSTATE[2];       /* '<S1>/Transfer Fcn2' */
  real_T TransferFcn3_CSTATE[2];       /* '<S1>/Transfer Fcn3' */
} XDot_MotorModel_T;

/* State disabled  */
typedef struct {
  boolean_T TransferFcn_CSTATE[2];     /* '<S1>/Transfer Fcn' */
  boolean_T TransferFcn1_CSTATE[2];    /* '<S1>/Transfer Fcn1' */
  boolean_T TransferFcn2_CSTATE[2];    /* '<S1>/Transfer Fcn2' */
  boolean_T TransferFcn3_CSTATE[2];    /* '<S1>/Transfer Fcn3' */
} XDis_MotorModel_T;

#ifndef ODE3_INTG
#define ODE3_INTG

/* ODE3 Integration Data */
typedef struct {
  real_T *y;                           /* output */
  real_T *f[3];                        /* derivatives */
} ODE3_IntgData;

#endif

/* Real-time Model Data Structure */
struct tag_RTM_MotorModel_T {
  const char_T *errorStatus;
  RTWSolverInfo solverInfo;
  X_MotorModel_T *contStates;
  int_T *periodicContStateIndices;
  real_T *periodicContStateRanges;
  real_T *derivs;
  boolean_T *contStateDisabled;
  boolean_T zCCacheNeedsReset;
  boolean_T derivCacheNeedsReset;
  boolean_T CTOutputIncnstWithState;
  real_T odeY[8];
  real_T odeF[3][8];
  ODE3_IntgData intgData;

  /*
   * Sizes:
   * The following substructure contains sizes information
   * for many of the model attributes such as inputs, outputs,
   * dwork, sample times, etc.
   */
  struct {
    int_T numContStates;
    int_T numPeriodicContStates;
    int_T numSampTimes;
  } Sizes;

  /*
   * Timing:
   * The following substructure contains information regarding
   * the timing information for the model.
   */
  struct {
    uint32_T clockTick0;
    time_T stepSize0;
    uint32_T clockTick1;
    SimTimeStep simTimeStep;
    boolean_T stopRequestedFlag;
    time_T *t;
    time_T tArray[2];
  } Timing;
};

/* Continuous states (default storage) */
extern X_MotorModel_T MotorModel_X;

/*
 * Exported Global Signals
 *
 * Note: Exported global signals are block signals with an exported global
 * storage class designation.  Code generation will declare the memory for
 * these signals and export their symbols.
 *
 */
extern real_T n_y1;                    /* '<S1>/Transfer Fcn' */
extern real_T n_y2;                    /* '<S1>/Transfer Fcn1' */
extern real_T n_y3;                    /* '<S1>/Transfer Fcn2' */
extern real_T n_y4;                    /* '<S1>/Transfer Fcn3' */
extern real_T n_r1;                    /* '<Root>/Step' */
extern real_T n_r2;                    /* '<Root>/Step1' */
extern real_T n_r3;                    /* '<Root>/Step2' */
extern real_T n_r4;                    /* '<Root>/Step3' */

/* Model entry point functions */
extern void MotorModel_initialize(void);
extern void MotorModel_step(void);
extern void MotorModel_terminate(void);

/* Real-time Model object */
extern RT_MODEL_MotorModel_T *const MotorModel_M;

/*-
 * These blocks were eliminated from the model due to optimizations:
 *
 * Block '<Root>/Scope' : Unused code path elimination
 * Block '<Root>/Scope1' : Unused code path elimination
 * Block '<Root>/Scope2' : Unused code path elimination
 * Block '<Root>/Scope3' : Unused code path elimination
 */

/*-
 * The generated code includes comments that allow you to trace directly
 * back to the appropriate location in the model.  The basic format
 * is <system>/block_name, where system is the system number (uniquely
 * assigned by Simulink) and block_name is the name of the block.
 *
 * Use the MATLAB hilite_system command to trace the generated code back
 * to the model.  For example,
 *
 * hilite_system('<S3>')    - opens system 3
 * hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'MotorModel'
 * '<S1>'   : 'MotorModel/MotorModel'
 */
#endif                                 /* RTW_HEADER_MotorModel_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
