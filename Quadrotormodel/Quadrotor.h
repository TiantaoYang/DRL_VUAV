/*
 * File: Quadrotor.h
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

#ifndef RTW_HEADER_Quadrotor_h_
#define RTW_HEADER_Quadrotor_h_
#ifndef Quadrotor_COMMON_INCLUDES_
#define Quadrotor_COMMON_INCLUDES_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#endif                                 /* Quadrotor_COMMON_INCLUDES_ */

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
typedef struct tag_RTM_Quadrotor_T RT_MODEL_Quadrotor_T;

/* Block signals (default storage) */
typedef struct {
  real_T ddPhi;                        /* '<S7>/Sum3' */
  real_T ddtheta;                      /* '<S9>/Sum2' */
  real_T ddx;                          /* '<S6>/Sum3' */
  real_T ddy;                          /* '<S6>/Sum6' */
  real_T ddz;                          /* '<S6>/Sum8' */
  real_T ddPsi;                        /* '<S8>/Sum1' */
} B_Quadrotor_T;

/* Continuous states (default storage) */
typedef struct {
  real_T Integrator1_CSTATE;           /* '<S6>/Integrator1' */
  real_T Integrator3_CSTATE;           /* '<S6>/Integrator3' */
  real_T Integrator5_CSTATE;           /* '<S6>/Integrator5' */
  real_T Integrator_CSTATE;            /* '<S6>/Integrator' */
  real_T Integrator2_CSTATE;           /* '<S6>/Integrator2' */
  real_T Integrator4_CSTATE;           /* '<S6>/Integrator4' */
  real_T Integrator1_CSTATE_f;         /* '<S9>/Integrator1' */
  real_T Integrator_CSTATE_j;          /* '<S9>/Integrator' */
  real_T Integrator1_CSTATE_m;         /* '<S8>/Integrator1' */
  real_T Integrator_CSTATE_k;          /* '<S8>/Integrator' */
  real_T Integrator1_CSTATE_a;         /* '<S7>/Integrator1' */
  real_T Integrator_CSTATE_i;          /* '<S7>/Integrator' */
} X_Quadrotor_T;

/* State derivatives (default storage) */
typedef struct {
  real_T Integrator1_CSTATE;           /* '<S6>/Integrator1' */
  real_T Integrator3_CSTATE;           /* '<S6>/Integrator3' */
  real_T Integrator5_CSTATE;           /* '<S6>/Integrator5' */
  real_T Integrator_CSTATE;            /* '<S6>/Integrator' */
  real_T Integrator2_CSTATE;           /* '<S6>/Integrator2' */
  real_T Integrator4_CSTATE;           /* '<S6>/Integrator4' */
  real_T Integrator1_CSTATE_f;         /* '<S9>/Integrator1' */
  real_T Integrator_CSTATE_j;          /* '<S9>/Integrator' */
  real_T Integrator1_CSTATE_m;         /* '<S8>/Integrator1' */
  real_T Integrator_CSTATE_k;          /* '<S8>/Integrator' */
  real_T Integrator1_CSTATE_a;         /* '<S7>/Integrator1' */
  real_T Integrator_CSTATE_i;          /* '<S7>/Integrator' */
} XDot_Quadrotor_T;

/* State disabled  */
typedef struct {
  boolean_T Integrator1_CSTATE;        /* '<S6>/Integrator1' */
  boolean_T Integrator3_CSTATE;        /* '<S6>/Integrator3' */
  boolean_T Integrator5_CSTATE;        /* '<S6>/Integrator5' */
  boolean_T Integrator_CSTATE;         /* '<S6>/Integrator' */
  boolean_T Integrator2_CSTATE;        /* '<S6>/Integrator2' */
  boolean_T Integrator4_CSTATE;        /* '<S6>/Integrator4' */
  boolean_T Integrator1_CSTATE_f;      /* '<S9>/Integrator1' */
  boolean_T Integrator_CSTATE_j;       /* '<S9>/Integrator' */
  boolean_T Integrator1_CSTATE_m;      /* '<S8>/Integrator1' */
  boolean_T Integrator_CSTATE_k;       /* '<S8>/Integrator' */
  boolean_T Integrator1_CSTATE_a;      /* '<S7>/Integrator1' */
  boolean_T Integrator_CSTATE_i;       /* '<S7>/Integrator' */
} XDis_Quadrotor_T;

#ifndef ODE3_INTG
#define ODE3_INTG

/* ODE3 Integration Data */
typedef struct {
  real_T *y;                           /* output */
  real_T *f[3];                        /* derivatives */
} ODE3_IntgData;

#endif

/* External outputs (root outports fed by signals with default storage) */
typedef struct {
  real_T x_j;                          /* '<Root>/x' */
  real_T y_l;                          /* '<Root>/y' */
  real_T z_f;                          /* '<Root>/z' */
  real_T Phi_n;                        /* '<Root>/Phi' */
  real_T theta_p;                      /* '<Root>/theta' */
  real_T Psi_f;                        /* '<Root>/Psi' */
  real_T dx_m;                         /* '<Root>/dx' */
  real_T dy_h;                         /* '<Root>/dy' */
  real_T dz_h;                         /* '<Root>/dz' */
  real_T dPhi_p;                       /* '<Root>/dPhi' */
  real_T dtheta_j;                     /* '<Root>/dtheta' */
  real_T dPsi_m;                       /* '<Root>/dPsi' */
} ExtY_Quadrotor_T;

/* Real-time Model Data Structure */
struct tag_RTM_Quadrotor_T {
  const char_T *errorStatus;
  RTWSolverInfo solverInfo;
  X_Quadrotor_T *contStates;
  int_T *periodicContStateIndices;
  real_T *periodicContStateRanges;
  real_T *derivs;
  boolean_T *contStateDisabled;
  boolean_T zCCacheNeedsReset;
  boolean_T derivCacheNeedsReset;
  boolean_T CTOutputIncnstWithState;
  real_T odeY[12];
  real_T odeF[3][12];
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

/* Block signals (default storage) */
extern B_Quadrotor_T Quadrotor_B;

/* Continuous states (default storage) */
extern X_Quadrotor_T Quadrotor_X;

/* External outputs (root outports fed by signals with default storage) */
extern ExtY_Quadrotor_T Quadrotor_Y;

/*
 * Exported Global Signals
 *
 * Note: Exported global signals are block signals with an exported global
 * storage class designation.  Code generation will declare the memory for
 * these signals and export their symbols.
 *
 */
extern real_T n1;                      /* '<Root>/n1' */
extern real_T n2;                      /* '<Root>/n2' */
extern real_T n3;                      /* '<Root>/n3' */
extern real_T n4;                      /* '<Root>/n4' */
extern real_T L1;                      /* '<Root>/L1' */
extern real_T L2;                      /* '<Root>/L2' */
extern real_T L3;                      /* '<Root>/L3' */
extern real_T L4;                      /* '<Root>/L4' */
extern real_T Ixx;                     /* '<Root>/Ixx' */
extern real_T Iyy;                     /* '<Root>/Iyy' */
extern real_T Izz;                     /* '<Root>/Izz' */
extern real_T x;                       /* '<S6>/Integrator1' */
extern real_T y;                       /* '<S6>/Integrator3' */
extern real_T z;                       /* '<S6>/Integrator5' */
extern real_T dx;                      /* '<S6>/Integrator' */
extern real_T dy;                      /* '<S6>/Integrator2' */
extern real_T dz;                      /* '<S6>/Integrator4' */
extern real_T theta;                   /* '<S9>/Integrator1' */
extern real_T dtheta;                  /* '<S9>/Integrator' */
extern real_T Psi;                     /* '<S8>/Integrator1' */
extern real_T dPsi;                    /* '<S8>/Integrator' */
extern real_T ddPhi_n;                 /* '<S7>/Step' */
extern real_T Phi;                     /* '<S7>/Integrator1' */
extern real_T dPhi;                    /* '<S7>/Integrator' */
extern real_T ddtheta_n;               /* '<S9>/Step' */
extern real_T ddx_n;                   /* '<S6>/Step' */
extern real_T ddy_n;                   /* '<S6>/Step1' */
extern real_T ddz_n;                   /* '<S6>/Step2' */
extern real_T ddPsi_n;                 /* '<S8>/Step' */

/*
 * Exported Global Parameters
 *
 * Note: Exported global parameters are tunable parameters with an exported
 * global storage class designation.  Code generation will declare the memory for
 * these parameters and exports their symbols.
 *
 */
extern real_T Phi_0;                   /* Variable: Phi_0
                                        * Referenced by: '<S7>/Integrator1'
                                        */
extern real_T Psi_0;                   /* Variable: Psi_0
                                        * Referenced by: '<S8>/Integrator1'
                                        */
extern real_T dPhi_0;                  /* Variable: dPhi_0
                                        * Referenced by: '<S7>/Integrator'
                                        */
extern real_T dPsi_0;                  /* Variable: dPsi_0
                                        * Referenced by: '<S8>/Integrator'
                                        */
extern real_T dtheta_0;                /* Variable: dtheta_0
                                        * Referenced by: '<S9>/Integrator'
                                        */
extern real_T dx_0;                    /* Variable: dx_0
                                        * Referenced by: '<S6>/Integrator'
                                        */
extern real_T dy_0;                    /* Variable: dy_0
                                        * Referenced by: '<S6>/Integrator2'
                                        */
extern real_T dz_0;                    /* Variable: dz_0
                                        * Referenced by: '<S6>/Integrator4'
                                        */
extern real_T theta_0;                 /* Variable: theta_0
                                        * Referenced by: '<S9>/Integrator1'
                                        */
extern real_T x_0;                     /* Variable: x_0
                                        * Referenced by: '<S6>/Integrator1'
                                        */
extern real_T y_0;                     /* Variable: y_0
                                        * Referenced by: '<S6>/Integrator3'
                                        */
extern real_T z_0;                     /* Variable: z_0
                                        * Referenced by: '<S6>/Integrator5'
                                        */

/* Model entry point functions */
extern void Quadrotor_initialize(void);
extern void Quadrotor_step(void);
extern void Quadrotor_terminate(void);

/* Real-time Model object */
extern RT_MODEL_Quadrotor_T *const Quadrotor_M;

/*-
 * The generated code includes comments that allow you to trace directly
 * back to the appropriate location in the model.  The basic format
 * is <system>/block_name, where system is the system number (uniquely
 * assigned by Simulink) and block_name is the name of the block.
 *
 * Note that this particular code originates from a subsystem build,
 * and has its own system numbers different from the parent model.
 * Refer to the system hierarchy for this subsystem below, and use the
 * MATLAB hilite_system command to trace the generated code back
 * to the parent model.  For example,
 *
 * hilite_system('QuadrotorContorllerv14/Subsystem/Quadrotor')    - opens subsystem QuadrotorContorllerv14/Subsystem/Quadrotor
 * hilite_system('QuadrotorContorllerv14/Subsystem/Quadrotor/Kp') - opens and selects block Kp
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'QuadrotorContorllerv14/Subsystem'
 * '<S1>'   : 'QuadrotorContorllerv14/Subsystem/Quadrotor'
 * '<S2>'   : 'QuadrotorContorllerv14/Subsystem/Quadrotor/ControlAllocation2'
 * '<S3>'   : 'QuadrotorContorllerv14/Subsystem/Quadrotor/Quadrotor'
 * '<S4>'   : 'QuadrotorContorllerv14/Subsystem/Quadrotor/ControlAllocation2/Controlinputs'
 * '<S5>'   : 'QuadrotorContorllerv14/Subsystem/Quadrotor/Quadrotor/Sub_Inertia'
 * '<S6>'   : 'QuadrotorContorllerv14/Subsystem/Quadrotor/Quadrotor/Sub_xyz'
 * '<S7>'   : 'QuadrotorContorllerv14/Subsystem/Quadrotor/Quadrotor/sub_Phi'
 * '<S8>'   : 'QuadrotorContorllerv14/Subsystem/Quadrotor/Quadrotor/sub_Psi'
 * '<S9>'   : 'QuadrotorContorllerv14/Subsystem/Quadrotor/Quadrotor/sub_theta'
 */
#endif                                 /* RTW_HEADER_Quadrotor_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
