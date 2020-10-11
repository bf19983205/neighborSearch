/*=====================================================================
  mps.c

    Program of the MPS method


=======================================================================*/

//#pragma warning(disable : 4996)
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <vector>
#include <ctime>


// #define DIM                  2
// #define PARTICLE_DISTANCE    0.025
// #define DT                   0.001
// #define OUTPUT_INTERVAL      20


/* for three-dimensional simulation */

#define DIM                  3
#define PARTICLE_DISTANCE    0.04
#define DT                   0.003
#define OUTPUT_INTERVAL      2


#define ARRAY_SIZE           9000
#define FINISH_TIME          2.0
#define KINEMATIC_VISCOSITY  (1.0E-6)
#define FLUID_DENSITY        1000.0
#define GRAVITY_X  0.0
#define GRAVITY_Y  -9.8
#define GRAVITY_Z  0.0
#define RADIUS_FOR_NUMBER_DENSITY  (2.1*PARTICLE_DISTANCE)
#define RADIUS_FOR_GRADIENT        (2.1*PARTICLE_DISTANCE)
#define RADIUS_FOR_LAPLACIAN       (3.1*PARTICLE_DISTANCE)
#define COLLISION_DISTANCE         (0.5*PARTICLE_DISTANCE)
#define THRESHOLD_RATIO_OF_NUMBER_DENSITY  0.97
#define COEFFICIENT_OF_RESTITUTION 0.2
#define COMPRESSIBILITY (0.45E-9)
#define EPS             (0.01 * PARTICLE_DISTANCE)
#define ON              1
#define OFF             0
#define RELAXATION_COEFFICIENT_FOR_PRESSURE 0.2
#define GHOST  -1
#define FLUID   0
#define WALL    2
#define DUMMY_WALL  3
#define GHOST_OR_DUMMY  -1
#define SURFACE_PARTICLE 1
#define INNER_PARTICLE   0
#define DIRICHLET_BOUNDARY_IS_NOT_CONNECTED 0
#define DIRICHLET_BOUNDARY_IS_CONNECTED     1
#define DIRICHLET_BOUNDARY_IS_CHECKED       2

double searchRadius = RADIUS_FOR_NUMBER_DENSITY;

double const LENGTH = 1.7;
double const WIDTH = 0.9;
double const HEIGHT = 0.9;


void initializeParticlePositionAndVelocity_for2dim(void);
void initializeParticlePositionAndVelocity_for3dim(void);
void calculateConstantParameter(void);
void calculateNZeroAndLambda(void);
double weight(double distance, double re);
void mainLoopOfSimulation(void);
void calculateGravity(void);
void calculateViscosity(void);
void moveParticle(void);
void collision(void);
void calculatePressure(void);
void calculateParticleNumberDensity(void);
void setBoundaryCondition(void);
void setSourceTerm(void);
void setMatrix(void);
void exceptionalProcessingForBoundaryCondition(void);
void checkBoundaryCondition(void);
void increaseDiagonalTerm(void);
void solveSimultaniousEquationsByGaussEliminationMethod(void);
void removeNegativePressure(void);
void setMinimumPressure(void);
void calculatePressureGradient(void);
void moveParticleUsingPressureGradient(void);
void writeData_inProfFormat(void);
void writeData_inVtuFormat(void);


static double AccelerationX[ARRAY_SIZE];
static double AccelerationY[ARRAY_SIZE];
static double AccelerationZ[ARRAY_SIZE];
static int    ParticleType[ARRAY_SIZE];
static double PositionX[ARRAY_SIZE];
static double PositionY[ARRAY_SIZE];
static double PositionZ[ARRAY_SIZE];
static double VelocityX[ARRAY_SIZE];
static double VelocityY[ARRAY_SIZE];
static double VelocityZ[ARRAY_SIZE];
static double Pressure[ARRAY_SIZE];
static double ParticleNumberDensity[ARRAY_SIZE];
static int    BoundaryCondition[ARRAY_SIZE];
static double SourceTerm[ARRAY_SIZE];
static int    FlagForCheckingBoundaryCondition[ARRAY_SIZE];
static double CoefficientMatrix[ARRAY_SIZE * ARRAY_SIZE];
static double MinimumPressure[ARRAY_SIZE];
int    FileNumber;
double Time;

static double r[ARRAY_SIZE];
static double d[ARRAY_SIZE];
static double oldr[ARRAY_SIZE];
//static double product[ARRAY_SIZE];




int    NumberOfParticles;
int    NumberOfNonzeros;


double Re_forParticleNumberDensity, Re2_forParticleNumberDensity;
double Re_forGradient, Re2_forGradient;
double Re_forLaplacian, Re2_forLaplacian;
double N0_forParticleNumberDensity;
double N0_forGradient;
double N0_forLaplacian;
double Lambda;
double collisionDistance, collisionDistance2;
double FluidDensity;



double dotProduct(double a[ARRAY_SIZE], double b[ARRAY_SIZE]);
double* matrixVectorProduct(double a[ARRAY_SIZE * ARRAY_SIZE], double b[ARRAY_SIZE]);
double norm2(double a[ARRAY_SIZE]);
void conjugateGradient(void);
double* preconditionedConjugateGradient(double A[ARRAY_SIZE * ARRAY_SIZE], double b[ARRAY_SIZE]);
double* gaussianelimination(double A[ARRAY_SIZE * ARRAY_SIZE], double b[ARRAY_SIZE]);
//std::vector<double> choleskyFactorization();
void incompleteCholeskyFactorization(double L[ARRAY_SIZE * ARRAY_SIZE], double ChoL[ARRAY_SIZE * ARRAY_SIZE]);
void matrixProduct(double L[ARRAY_SIZE * ARRAY_SIZE], double M[ARRAY_SIZE * ARRAY_SIZE]);
void tranposeMatrix(double L[ARRAY_SIZE * ARRAY_SIZE], double R[ARRAY_SIZE * ARRAY_SIZE]);

size_t getHashKeyFromPosition(int i);
std::vector<int> getBucketIndex(int i);
std::vector<size_t> getNearbyKeys(int originIndex);
size_t getHashKeyFromBucketIndex(std::vector<int> bucketIndex);

void forEachNearbyPoint(int originIndex);
void buildParticlesInBuckets(void);
void buildNeighborList(void);

double AV[50 * ARRAY_SIZE];
int AJ[50 * ARRAY_SIZE];
int AI[ARRAY_SIZE];
double tempMatrix[ARRAY_SIZE];

std::vector<std::vector<size_t>> neighborList;
std::vector<std::vector<size_t>> buckets;


int main(void) {

    printf("\n*** START MPS-SIMULATION ***\n");
    if (DIM == 2) {
        initializeParticlePositionAndVelocity_for2dim();
    }
    else {
        initializeParticlePositionAndVelocity_for3dim();
    }
    calculateConstantParameter();
    mainLoopOfSimulation();
    printf("*** END ***\n\n");

    //double A[2 * 2];
    //double b[2];

    //A[0] = 2; A[1] = 2; A[2] = 2; A[3] = 5;
    //b[0] = 6;
    //b[1] = 3;

    ////double*lll = preconditionedConjugateGradient(A, b);
    //double*lll = conjugateGradient(A, b);

    //for (int i = 0; i < NumberOfParticles; ++i) {
    //	std::cout << lll[i] << std::endl;
    //}

    
    return 0;

}


void initializeParticlePositionAndVelocity_for2dim( void ){

  int iX, iY;
  int nX, nY;
  double x, y, z;
  int i = 0;
  int flagOfParticleGeneration;

  nX = (int)(1.0/PARTICLE_DISTANCE)+5;
  nY = (int)(0.6/PARTICLE_DISTANCE)+5;
  for(iX= -4;iX<nX;iX++){
    for(iY= -4;iY<nY;iY++){
      x = PARTICLE_DISTANCE * (double)(iX);
      y = PARTICLE_DISTANCE * (double)(iY);
      z = 0.0;
      flagOfParticleGeneration = OFF;

      if( ((x>-4.0*PARTICLE_DISTANCE+EPS)&&(x<=1.00+4.0*PARTICLE_DISTANCE+EPS))&&( (y>0.0-4.0*PARTICLE_DISTANCE+EPS )&&(y<=0.6+EPS)) ){  /* dummy wall region */
    ParticleType[i]=DUMMY_WALL;
    flagOfParticleGeneration = ON;
      }

      if( ((x>-2.0*PARTICLE_DISTANCE+EPS)&&(x<=1.00+2.0*PARTICLE_DISTANCE+EPS))&&( (y>0.0-2.0*PARTICLE_DISTANCE+EPS )&&(y<=0.6+EPS)) ){ /* wall region */
    ParticleType[i]=WALL;
    flagOfParticleGeneration = ON;
      }

      if( ((x>-4.0*PARTICLE_DISTANCE+EPS)&&(x<=1.00+4.0*PARTICLE_DISTANCE+EPS))&&( (y>0.6-2.0*PARTICLE_DISTANCE+EPS )&&(y<=0.6+EPS)) ){  /* wall region */
    ParticleType[i]=WALL;
    flagOfParticleGeneration = ON;
      }

      if( ((x>0.0+EPS)&&(x<=1.00+EPS))&&( y>0.0+EPS )){  /* empty region */
    flagOfParticleGeneration = OFF;
      }

      if( ((x>0.0+EPS)&&(x<=0.25+EPS)) &&((y>0.0+EPS)&&(y<=0.50+EPS)) ){  /* fluid region */
    ParticleType[i]=FLUID;
    flagOfParticleGeneration = ON;
      }

      if( flagOfParticleGeneration == ON){
    PositionX[i]=x; PositionY[i]=y; PositionZ[i]=z;
    i++;
      }
    }
  }
  NumberOfParticles = i;
  for(i=0;i<NumberOfParticles;i++) { VelocityX[i]=0.0; VelocityY[i]=0.0; VelocityZ[i]=0.0; }
}


void initializeParticlePositionAndVelocity_for3dim(void) {
    int iX, iY, iZ;
    int nX, nY, nZ;
    double x, y, z;
    int i = 0;
    int flagOfParticleGeneration;

    nX = (int)(1.0 / PARTICLE_DISTANCE) + 5;
    nY = (int)(0.6 / PARTICLE_DISTANCE) + 5;
    nZ = (int)(0.3 / PARTICLE_DISTANCE) + 5;
    for (iX = -4; iX < nX; iX++) {
        for (iY = -4; iY < nY; iY++) {
            for (iZ = -4; iZ < nZ; iZ++) {
                x = PARTICLE_DISTANCE * iX;
                y = PARTICLE_DISTANCE * iY;
                z = PARTICLE_DISTANCE * iZ;
                flagOfParticleGeneration = OFF;

                /* dummy wall region */
                if ((((x > -4.0 * PARTICLE_DISTANCE + EPS) && (x <= 1.00 + 4.0 * PARTICLE_DISTANCE + EPS)) && ((y > 0.0 - 4.0 * PARTICLE_DISTANCE + EPS) && (y <= 0.6 + EPS))) && ((z > 0.0 - 4.0 * PARTICLE_DISTANCE + EPS) && (z <= 0.3 + 4.0 * PARTICLE_DISTANCE + EPS))) {
                    ParticleType[i] = DUMMY_WALL;
                    flagOfParticleGeneration = ON;
                }

                /* wall region */
                if ((((x > -2.0 * PARTICLE_DISTANCE + EPS) && (x <= 1.00 + 2.0 * PARTICLE_DISTANCE + EPS)) && ((y > 0.0 - 2.0 * PARTICLE_DISTANCE + EPS) && (y <= 0.6 + EPS))) && ((z > 0.0 - 2.0 * PARTICLE_DISTANCE + EPS) && (z <= 0.3 + 2.0 * PARTICLE_DISTANCE + EPS))) {
                    ParticleType[i] = WALL;
                    flagOfParticleGeneration = ON;
                }

                /* wall region */
                if ((((x > -4.0 * PARTICLE_DISTANCE + EPS) && (x <= 1.00 + 4.0 * PARTICLE_DISTANCE + EPS)) && ((y > 0.6 - 2.0 * PARTICLE_DISTANCE + EPS) && (y <= 0.6 + EPS))) && ((z > 0.0 - 4.0 * PARTICLE_DISTANCE + EPS) && (z <= 0.3 + 4.0 * PARTICLE_DISTANCE + EPS))) {
                    ParticleType[i] = WALL;
                    flagOfParticleGeneration = ON;
                }

                /* empty region */
                if ((((x > 0.0 + EPS) && (x <= 1.00 + EPS)) && (y > 0.0 + EPS)) && ((z > 0.0 + EPS) && (z <= 0.3 + EPS))) {
                    flagOfParticleGeneration = OFF;
                }

                /* fluid region */
                if ((((x > 0.0 + EPS) && (x <= 0.25 + EPS)) && ((y > 0.0 + EPS) && (y < 0.5 + EPS))) && ((z > 0.0 + EPS) && (z <= 0.3 + EPS))) {
                    ParticleType[i] = FLUID;
                    flagOfParticleGeneration = ON;
                }

                if (flagOfParticleGeneration == ON) {
                    PositionX[i] = x;
                    PositionY[i] = y;
                    PositionZ[i] = z;
                    i++;
                }
            }
        }
    }
    NumberOfParticles = i;
    for (i = 0; i < NumberOfParticles; i++) { VelocityX[i] = 0.0; VelocityY[i] = 0.0; VelocityZ[i] = 0.0; }
}


void calculateConstantParameter(void) {

    Re_forParticleNumberDensity = RADIUS_FOR_NUMBER_DENSITY;
    Re_forGradient = RADIUS_FOR_GRADIENT;
    Re_forLaplacian = RADIUS_FOR_LAPLACIAN;
    Re2_forParticleNumberDensity = Re_forParticleNumberDensity * Re_forParticleNumberDensity;
    Re2_forGradient = Re_forGradient * Re_forGradient;
    Re2_forLaplacian = Re_forLaplacian * Re_forLaplacian;
    calculateNZeroAndLambda();
    FluidDensity = FLUID_DENSITY;
    collisionDistance = COLLISION_DISTANCE;
    collisionDistance2 = collisionDistance * collisionDistance;
    FileNumber = 0;
    Time = 0.0;
}


void calculateNZeroAndLambda(void) {
    int iX, iY, iZ;
    int iZ_start, iZ_end;
    double xj, yj, zj, distance, distance2;
    double xi, yi, zi;

    if (DIM == 2) {
        iZ_start = 0; iZ_end = 1;
    }
    else {
        iZ_start = -4; iZ_end = 5;
    }

    N0_forParticleNumberDensity = 0.0;
    N0_forGradient = 0.0;
    N0_forLaplacian = 0.0;
    Lambda = 0.0;
    xi = 0.0;  yi = 0.0;  zi = 0.0;

    for (iX = -4; iX < 5; iX++) {
        for (iY = -4; iY < 5; iY++) {
            for (iZ = iZ_start; iZ < iZ_end; iZ++) {
                if (((iX == 0) && (iY == 0)) && (iZ == 0))continue;
                xj = PARTICLE_DISTANCE * (double)(iX);
                yj = PARTICLE_DISTANCE * (double)(iY);
                zj = PARTICLE_DISTANCE * (double)(iZ);
                distance2 = (xj - xi) * (xj - xi) + (yj - yi) * (yj - yi) + (zj - zi) * (zj - zi);
                distance = sqrt(distance2);
                N0_forParticleNumberDensity += weight(distance, Re_forParticleNumberDensity);
                N0_forGradient += weight(distance, Re_forGradient);
                N0_forLaplacian += weight(distance, Re_forLaplacian);
                Lambda += distance2 * weight(distance, Re_forLaplacian);
            }
        }
    }
    Lambda = Lambda / N0_forLaplacian;
}

// MPS方法开始的和函数，特点是粒子距离为0时，作用力无穷大
//double weight(double distance, double re) {
//	double weightIJ;

//	if (distance >= re) {
//		weightIJ = 0.0;
//	}
//	else {
//		weightIJ = (re / distance) - 1.0;
//	}
//	return weightIJ;
//}


double weight(double distance, double re) {
    double weightij;

    if (distance >= re) {
        weightij = 0.0;
    }
    else {
        weightij = re / (0.85 * distance + 0.15 * re) - 1.0;
    }
    return weightij;
}


void mainLoopOfSimulation(void) {
    int iTimeStep = 0;

    writeData_inVtuFormat();
    writeData_inProfFormat();

    while (1) {

        buildParticlesInBuckets();
        buildNeighborList();

        calculateGravity();
        calculateViscosity();
        moveParticle();
        collision();
        calculatePressure();
        calculatePressureGradient();
        moveParticleUsingPressureGradient();


        iTimeStep++;
        Time += DT;
        if ((iTimeStep % OUTPUT_INTERVAL) == 0) {
            printf("TimeStepNumber: %4d   Time: %lf(s)   NumberOfParticles: %d\n", iTimeStep, Time, NumberOfParticles);
            writeData_inVtuFormat();
            writeData_inProfFormat();
        }
        if (Time >= FINISH_TIME) { break; }
    }
}


void calculateGravity(void) {
    int i;

    for (i = 0; i < NumberOfParticles; i++) {
        if (ParticleType[i] == FLUID) {
            AccelerationX[i] = GRAVITY_X;
            AccelerationY[i] = GRAVITY_Y;
            AccelerationZ[i] = GRAVITY_Z;
        }
        else {
            AccelerationX[i] = 0.0;
            AccelerationY[i] = 0.0;
            AccelerationZ[i] = 0.0;
        }
    }
}


void calculateViscosity(void) {
    int i, j;
    double viscosityTermX, viscosityTermY, viscosityTermZ;
    double distance, distance2;
    double w;
    double xij, yij, zij;
    double a;

    a = (KINEMATIC_VISCOSITY) * (2.0 * DIM) / (N0_forLaplacian * Lambda);
    for (i = 0; i < NumberOfParticles; i++) {
        if (ParticleType[i] != FLUID) continue;
        viscosityTermX = 0.0;  viscosityTermY = 0.0;  viscosityTermZ = 0.0;

        for (j = 0; j < NumberOfParticles; j++) {
            if ((j == i) || (ParticleType[j] == GHOST)) continue;
            xij = PositionX[j] - PositionX[i];
            yij = PositionY[j] - PositionY[i];
            zij = PositionZ[j] - PositionZ[i];
            distance2 = (xij * xij) + (yij * yij) + (zij * zij);
            distance = sqrt(distance2);
            if (distance < Re_forLaplacian) {
                w = weight(distance, Re_forLaplacian);
                viscosityTermX += (VelocityX[j] - VelocityX[i]) * w;
                viscosityTermY += (VelocityY[j] - VelocityY[i]) * w;
                viscosityTermZ += (VelocityZ[j] - VelocityZ[i]) * w;
            }
        }
        viscosityTermX = viscosityTermX * a;
        viscosityTermY = viscosityTermY * a;
        viscosityTermZ = viscosityTermZ * a;
        AccelerationX[i] += viscosityTermX;
        AccelerationY[i] += viscosityTermY;
        AccelerationZ[i] += viscosityTermZ;
    }
}


void moveParticle(void) {
    int i;

    for (i = 0; i < NumberOfParticles; i++) {
        if (ParticleType[i] == FLUID) {
            VelocityX[i] += AccelerationX[i] * DT;
            VelocityY[i] += AccelerationY[i] * DT;
            VelocityZ[i] += AccelerationZ[i] * DT;

            PositionX[i] += VelocityX[i] * DT;
            PositionY[i] += VelocityY[i] * DT;
            PositionZ[i] += VelocityZ[i] * DT;
        }
        AccelerationX[i] = 0.0;
        AccelerationY[i] = 0.0;
        AccelerationZ[i] = 0.0;
    }
}


void collision(void) {
    int    i, j;
    double xij, yij, zij;
    double distance, distance2;
    double forceDT; /* forceDT is the impulse of collision between particles */
    double mi, mj;
    double velocity_ix, velocity_iy, velocity_iz;
    double e = COEFFICIENT_OF_RESTITUTION;
    static double VelocityAfterCollisionX[ARRAY_SIZE];
    static double VelocityAfterCollisionY[ARRAY_SIZE];
    static double VelocityAfterCollisionZ[ARRAY_SIZE];

    for (i = 0; i < NumberOfParticles; i++) {
        VelocityAfterCollisionX[i] = VelocityX[i];
        VelocityAfterCollisionY[i] = VelocityY[i];
        VelocityAfterCollisionZ[i] = VelocityZ[i];
    }
    for (i = 0; i < NumberOfParticles; i++) {
        if (ParticleType[i] == FLUID) {
            mi = FluidDensity;
            velocity_ix = VelocityX[i];
            velocity_iy = VelocityY[i];
            velocity_iz = VelocityZ[i];
            for (j = 0; j < NumberOfParticles; j++) {
                if ((j == i) || (ParticleType[j] == GHOST)) continue;
                xij = PositionX[j] - PositionX[i];
                yij = PositionY[j] - PositionY[i];
                zij = PositionZ[j] - PositionZ[i];
                distance2 = (xij * xij) + (yij * yij) + (zij * zij);
                if (distance2 < collisionDistance2) {
                    distance = sqrt(distance2);
                    forceDT = (velocity_ix - VelocityX[j]) * (xij / distance)
                        + (velocity_iy - VelocityY[j]) * (yij / distance)
                        + (velocity_iz - VelocityZ[j]) * (zij / distance);
                    if (forceDT > 0.0) {
                        mj = FluidDensity;
                        forceDT *= (1.0 + e) * mi * mj / (mi + mj);
                        velocity_ix -= (forceDT / mi) * (xij / distance);
                        velocity_iy -= (forceDT / mi) * (yij / distance);
                        velocity_iz -= (forceDT / mi) * (zij / distance);
                        /*
                        if(j>i){ fprintf(stderr,"WARNING: Collision occured between %d and %d particles.\n",i,j); }
                        */
                    }
                }
            }
            VelocityAfterCollisionX[i] = velocity_ix;
            VelocityAfterCollisionY[i] = velocity_iy;
            VelocityAfterCollisionZ[i] = velocity_iz;
        }
    }
    for (i = 0; i < NumberOfParticles; i++) {
        if (ParticleType[i] == FLUID) {
            PositionX[i] += (VelocityAfterCollisionX[i] - VelocityX[i]) * DT;
            PositionY[i] += (VelocityAfterCollisionY[i] - VelocityY[i]) * DT;
            PositionZ[i] += (VelocityAfterCollisionZ[i] - VelocityZ[i]) * DT;
            VelocityX[i] = VelocityAfterCollisionX[i];
            VelocityY[i] = VelocityAfterCollisionY[i];
            VelocityZ[i] = VelocityAfterCollisionZ[i];
        }
    }
}


void calculatePressure(void) {
    calculateParticleNumberDensity();
    setBoundaryCondition();
    setSourceTerm();
    setMatrix();
    //solveSimultaniousEquationsByGaussEliminationMethod();
    //int i = 250;

    //clock_t start0, end0;
    //start0 = clock();

    conjugateGradient();

    //end0 = clock();   //结束时间
    //std::cout << "conjugateGradient(); time = " << double(end0 - start0) / CLOCKS_PER_SEC << "s" << std::endl;  //输出时间（单位：ｓ）


    //double* x = preconditionedConjugateGradient(CoefficientMatrix, SourceTerm);

    removeNegativePressure();
    setMinimumPressure();
}


void calculateParticleNumberDensity(void) {
    int    i, j;
    double xij, yij, zij;
    double distance, distance2;
    double w;

    for (i = 0; i < NumberOfParticles; i++) {
        ParticleNumberDensity[i] = 0.0;
        if (ParticleType[i] == GHOST) continue;
        for (j = 0; j < NumberOfParticles; j++) {
            if ((j == i) || (ParticleType[j] == GHOST)) continue;
            xij = PositionX[j] - PositionX[i];
            yij = PositionY[j] - PositionY[i];
            zij = PositionZ[j] - PositionZ[i];
            distance2 = (xij * xij) + (yij * yij) + (zij * zij);
            distance = sqrt(distance2);
            w = weight(distance, Re_forParticleNumberDensity);
            ParticleNumberDensity[i] += w;
        }
    }



    // for (i = 0; i < NumberOfParticles; i++) {
    //     ParticleNumberDensity[i] = 0.0;

    //     for(int j = 0 ;j != neighborList[i].size(); ++j) {
    //         if (ParticleType[j] == GHOST) continue;
    //         int particleIndex = neighborList[i][j];
    //         xij = PositionX[particleIndex] - PositionX[i];
    //         yij = PositionY[particleIndex] - PositionY[i];
    //         zij = PositionZ[particleIndex] - PositionZ[i];
    //         distance2 = (xij * xij) + (yij * yij) + (zij * zij);
    //         distance = sqrt(distance2);
    //         w = weight(distance, Re_forParticleNumberDensity);
    //         ParticleNumberDensity[i] += w;
    //     }
    // }
}


void setBoundaryCondition(void) {
    int i;
    double n0 = N0_forParticleNumberDensity;
    double beta = THRESHOLD_RATIO_OF_NUMBER_DENSITY;

    for (i = 0; i < NumberOfParticles; i++) {
        if (ParticleType[i] == GHOST || ParticleType[i] == DUMMY_WALL) {
            BoundaryCondition[i] = GHOST_OR_DUMMY;
        }
        else if (ParticleNumberDensity[i] < beta * n0) {
            BoundaryCondition[i] = SURFACE_PARTICLE;
        }
        else {
            BoundaryCondition[i] = INNER_PARTICLE;
        }
    }
}


void setSourceTerm(void) {
    int i;
    double n0 = N0_forParticleNumberDensity;
    double gamma = RELAXATION_COEFFICIENT_FOR_PRESSURE;

    for (i = 0; i < NumberOfParticles; i++) {
        SourceTerm[i] = 0.0;
        if (ParticleType[i] == GHOST || ParticleType[i] == DUMMY_WALL) continue;
        if (BoundaryCondition[i] == INNER_PARTICLE) {
            SourceTerm[i] = gamma * (1.0 / (DT * DT)) * ((ParticleNumberDensity[i] - n0) / n0);
        }
        else if (BoundaryCondition[i] == SURFACE_PARTICLE) {
            SourceTerm[i] = 0.0;
        }
    }
}


void setMatrix(void) {
    double xij, yij, zij;
    double distance, distance2;
    double coefficientIJ;
    double n0 = N0_forLaplacian;
    int    i, j;
    double a;
    int n = NumberOfParticles;

    NumberOfNonzeros = 0;

    /*for (i = 0; i < NumberOfParticles; i++) {
        for (j = 0; j < NumberOfParticles; j++) {
            CoefficientMatrix[i*n + j] = 0.0;
        }
    }*/
    a = 2.0 * DIM / (n0 * Lambda);
    for (i = 0; i < NumberOfParticles; i++) {
        AI[i] = -1;

        for (int n = 0; n < NumberOfParticles; ++n) {
            tempMatrix[n] = 0.0;
        }
        if (BoundaryCondition[i] != INNER_PARTICLE) continue;
        for (j = 0; j < NumberOfParticles; j++) {
            if ((j == i) || (BoundaryCondition[j] == GHOST_OR_DUMMY)) continue;
            xij = PositionX[j] - PositionX[i];
            yij = PositionY[j] - PositionY[i];
            zij = PositionZ[j] - PositionZ[i];
            distance2 = (xij * xij) + (yij * yij) + (zij * zij);
            distance = sqrt(distance2);
            if (distance >= Re_forLaplacian)continue;
            coefficientIJ = a * weight(distance, Re_forLaplacian) / FluidDensity;
            //CoefficientMatrix[i*n + j] = (-1.0)*coefficientIJ;
            //CoefficientMatrix[i*n + i] += coefficientIJ;

            tempMatrix[j] = (-1.0) * coefficientIJ;
            tempMatrix[i] += coefficientIJ;
        }

        int iflag = 0;
        for (int m = 0; m < NumberOfParticles; ++m) {
            if (tempMatrix[m] != 0) {
                if (iflag == 0) {
                    AI[i] = NumberOfNonzeros;
                    iflag++;
                }
                AV[NumberOfNonzeros] = tempMatrix[m];
                AJ[NumberOfNonzeros] = m;

                NumberOfNonzeros++;
            }
        }
        //CoefficientMatrix[i*n + i] += (COMPRESSIBILITY) / (DT*DT);
    }
    AI[i] = NumberOfNonzeros;
    //exceptionalProcessingForBoundaryCondition();
}


void exceptionalProcessingForBoundaryCondition(void) {
    /* If tere is no Dirichlet boundary condition on the fluid,
       increase the diagonal terms of the matrix for an exception. This allows us to solve the matrix without Dirichlet boundary conditions. */
    checkBoundaryCondition();
    increaseDiagonalTerm();
}


void checkBoundaryCondition(void) {
    int i, j, count;
    double xij, yij, zij, distance2;

    for (i = 0; i < NumberOfParticles; i++) {
        if (BoundaryCondition[i] == GHOST_OR_DUMMY) {
            FlagForCheckingBoundaryCondition[i] = GHOST_OR_DUMMY;
        }
        else if (BoundaryCondition[i] == SURFACE_PARTICLE) {
            FlagForCheckingBoundaryCondition[i] = DIRICHLET_BOUNDARY_IS_CONNECTED;
        }
        else {
            FlagForCheckingBoundaryCondition[i] = DIRICHLET_BOUNDARY_IS_NOT_CONNECTED;
        }
    }

    do {
        count = 0;
        for (i = 0; i < NumberOfParticles; i++) {
            if (FlagForCheckingBoundaryCondition[i] == DIRICHLET_BOUNDARY_IS_CONNECTED) {
                for (j = 0; j < NumberOfParticles; j++) {
                    if (j == i) continue;
                    if ((ParticleType[j] == GHOST) || (ParticleType[j] == DUMMY_WALL)) continue;
                    if (FlagForCheckingBoundaryCondition[j] == DIRICHLET_BOUNDARY_IS_NOT_CONNECTED) {
                        xij = PositionX[j] - PositionX[i];
                        yij = PositionY[j] - PositionY[i];
                        zij = PositionZ[j] - PositionZ[i];
                        distance2 = (xij * xij) + (yij * yij) + (zij * zij);
                        if (distance2 >= Re2_forLaplacian)continue;
                        FlagForCheckingBoundaryCondition[j] = DIRICHLET_BOUNDARY_IS_CONNECTED;
                    }
                }
                FlagForCheckingBoundaryCondition[i] = DIRICHLET_BOUNDARY_IS_CHECKED;
                count++;
            }
        }
    } while (count != 0); /* This procedure is repeated until the all fluid or wall particles (which have Dirhchlet boundary condition in the particle group) are in the state of "DIRICHLET_BOUNDARY_IS_CHECKED".*/

    for (i = 0; i < NumberOfParticles; i++) {
        if (FlagForCheckingBoundaryCondition[i] == DIRICHLET_BOUNDARY_IS_NOT_CONNECTED) {
            fprintf(stderr, "WARNING: There is no dirichlet boundary condition for %d-th particle.\n", i);
        }
    }
}


void increaseDiagonalTerm(void) {
    int i;
    int n = NumberOfParticles;

    for (i = 0; i < n; i++) {
        if (FlagForCheckingBoundaryCondition[i] == DIRICHLET_BOUNDARY_IS_NOT_CONNECTED) {
            CoefficientMatrix[i * n + i] = 2.0 * CoefficientMatrix[i * n + i];
        }
    }
}


void solveSimultaniousEquationsByGaussEliminationMethod(void) {
    int    i, j, k;
    double c;
    double sumOfTerms;
    int    n = NumberOfParticles;

    for (i = 0; i < n; i++) {
        Pressure[i] = 0.0;
    }
    for (i = 0; i < n - 1; i++) {
        if (BoundaryCondition[i] != INNER_PARTICLE) continue;
        for (j = i + 1; j < n; j++) {
            if (BoundaryCondition[j] == GHOST_OR_DUMMY) continue;
            c = CoefficientMatrix[j * n + i] / CoefficientMatrix[i * n + i];
            for (k = i + 1; k < n; k++) {
                CoefficientMatrix[j * n + k] -= c * CoefficientMatrix[i * n + k];
            }
            SourceTerm[j] -= c * SourceTerm[i];
        }
    }
    for (i = n - 1; i >= 0; i--) {
        if (BoundaryCondition[i] != INNER_PARTICLE) continue;
        sumOfTerms = 0.0;
        for (j = i + 1; j < n; j++) {
            if (BoundaryCondition[j] == GHOST_OR_DUMMY) continue;
            sumOfTerms += CoefficientMatrix[i * n + j] * Pressure[j];
        }
        Pressure[i] = (SourceTerm[i] - sumOfTerms) / CoefficientMatrix[i * n + i];
    }
}


void removeNegativePressure(void) {
    int i;

    for (i = 0; i < NumberOfParticles; i++) {
        if (Pressure[i] < 0.0)Pressure[i] = 0.0;
    }
}


void setMinimumPressure(void) {
    double xij, yij, zij, distance2;
    int i, j;

    for (i = 0; i < NumberOfParticles; i++) {
        if (ParticleType[i] == GHOST || ParticleType[i] == DUMMY_WALL)continue;
        MinimumPressure[i] = Pressure[i];
        for (j = 0; j < NumberOfParticles; j++) {
            if ((j == i) || (ParticleType[j] == GHOST)) continue;
            if (ParticleType[j] == DUMMY_WALL) continue;
            xij = PositionX[j] - PositionX[i];
            yij = PositionY[j] - PositionY[i];
            zij = PositionZ[j] - PositionZ[i];
            distance2 = (xij * xij) + (yij * yij) + (zij * zij);
            if (distance2 >= Re2_forGradient)continue;
            if (MinimumPressure[i] > Pressure[j]) {
                MinimumPressure[i] = Pressure[j];
            }
        }
    }
}


void calculatePressureGradient(void) {
    int    i, j;
    double gradient_x, gradient_y, gradient_z;
    double xij, yij, zij;
    double distance, distance2;
    double w, pij;
    double a;

    a = DIM / N0_forGradient;
    for (i = 0; i < NumberOfParticles; i++) {
        if (ParticleType[i] != FLUID) continue;
        gradient_x = 0.0;  gradient_y = 0.0;  gradient_z = 0.0;
        for (j = 0; j < NumberOfParticles; j++) {
            if (j == i) continue;
            if (ParticleType[j] == GHOST) continue;
            if (ParticleType[j] == DUMMY_WALL) continue;
            xij = PositionX[j] - PositionX[i];
            yij = PositionY[j] - PositionY[i];
            zij = PositionZ[j] - PositionZ[i];
            distance2 = (xij * xij) + (yij * yij) + (zij * zij);
            distance = sqrt(distance2);
            if (distance < Re_forGradient) {
                w = weight(distance, Re_forGradient);
                pij = (Pressure[j] - MinimumPressure[i]) / distance2;
                gradient_x += xij * pij * w;
                gradient_y += yij * pij * w;
                gradient_z += zij * pij * w;
            }
        }
        gradient_x *= a;
        gradient_y *= a;
        gradient_z *= a;
        AccelerationX[i] = (-1.0) * gradient_x / FluidDensity;
        AccelerationY[i] = (-1.0) * gradient_y / FluidDensity;
        AccelerationZ[i] = (-1.0) * gradient_z / FluidDensity;
    }
}


void moveParticleUsingPressureGradient(void) {
    int i;

    for (i = 0; i < NumberOfParticles; i++) {
        if (ParticleType[i] == FLUID) {
            VelocityX[i] += AccelerationX[i] * DT;
            VelocityY[i] += AccelerationY[i] * DT;
            VelocityZ[i] += AccelerationZ[i] * DT;

            PositionX[i] += AccelerationX[i] * DT * DT;
            PositionY[i] += AccelerationY[i] * DT * DT;
            PositionZ[i] += AccelerationZ[i] * DT * DT;
        }
        AccelerationX[i] = 0.0;
        AccelerationY[i] = 0.0;
        AccelerationZ[i] = 0.0;
    }
}


void writeData_inProfFormat(void) {
    int i;
    FILE* fp;
    char fileName[256];

    sprintf(fileName, "output_%04d.prof", FileNumber);
    fp = fopen(fileName, "w");
    fprintf(fp, "%lf\n", Time);
    fprintf(fp, "%d\n", NumberOfParticles);
    for (i = 0; i < NumberOfParticles; i++) {
        fprintf(fp, "%d %lf %lf %lf %lf %lf %lf %lf %lf\n"
            , ParticleType[i], PositionX[i], PositionY[i], PositionZ[i]
            , VelocityX[i], VelocityY[i], VelocityZ[i], Pressure[i], ParticleNumberDensity[i]);
    }
    fclose(fp);
    FileNumber++;
}


void writeData_inVtuFormat(void) {
    int i;
    double absoluteValueOfVelocity;
    FILE* fp;
    char fileName[1024];

    sprintf(fileName, "particle_%04d.vtu", FileNumber);
    fp = fopen(fileName, "w");
    fprintf(fp, "<?xml version='1.0' encoding='UTF-8'?>\n");
    fprintf(fp, "<VTKFile xmlns='VTK' byte_order='LittleEndian' version='0.1' type='UnstructuredGrid'>\n");
    fprintf(fp, "<UnstructuredGrid>\n");
    fprintf(fp, "<Piece NumberOfCells='%d' NumberOfPoints='%d'>\n", NumberOfParticles, NumberOfParticles);
    fprintf(fp, "<Points>\n");
    fprintf(fp, "<DataArray NumberOfComponents='3' type='Float32' Name='Position' format='ascii'>\n");
    for (i = 0; i < NumberOfParticles; i++) {
        fprintf(fp, "%lf %lf %lf\n", PositionX[i], PositionY[i], PositionZ[i]);
    }
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "</Points>\n");
    fprintf(fp, "<PointData>\n");
    fprintf(fp, "<DataArray NumberOfComponents='1' type='Int32' Name='ParticleType' format='ascii'>\n");
    for (i = 0; i < NumberOfParticles; i++) {
        fprintf(fp, "%d\n", ParticleType[i]);
    }
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "<DataArray NumberOfComponents='1' type='Float32' Name='Velocity' format='ascii'>\n");
    for (i = 0; i < NumberOfParticles; i++) {
        absoluteValueOfVelocity =
            sqrt(VelocityX[i] * VelocityX[i] + VelocityY[i] * VelocityY[i] + VelocityZ[i] * VelocityZ[i]);
        fprintf(fp, "%f\n", (float)absoluteValueOfVelocity);
    }
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "<DataArray NumberOfComponents='1' type='Float32' Name='Pressure' format='ascii'>\n");
    for (i = 0; i < NumberOfParticles; i++) {
        fprintf(fp, "%f\n", (float)Pressure[i]);
    }
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "<DataArray NumberOfComponents='1' type='Float32' Name='ParticleNumberDensity' format='ascii'>\n");
    for (i = 0; i < NumberOfParticles; i++) {
        fprintf(fp, "%f\n", (float)ParticleNumberDensity[i]);
    }
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "</PointData>\n");
    fprintf(fp, "<Cells>\n");
    fprintf(fp, "<DataArray type='Int32' Name='connectivity' format='ascii'>\n");
    for (i = 0; i < NumberOfParticles; i++) {
        fprintf(fp, "%d\n", i);
    }
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "<DataArray type='Int32' Name='offsets' format='ascii'>\n");
    for (i = 0; i < NumberOfParticles; i++) {
        fprintf(fp, "%d\n", i + 1);
    }
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "<DataArray type='UInt8' Name='types' format='ascii'>\n");
    for (i = 0; i < NumberOfParticles; i++) {
        fprintf(fp, "1\n");
    }
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "</Cells>\n");
    fprintf(fp, "</Piece>\n");
    fprintf(fp, "</UnstructuredGrid>\n");
    fprintf(fp, "</VTKFile>\n");
    fclose(fp);
}






double dotProduct(double* a, double* b) {


    /*std::vector<double> d(NumberOfParticles);
    std::vector<double> e(NumberOfParticles);
    for (int i = 0; i < NumberOfParticles; ++i) {
        d[i] = a[i];
        e[i] = b[i];
    }*/


    int n = NumberOfParticles;
    double c = 0.0;
    for (int i = 0; i < n; ++i) {
        //if (AI[i] == -1) continue;
        c += a[i] * b[i];

    }
    return c;
}

double* matrixVectorProduct(double* a, double* b)
{

    int n = NumberOfParticles;
    double* c = new double[n];


    /*std::vector<double> e(n), f(n);
    for (int i = 0; i < n; ++i) {
        e[i] = a[i];
        f[i] = b[i];
    }*/


    for (int i = 0; i < n; ++i) {
        c[i] = 0.0;
    }

    for (int i = 0; i < n; ++i) {
        if (AI[i] == -1) continue;
        double temp = 0.0;
        for (int j = i + 1; j <= n; ++j) {
            if (AI[j] == -1) continue;
            for (int k = AI[i]; k < AI[j]; ++k) {
                int m = AJ[k];
                temp += a[k] * b[m];
            }
            c[i] = temp;
            //i = j;
            break;
        }
    }

    /*std::vector<double> d(n);
    for (int i = 0; i < n; ++i) {
        d[i] = c[i];
    }*/
    return c;
}

double norm2(double* a) {

    int n = NumberOfParticles;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        if (AI[i] == -1) continue;

        sum += a[i] * a[i];

    }
    return sqrt(sum);
}


void conjugateGradient(void)
{
    int n = NumberOfParticles;


    for (int i = 0; i < n; ++i) {
        Pressure[i] = 0.0;
        r[i] = 0.0;
        d[i] = 0.0;
        oldr[i] = 0.0;
    }

    double temp;
    for (int i = 0; i < n; ++i) {
        if (AI[i] == -1) continue;

        temp = 0.0;

        for (int j = i + 1; j <= n; ++j) {
            if (AI[j] == -1) continue;

            for (int k = AI[i]; k < AI[j]; ++k) {
                int m = AJ[k];

                temp += AV[k] * Pressure[m];
            }
            break;
        }


        r[i] = SourceTerm[i] - temp;
        d[i] = r[i];
    }


    static int iii;

    for (int iii = 0; iii < n; ++iii) { //循环，不是遍历粒子


        double* product = matrixVectorProduct(AV, d);

        std::vector<double> e(n);

        for (int i = 0; i < n; ++i) {
            e[i] = product[i];
        }

        double a = dotProduct(r, r);
        double b = dotProduct(d, product);
        double alpha = a / b;


        for (int i = 0; i < NumberOfParticles; ++i) {  //遍历粒子
            if (AI[i] == -1) continue;

            oldr[i] = r[i];
            Pressure[i] = Pressure[i] + alpha * d[i];
            r[i] = r[i] - alpha * product[i];
        }


        double l = norm2(r);
        double belta = dotProduct(r, r) / dotProduct(oldr, oldr);

        if (norm2(r) <= 0.001)
            break;


        for (int i = 0; i < NumberOfParticles; ++i) {
            if (AI[i] == -1) continue;
            d[i] = r[i] + belta * d[i];

        }

        delete product;
 //std::cout << "the cycle times of Poisson equation solving  " << iii << std::endl;
    }




}


//double* preconditionedConjugateGradient(double A[ARRAY_SIZE * ARRAY_SIZE], double b[ARRAY_SIZE])
//{
//	int n = NumberOfParticles;
//	double* x = new double[ARRAY_SIZE];
//
//	double x0[ARRAY_SIZE];
//	double r0[ARRAY_SIZE];
//	double d0[ARRAY_SIZE];
//	//double z0[ARRAY_SIZE];
//
//	double r[ARRAY_SIZE];
//	double d[ARRAY_SIZE];
//	double z[ARRAY_SIZE];
//
//
//	double temp[ARRAY_SIZE];
//	double ChoM[ARRAY_SIZE * ARRAY_SIZE];
//	choleskyDecomposition(A, L);
//	incompleteCholeskyFactorization(L, ChoM);
//	matrixProduct(L, M);
//
//	for (int i = 0; i < NumberOfParticles; ++i) {
//		x0[i] = 0.0;
//	}
//
//
//	for (int i = 0; i < NumberOfParticles; ++i) {
//		temp[i] = 0.0;
//		for (int j = 0; j < NumberOfParticles; ++j) {
//			temp[i] += A[i * NumberOfParticles + j] * x0[j];
//		}
//		r0[i] = b[i] - temp[i];
//	}
//
//	double* z0 = gaussianelimination(M, r0);
//	for (int i = 0; i < NumberOfParticles; ++i) {
//		d0[i] = z0[i];
//
//		r[i] = r0[i];
//		d[i] = d0[i];
//		x[i] = x0[i];
//		z[i] = z0[i];
//	}
//
//	for (int i = 0; i < NumberOfParticles; ++i) {
//		if (norm2(r) <= 0.01)
//			break;
//
//		double* product = matrixVectorProduct(A, d);
//		double a = dotProduct(r, z);
//		double b = dotProduct(d, product);
//
//		double alpha = dotProduct(r, z) / (dotProduct(d, product));
//
//
//		double oldr[ARRAY_SIZE];
//		double oldz[ARRAY_SIZE];
//		for (int i = 0; i < NumberOfParticles; ++i) {
//			oldr[i] = r[i];
//			oldz[i] = z[i];
//			x[i] = x[i] + alpha * d[i];
//			r[i] = r[i] - alpha * product[i];
//
//		}
//
//
//		double* zPointer = gaussianelimination(M, r);
//		for (int i = 0; i < n; ++i) {
//			z[i] = zPointer[i];
//		}
//
//		double belta = dotProduct(r, z) / dotProduct(oldr, oldz);
//
//		for (int i = 0; i < NumberOfParticles; ++i) {
//			d[i] = z[i] + belta * d[i];
//
//		}
//
//		delete product;
//		delete zPointer;
//
//	}
//
//	return x;
//
//}


double* gaussianelimination(double AA[ARRAY_SIZE * ARRAY_SIZE], double bb[ARRAY_SIZE]) {

    double A[ARRAY_SIZE * ARRAY_SIZE];
    double b[ARRAY_SIZE];
    int n = NumberOfParticles;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = AA[i * n + j];
        }
        b[i] = bb[i];
    }


    for (int i = 0; i < n; ++i) {
        if (A[i * n + i] == 0) {
            break;
        }

        for (int j = i + 1; j < n; ++j) {
            double lambda = A[j * n + i] / A[i * n + i];

            for (int k = i; k < n; ++k) {
                A[j * n + k] = A[j * n + k] - lambda * A[i * n + k];
            }

            b[j] = b[j] - lambda * b[i];
        }
    }

    double* x = new double[ARRAY_SIZE];
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += A[i * n + j] * x[j];
        }

        x[i] = (b[i] - sum) / A[i * n + i];
    }


    return x;
}


//std::vector<double> choleskyFactorization() {
//    std::vector<double> choleskyR;
//    int n = NumberOfParticles;
//    for(int k = 0; k < n; ++k) {
//        if(AI[k] == -1) continue;
//        choleskyR = sqrt(AI)
//    }






//}


//void incompleteCholeskyFactorization() {

//    int n = NumberOfParticles;
//    for (int i = 0; i < n; ++i) {
//        for (int j = i; j < n; ++j) {
//            if (A[i * n + j] == 0) {
//                L[i * n + j] = 0;
//            }
//        }
//    }
//}


void tranposeMatrix(double L[ARRAY_SIZE * ARRAY_SIZE], double R[ARRAY_SIZE * ARRAY_SIZE]) {
    int n = NumberOfParticles;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {

            R[i * n + j] = L[j * n + i];

        }
    }
}




void matrixProduct(double L[ARRAY_SIZE * ARRAY_SIZE], double M[ARRAY_SIZE * ARRAY_SIZE]) {
    int n = NumberOfParticles;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            M[i * n + j] = 0.0;
        }
    }

    double R[ARRAY_SIZE * ARRAY_SIZE];
    tranposeMatrix(L, R);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                M[i * n + j] += L[i * n + k] * R[k * n + j];
            }

        }

    }
}


void buildParticlesInBuckets(void) {
    int n = NumberOfParticles;
    buckets.clear();

    buckets.resize(n);
    for (int i = 0; i < n; ++i) {
        size_t key = getHashKeyFromPosition(i);
        buckets[key].push_back(i);
    }
}

size_t getHashKeyFromPosition(int i) {
    std::vector<int> bucketIndex = getBucketIndex(i);
    return getHashKeyFromBucketIndex(bucketIndex);
}

std::vector<int> getBucketIndex(int i) {
    double diameter = 2 * searchRadius;
    int bucketIndex_X = static_cast<int>(std::floor(PositionX[i] / diameter));
    int bucketIndex_Y = static_cast<int>(std::floor(PositionY[i] / diameter));
    int bucketIndex_Z = static_cast<int>(std::floor(PositionZ[i] / diameter));

    std::vector<int> bucketIndex;
    bucketIndex.push_back(bucketIndex_X);
    bucketIndex.push_back(bucketIndex_Y);
    bucketIndex.push_back(bucketIndex_Z);

    return bucketIndex;
}


size_t getHashKeyFromBucketIndex(std::vector<int> bucketIndex) {
    double diameter = 2 * searchRadius;

    int M = std::floor(LENGTH / diameter + 1);
    int N = std::floor(WIDTH / diameter + 1);
    int Q = std::floor(HEIGHT / diameter + 1);

    int wrappedIndex_X = bucketIndex[0] % M ;
    int wrappedIndex_Y = bucketIndex[1] % N;
    int wrappedIndex_Z = bucketIndex[2] % Q;

    if (wrappedIndex_X < 0) { wrappedIndex_X += M; }
    if (wrappedIndex_Y < 0) { wrappedIndex_Y += N; }
    if (wrappedIndex_Z < 0) { wrappedIndex_Z += Q; }

    return static_cast<size_t>((wrappedIndex_Z * N + wrappedIndex_Y) * M
         + wrappedIndex_X);
}

void forEachNearbyPoint(int originIndex) {
    
    std::vector<size_t> nearbyKeys(8);
    nearbyKeys = getNearbyKeys(originIndex);
    //neighborList.resize(NumberOfParticles);



    const double queryRadiusSquared = searchRadius * searchRadius;

    if (DIM == 2) {
        for (int i = 0; i < 4; i++) {

        const auto& bucket = buckets[nearbyKeys[i]];
        size_t numberOfPointsInBucket = bucket.size();

        for (size_t j = 0; j < numberOfPointsInBucket; ++j) {
            size_t pointIndex = bucket[j];

            double xij = PositionX[pointIndex] - PositionX[originIndex];
            double yij = PositionY[pointIndex] - PositionY[originIndex];
            


            double rSquared = xij * xij + yij * yij;
            if (rSquared <= queryRadiusSquared) {
                if (originIndex != pointIndex)
                neighborList[originIndex].push_back(pointIndex);
            }
        }
    }

    }

    else {
    for (int i = 0; i < 8; i++) {

        const auto& bucket = buckets[nearbyKeys[i]];
        size_t numberOfPointsInBucket = bucket.size();

        for (size_t j = 0; j < numberOfPointsInBucket; ++j) {
            size_t pointIndex = bucket[j];

            double xij = PositionX[pointIndex] - PositionX[originIndex];
            double yij = PositionY[pointIndex] - PositionY[originIndex];
            double zij = PositionZ[pointIndex] - PositionZ[originIndex];


            double rSquared = xij * xij + yij * yij + zij * zij;
            if (rSquared <= queryRadiusSquared) {
                if (originIndex != pointIndex)
                neighborList[originIndex].push_back(pointIndex);
            }
        }
    }
    }
}


std::vector<size_t> getNearbyKeys(int originIndex){
    double diameter = 2 * PARTICLE_DISTANCE;

    std::vector<size_t> nearbyKeys(8);
    std::vector<int> nearbyBucketIndices[8];

    std::vector<int> bucketIndex = getBucketIndex(originIndex);

    for (int i = 0; i < 8; i++) {
        nearbyBucketIndices[i] = bucketIndex;
    }

    if (DIM == 2) {
        if ((bucketIndex[0]+ 0.5f + 0.5) * diameter <= PositionX[originIndex]) {
            nearbyBucketIndices[1][0] += 1;
            nearbyBucketIndices[2][0] += 1;
        } else {
            nearbyBucketIndices[1][0] -= 1;
            nearbyBucketIndices[2][0] -= 1;
        }

        if ((bucketIndex[1]+ 0.5f + 0.5) * diameter <= PositionY[originIndex]) {
            nearbyBucketIndices[2][1] += 1;
            nearbyBucketIndices[3][1] += 1;
        } else {
            nearbyBucketIndices[2][1] -= 1;
            nearbyBucketIndices[3][1] -= 1;
        }

    } else {

    if ((bucketIndex[0]+ 0.5f) * diameter <= PositionX[originIndex]) {
        nearbyBucketIndices[4][0] += 1; nearbyBucketIndices[5][0] += 1;
        nearbyBucketIndices[6][0] += 1; nearbyBucketIndices[7][0] += 1;
    } else {
        nearbyBucketIndices[4][0] -= 1; nearbyBucketIndices[5][0] -= 1;
        nearbyBucketIndices[6][0] -= 1; nearbyBucketIndices[7][0] -= 1;
    }

    if ((bucketIndex[1] + 0.5f) * diameter <= PositionY[originIndex]) {
        nearbyBucketIndices[2][1] += 1; nearbyBucketIndices[3][1] += 1;
        nearbyBucketIndices[6][1] += 1; nearbyBucketIndices[7][1] += 1;
     } else {
        nearbyBucketIndices[2][1] -= 1; nearbyBucketIndices[3][1] -= 1;
        nearbyBucketIndices[6][1] -= 1; nearbyBucketIndices[7][1] -= 1;
     }

    if ((bucketIndex[2] + 0.5f) * diameter <= PositionZ[originIndex]) {
        nearbyBucketIndices[1][2] += 1; nearbyBucketIndices[3][2] += 1;
        nearbyBucketIndices[5][2] += 1; nearbyBucketIndices[7][2] += 1;
     } else {
        nearbyBucketIndices[1][2] -= 1; nearbyBucketIndices[3][2] -= 1;
        nearbyBucketIndices[5][2] -= 1; nearbyBucketIndices[7][2] -= 1;
     }
    }

    for (size_t i = 0; i < 8; i++) {
        std::vector<int> bucketIndex = nearbyBucketIndices[i];
        nearbyKeys[i] = getHashKeyFromBucketIndex(bucketIndex);
    }

    return nearbyKeys;
}

void buildNeighborList(void) {
    int n = NumberOfParticles;
    neighborList.clear();
    neighborList.resize(n);
    for(int i = 0; i < n; ++i) {
        forEachNearbyPoint(i);
    }
}
