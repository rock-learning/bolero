#include "FifthOrderPolynomial.h"
#include <assert.h>
#include <vector>
#include <cmath>

using namespace std;
using namespace Eigen;
namespace dmp {


FifthOrderPolynomial::FifthOrderPolynomial() : startTime(0.0), endTime(0.0)
{}


void FifthOrderPolynomial::setConstraints(const VectorXd& gt, const VectorXd& gdt, const VectorXd& gddt,
                                          const VectorXd& gf, const VectorXd& gdf, const VectorXd& gddf,
                                          const double startTime, const double endTime) {
  //all dimensions should be equal
  assert(gdt.size() == gt.size());
  assert(gddt.size() == gt.size());
  assert(gf.size() == gt.size());
  assert(gdf.size() == gt.size());
  assert(gddf.size() == gt.size());
  assert(startTime < endTime);

  this->startTime = startTime;
  this->endTime = endTime;
  goalPos = gf;
  goalVel = gdf;
  goalAcc = gddf;

  //rename startTime and endTime to make formulas readable :)
  const double t = startTime;
  const double f = endTime;

  const double t2 = std::pow(t, 2);
  const double t3 = std::pow(t, 3);
  const double t4 = std::pow(t, 4);
  const double t5 = std::pow(t, 5);
  const double f2 = std::pow(f, 2);
  const double f3 = std::pow(f, 3);
  const double f4 = std::pow(f, 4);
  const double f5 = std::pow(f, 5);

  Mtype M;
  M << 1,   t,      t2,       t3,        t4,        t5,
       0,   1,   2 * t,   3 * t2,    4 * t3,    5 * t4,
       0,   0,       2,    6 * t,   12 * t2,   20 * t3,
       1,   f,      f2,       f3,        f4,        f5,
       0,   1,   2 * f,   3 * f2,    4 * f3,    5 * f4,
       0,   0,       2,    6 * f,   12 * f2,   20 * f3;

  //Solve M*b = y for b in each DOF separately
  PartialPivLU<Mtype> luOfM(M);
  coefficients.resize(gt.size());
  for(int i = 0; i < gt.size(); ++i) {
    Vector6d x;
    x << gt[i], gdt[i], gddt[i], gf[i], gdf[i], gddf[i];
    coefficients[i] = luOfM.solve(x);
  }
}

void FifthOrderPolynomial::getValueAt(const double t, ArrayXd& outPosition, ArrayXd& outvelocity,
                                      ArrayXd& outAcceleration) const {
  assert(outPosition.size() == outvelocity.size());
  assert(outPosition.size() == outAcceleration.size());
  assert(outPosition.size() > 0);
  assert(t >= startTime);

  if(t > endTime)
  {
    /**For times > endTime the polynomial should always 'pull' to the goal position.
     * But velocity and acceleration should be zero.
     * This is done to avoid diverging from the goal if the dmp is executed
     * longer than expected. */
    outPosition = goalPos;
    outvelocity.setZero();
    outAcceleration.setZero();
  }
  else
  {
    Matrix<double, 1, 6> pos;
    Matrix<double, 1, 6> vel;
    Matrix<double, 1, 6> acc;
    const double t2 = std::pow(t, 2);
    const double t3 = std::pow(t, 3);
    const double t4 = std::pow(t, 4);
    const double t5 = std::pow(t, 5);
    pos << 1,   t,      t2,       t3,        t4,        t5;
    vel << 0,   1,   2 * t,   3 * t2,    4 * t3,    5 * t4;
    acc << 0,   0,       2,    6 * t,   12 * t2,   20 * t3;

    //for each DOF
    for(int i = 0; i < outPosition.size(); ++i) {
      outPosition[i] = pos * coefficients[i];
      outvelocity[i] = vel * coefficients[i];
      outAcceleration[i] = acc * coefficients[i];
    }
  }
}

}//end namespace
