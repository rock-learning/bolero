#include <catch/catch.hpp>
#include <Eigen/Core>
#include <Eigen/StdVector>
#define private public //to be able to access private members from the polynome
#include "FifthOrderPolynomial.h"
#include "CanonicalSystem.h"
using namespace dmp;
using namespace Eigen;

/**
 * A Random 5th order polynomial
 */
class RandomPolynomial {
public:
  RandomPolynomial() : coefficients(VectorXd::Random(6)){}

  //f(x)
  double getValue(const double x) {
    double ret = 0.0;
    for(int i = 0; i < 6; ++i) {
      ret += coefficients[i] * std::pow(x, i);
    }
    return ret;
  }

  //f'(x)
  double getFirstDerivation(const double x) {
    double ret = 0.0;
    for(int i = 1; i < 6; ++i) {
      ret += i * coefficients[i] * std::pow(x, i-1);
    }
    return ret;
  }

  //f''(x)
  double getSecondDerivation(const double x) {
    double ret = 0.0;
    for(int i = 2; i < 6; ++i) {
      ret += (std::pow(i,2)-i) * coefficients[i] * std::pow(x, i-2);
    }
    return ret;
  }


private:
  VectorXd coefficients;

};

TEST_CASE("manual", "[FifthOrderPolynomial]") {

  const int numPhases = 100;
  const double tau = 1.0;
  const double lastPhaseValue = 0.01;
  const int taskSpaceDimensions = 4;
  const double startPhase = 1.0;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);
  CanonicalSystem cs(numPhases, tau, alpha);
  FifthOrderPolynomial fop;

  VectorXd gt = VectorXd::Zero(taskSpaceDimensions);
  VectorXd gdt = VectorXd::Zero(taskSpaceDimensions);
  VectorXd gddt = VectorXd::Zero(taskSpaceDimensions);
  VectorXd gf = VectorXd::Ones(taskSpaceDimensions);
  VectorXd gdf = VectorXd::Ones(taskSpaceDimensions);
  VectorXd gddf = VectorXd::Zero(taskSpaceDimensions);
  fop.setConstraints(gt, gdt, gddt, gf, gdf, gddf, cs.getTime(startPhase), tau);

  //these numbers were taken from the original python test
  REQUIRE(fop.coefficients[0][0] == Approx(0));
  REQUIRE(fop.coefficients[0][1] == Approx(0));
  REQUIRE(fop.coefficients[0][2] == Approx(0));
  REQUIRE(fop.coefficients[0][3] == Approx(6));
  REQUIRE(fop.coefficients[0][4] == Approx(-8));
  REQUIRE(fop.coefficients[0][5] == Approx(3));
}

TEST_CASE("match random polynom", "[FifthOrderPolynomial]") {
  /**
   * - Create a RandomPolynomial for each task space dimension.
   * - Initialize the FifthOrderPolynomial with data from RandomPolynomial.
   * - Check if FifthOrderPolynomial and RandomPolynomial return the same result.
   */
  const int numPhases = 100;
  const double tau = 12.0;
  const double lastPhaseValue = 0.01;
  const int taskSpaceDimensions = 4;
  const double startPhase = 1.0;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);
  CanonicalSystem cs(numPhases, tau, alpha);
  FifthOrderPolynomial fop;
  VectorXd gt = VectorXd::Zero(taskSpaceDimensions);
  VectorXd gdt = VectorXd::Zero(taskSpaceDimensions);
  VectorXd gddt = VectorXd::Zero(taskSpaceDimensions);
  VectorXd gf = VectorXd::Zero(taskSpaceDimensions);
  VectorXd gdf = VectorXd::Zero(taskSpaceDimensions);
  VectorXd gddf = VectorXd::Zero(taskSpaceDimensions);
  std::vector<RandomPolynomial> polys;

  polys.resize(taskSpaceDimensions);
  for(int i = 0; i < taskSpaceDimensions; ++i){
    polys[i] = RandomPolynomial();
    gt[i] = polys[i].getValue(0.0);//0.0 is the start time of the canonical system
    gdt[i] = polys[i].getFirstDerivation(0.0);
    gddt[i] = polys[i].getSecondDerivation(0.0);
    gf[i] = polys[i].getValue(tau); //Tau is the end time of the canonical system
    gdf[i] = polys[i].getFirstDerivation(tau);
    gddf[i] = polys[i].getSecondDerivation(tau);
  }
  fop.setConstraints(gt, gdt, gddt, gf, gdf, gddf, cs.getTime(startPhase), tau);

  for(double s = 1.0; s > lastPhaseValue; s -= 0.01) {
    double t = cs.getTime(s);
    ArrayXd pos = ArrayXd::Zero(taskSpaceDimensions);
    ArrayXd vel = ArrayXd::Zero(taskSpaceDimensions);
    ArrayXd acc = ArrayXd::Zero(taskSpaceDimensions);
    fop.getValueAt(t, pos, vel, acc);
    for(int i = 0; i < taskSpaceDimensions; ++i) {
      REQUIRE(pos[i] == Approx(polys[i].getValue(t)));
      REQUIRE(vel[i] == Approx(polys[i].getFirstDerivation(t)));
      REQUIRE(acc[i] == Approx(polys[i].getSecondDerivation(t)));
    }
  }

}
