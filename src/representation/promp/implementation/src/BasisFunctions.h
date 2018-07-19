#pragma once

#include <Eigen/Dense>
namespace promp
{
using namespace Eigen;

class BasisFunctions
{
public:
  MatrixXd getValue(const VectorXd &time, int dimensions = 1) const;
  MatrixXd getValue(const double time, int dimensions = 1) const;

  MatrixXd getValueDeriv(const VectorXd &time, int dimensions = 1) const;
  MatrixXd getValueDeriv(const double time, int dimensions = 1) const;

  MatrixXd getValueAndDeriv(const VectorXd &time, int dimensions = 1) const;
  MatrixXd getValueAndDeriv(const double time, int dimensions = 1) const;

protected:
  BasisFunctions(const int numBF, const double overlap = 0.5);

  virtual VectorXd calcBasisFunction(const VectorXd &z, const int i) const = 0;
  virtual VectorXd calcBasisFunctionDeriv(const VectorXd &z, const int i) const = 0;

  VectorXd calcNormalizedBasisFunction(const VectorXd &z, const int i) const;
  VectorXd calcNormalizedBasisFunctionDeriv(const VectorXd &z, const int i) const;

  double getMean(int bf) const;

  int numBF_;
  double h_;
};

class StrokeBasisFunctions : public BasisFunctions
{
public:
  StrokeBasisFunctions(const int numBF, const double overlap = 0.5) : BasisFunctions(numBF, overlap){};

private:
  VectorXd calcBasisFunction(const VectorXd &z, const int i) const;
  VectorXd calcBasisFunctionDeriv(const VectorXd &z, const int i) const;
};

class PeriodicBasisFunctions : public BasisFunctions
{
public:
  PeriodicBasisFunctions(const int numBF, const double overlap = 0.5) : BasisFunctions(numBF, overlap){};

private:
  VectorXd calcBasisFunction(const VectorXd &z, const int i) const;
  VectorXd calcBasisFunctionDeriv(const VectorXd &z, const int i) const;
};

}; // namespace promp