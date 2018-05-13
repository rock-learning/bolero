#include "BasisFunctions.h"
//#define M_PI 3.14159265359
using namespace promp;

BasisFunctions::BasisFunctions(const int numBF,const  double overlap) : numBF_(numBF){
     // https://www.wolframalpha.com/input/?i=exp(-((0.5%2Fb)%5E2)+%2F+(2*(-1%2F(8*b%5E2*log(x))))+%3D+x
     h_ = -1 / (8*std::pow(numBF_,2)* std::log(overlap) );  
}

double BasisFunctions::getMean(const int i) const{
  return i/(numBF_- 1.0);
}

VectorXd BasisFunctions::calcNormalizedBasisFunction(const VectorXd& z, const int i) const{
  VectorXd ret(z.size());
  VectorXd sumPhi = VectorXd::Zero(z.size());
  
  for(int j = 0; j < numBF_; j++){
    sumPhi += calcBasisFunction(z,j); 
  }
  ret = calcBasisFunction(z,i);
  ret.array() /= sumPhi.array();
  return ret;
}

VectorXd BasisFunctions::calcNormalizedBasisFunctionDeriv(const VectorXd& z, const int i) const{
  VectorXd ret(z.size());
  VectorXd sumPhi = VectorXd::Zero(z.size());
  VectorXd sumPhiDot = VectorXd::Zero(z.size());
  for(int j = 0; j < numBF_; j++){ 
    ret.array() /= sumPhi.array();   
    sumPhi  += calcBasisFunction(z,j);
    sumPhiDot += calcBasisFunctionDeriv(z,j);
  }
  ret.array() =  ((sumPhi.array() * calcBasisFunctionDeriv(z,i).array()) - (sumPhiDot.array() * calcBasisFunction(z,i).array())) / sumPhi.array().pow(2);
  return ret;
}

VectorXd StrokeBasisFunctions::calcBasisFunction(const VectorXd& z, const int i) const{
  VectorXd ret = z;
  ret.array() -= getMean(i);
  ret.array() = ret.array().pow(2); 
  ret.array() /= (2*h_);
  ret.array() *= -1;
  ret.array() = ret.array().exp();
  return ret;
}

VectorXd StrokeBasisFunctions::calcBasisFunctionDeriv(const VectorXd& z, const int i) const{
  VectorXd ret = z;
  ret.array() =  ((getMean(i) - z.array()) * calcBasisFunction(z,i).array() )/ h_;
  return ret;
}

VectorXd PeriodicBasisFunctions::calcBasisFunction(const VectorXd& z, const int i) const{
  VectorXd ret = z;
  ret.array() = (h_ * (2* M_PI * (z.array()-getMean(i)) ).cos()).exp();
  return ret;
}

VectorXd PeriodicBasisFunctions::calcBasisFunctionDeriv(const VectorXd& z, const int i) const{
  VectorXd ret = z;
  ret.array() =  -2*M_PI * h_ * (2* M_PI * (z.array()-getMean(i))).sin() * calcBasisFunction(z,i).array();
  return ret;
}

MatrixXd BasisFunctions::getValueDeriv(const VectorXd& time,int dimensions) const{
      MatrixXd ret = MatrixXd::Zero(dimensions * time.size(),dimensions * numBF_);
      for( int i = 0; i < numBF_; i++){
        ret.block(0,0,time.size(),numBF_).col(i) = calcNormalizedBasisFunctionDeriv(time,i);
      }
      for( int i = 1; i < dimensions; i++){
	ret.block(i*time.size(),i*numBF_,time.size(),numBF_) = ret.block(0,0,time.size(),numBF_);	
      }
      return ret;
}

MatrixXd BasisFunctions::getValueDeriv(const double time,int dimensions) const{
      VectorXd timeVec(1);
      timeVec << time;
      return getValueDeriv(timeVec);
}

MatrixXd BasisFunctions::getValue(const double time,int dimensions) const{
      Eigen::VectorXd timeVec(1);
      timeVec << time;
      return getValue(timeVec);
}

MatrixXd BasisFunctions::getValue(const VectorXd& time,int dimensions) const{
      Eigen::MatrixXd ret= MatrixXd::Zero(dimensions * time.size(),dimensions * numBF_);
      for( int i = 0; i < numBF_; i++){
	ret.block(0,0,time.size(),numBF_).col(i) = calcNormalizedBasisFunction(time,i); 
      }
      for( int i = 1; i < dimensions; i++){
	ret.block(i*time.size(),i*numBF_,time.size(),numBF_) = ret.block(0,0,time.size(),numBF_);	
      }
      return ret;
}

MatrixXd BasisFunctions::getValueAndDeriv(const VectorXd& time,int dimensions) const{
      Eigen::MatrixXd ret= MatrixXd::Zero(2 * dimensions * time.size(),dimensions * numBF_);
      for( int i = 0; i < numBF_; i++){
	ret.block(0,0,time.size(),numBF_).col(i) = calcNormalizedBasisFunction(time,i);
	ret.block(1,0,time.size(),numBF_).col(i) = calcNormalizedBasisFunctionDeriv(time,i); 
      }
      for( int i = 1; i < dimensions; i++){
	ret.block(i*2*time.size(),i*numBF_,2*time.size(),numBF_) = ret.block(0,0,2*time.size(),numBF_);	
      }    
      return ret;
}

MatrixXd BasisFunctions::getValueAndDeriv(const double time,int dimensions) const{
      VectorXd timeVec(1);
      timeVec << time;
      return getValueDeriv(timeVec).row(0);
}
