#ifndef DMPMODEL_H
#define DMPMODEL_H
#ifndef PRIVATE
#define PRIVATE private
#endif

#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <istream>
#include <iomanip>
#include <math.h>

namespace dmp_cpp{
class DMPModel{
public:
    std::string model_name;
    std::vector<double> rbf_centers;
    std::vector<double> rbf_widths;
    double ts_alpha_z;
    double ts_beta_z; //should be always ts_alpha_z/4.0 for critical damping
    double ts_tau;
    double cs_execution_time; //should always be the same as ts_tau
    double cs_alpha;
    double cs_dt;
    double ts_dt; //ts_dt*num_phases = execution time
    std::vector<std::vector<double> > ft_weights;

    DMPModel(){}

    /**
     * \throw std::runtime_error if model could not be loaded
     */
    DMPModel(const std::string& yamlFile, const std::string& name);

    bool from_yaml_file(std::string filepath, std::string name);
    bool from_yaml_istream(std::istream& stream, std::string name);
    bool from_yaml_string(const std::string& yaml, std::string name);



    /**
     * \return true if the model is consistent, false otherwise
     */
    bool is_valid() const;

    void to_yaml_file(std::string filepath);

    friend std::ostream& operator << (std::ostream& os, DMPModel& val);

    /**
     * \return true if the vector contains at least one number that is NaN or Inf
     */
    bool containsNanOrInf(const std::vector<double>& data) const;
};

template<typename T>
inline std::ostream& operator<<(std::ostream& stm, const std::vector<T>& obj) {
    stm << "[";
    if (!obj.empty()) {
        for (size_t i = 0 ; i<obj.size()-1 ; ++i) {
            stm << obj[i] << ",";
        }
        stm << obj.back();
    }
    stm << "]";
    return stm;
}

inline std::ostream& operator << (std::ostream& os, dmp_cpp::DMPModel& op)
{
    os << "name: " << op.model_name << std::endl;
    os << "rbf_centers: " << op.rbf_centers << std::endl;
    os << "rbf_widths: " << op.rbf_widths << std::endl;
    os << "ts_alpha_z: " << op.ts_alpha_z << std::endl;
    os << "ts_beta_z: " << op.ts_beta_z << std::endl;
    os << "ts_tau: " << op.ts_tau << std::endl;
    os << "ts_dt: " << op.ts_dt << std::endl;
    os << "cs_execution_time: " << op.cs_execution_time << std::endl;
    os << "cs_alpha: " << op.cs_alpha << std::endl;
    os << "cs_dt: " << op.cs_dt << std::endl;
    os << "ft_weights: " << op.ft_weights << std::endl;
    return os;
}
}
#endif

