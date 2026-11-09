#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/level_set/curvature_data.hpp>
#include <meltpooldg/level_set/normal_vector_data.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>
#include <meltpooldg/utilities/amr_data.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>


namespace MeltPoolDG::LevelSet
{
  template <typename number>
  struct ReinitializationCaseParameters : public ParametersBase
  {
  protected:
    void
    add_parameters(dealii::ParameterHandler &prm) final
    {
      base.add_parameters(prm);
      time_stepping.add_parameters(prm);
      amr.add_parameters(prm);
      reinit.add_parameters(prm);
      normal_vec.add_parameters(prm);
      curv.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);

      prm.enter_subsection("application specific parameters");
      {
        prm.add_parameter(
          "do update normal vector",
          application_specific_parameters.do_update_normal_vector,
          "Set this parameter to true to update the normal vector each pseudo-time step.");
        prm.add_parameter(
          "interface faces only",
          application_specific_parameters.interface_faces_only,
          "Set this parameter to true to evaluate the SIPG face terms of the elliptic "
          "reinitialization on intersected cells only on the faces between two intersected cells, "
          "which are crossed by the interface. Then, interface branches in neighboring cells are "
          "not coupled. By default, the face terms are evaluated on all faces between two "
          "intersected cells.");
      }
      prm.leave_subsection();
    }

    void
    post(const std::string &parameter_filename) final
    {
      amr.post(base.global_refinements, false /*restart not supported*/);
      reinit.post(base.fe);
      normal_vec.post(base.verbosity_level);
      curv.post(base.verbosity_level);
      output.post(time_stepping.time_step_size, parameter_filename);

      // check input parameters for validity
      reinit.check_input_parameters(normal_vec.linear_solver.do_matrix_free);
      normal_vec.check_input_parameters(reinit.interface_thickness_parameter.type);
      curv.check_input_parameters(reinit.interface_thickness_parameter.type);
      profiling.check_input_parameters(time_stepping.time_step_size);

      AssertThrow(not application_specific_parameters.interface_faces_only or
                    (reinit.modeltype == ModelType::elliptic and
                     reinit.elliptic.intersected_cells_only),
                  dealii::ExcMessage("The parameter <interface faces only> requires the elliptic "
                                     "reinitialization on intersected cells only."));
    }

  public:
    BaseData                                  base;
    TimeIntegration::TimeSteppingData<number> time_stepping;
    AdaptiveMeshingData<number>               amr;
    ReinitializationData<number>              reinit;
    NormalVectorData<number>                  normal_vec;
    CurvatureData<number>                     curv;
    OutputData<number>                        output;
    Profiling::ProfilingData<number>          profiling;

    struct
    {
      bool do_update_normal_vector = false;
      bool interface_faces_only    = false;
    } application_specific_parameters;
  };

  template <int dim, typename number>
  class ReinitializationCase : public SimulationCaseBase<dim, number>
  {
  public:
    ReinitializationCaseParameters<number> parameters;

    ReinitializationCase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim, number>(parameter_file_in, mpi_communicator_in)
    {
      dealii::ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }
  };
} // namespace MeltPoolDG::LevelSet
