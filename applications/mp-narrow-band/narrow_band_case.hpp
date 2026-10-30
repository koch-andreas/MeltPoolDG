#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/level_set/narrow_band_data.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>

#include <string>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Parameters of the narrow-band DG level-set application.
   */
  template <typename number>
  struct NarrowBandCaseParameters : public ParametersBase
  {
  protected:
    void
    add_parameters(dealii::ParameterHandler &prm) final
    {
      base.add_parameters(prm);
      time_stepping.add_parameters(prm);
      narrow_band.add_parameters(prm);
      reinit.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);
    }

    void
    post(const std::string &parameter_filename) final
    {
      reinit.post(base.fe);
      output.post(time_stepping.time_step_size, parameter_filename);

      // check input parameters for validity
      AssertThrow(base.fe.type == FiniteElementType::FE_DGQ,
                  dealii::ExcMessage("The narrow-band level set requires FE_DGQ elements."));
      AssertThrow(reinit.modeltype == ModelType::elliptic and
                    reinit.elliptic.intersected_cells_only,
                  dealii::ExcMessage(
                    "The narrow-band level set requires the elliptic reinitialization on "
                    "intersected cells only: set 'reinitialization/type' to 'elliptic' and "
                    "'reinitialization/elliptic/intersected cells only' to true."));
      reinit.check_input_parameters(reinit.linear_solver.do_matrix_free);
      narrow_band.check_input_parameters(base.fe.degree);
      profiling.check_input_parameters(time_stepping.time_step_size);
    }

  public:
    BaseData                                  base;
    TimeIntegration::TimeSteppingData<number> time_stepping;
    NarrowBandData<number>                    narrow_band;
    ReinitializationData<number>              reinit;
    OutputData<number>                        output;
    Profiling::ProfilingData<number>          profiling;
  };

  /**
   * @brief Base class of the simulation cases of the narrow-band DG level-set application.
   *
   * A case has to provide the initial level-set field ("initial_condition", "level_set") and the
   * advection velocity ("prescribed_velocity", "level_set") as field functions.
   */
  template <int dim, typename number>
  class NarrowBandCase : public SimulationCaseBase<dim, number>
  {
  public:
    NarrowBandCaseParameters<number> parameters;

    NarrowBandCase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim, number>(parameter_file_in, mpi_communicator_in)
    {
      dealii::ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }
  };
} // namespace MeltPoolDG::LevelSet
