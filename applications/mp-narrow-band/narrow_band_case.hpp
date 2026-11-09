#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/level_set/curvature_data.hpp>
#include <meltpooldg/level_set/narrow_band_data.hpp>
#include <meltpooldg/level_set/narrow_band_level_set_operation.hpp>
#include <meltpooldg/level_set/normal_vector_data.hpp>
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
    NarrowBandCaseParameters()
    {
      // Defaults of the normal vector and the curvature of the continuous level set on the cut
      // cells (narrow band/continuous level set): no damping. The cut cells form a band of one to
      // two cells with a natural boundary at the cell faces; the damping eta = c h^2 acts across
      // the whole band and biases the normal vector and the curvature independently of h (see
      // NarrowBandContinuousLevelSet).
      normal_vector.filter_parameter = 0.;
      curvature.filter_parameter     = 0.;
    }

  protected:
    void
    add_parameters(dealii::ParameterHandler &prm) final
    {
      base.add_parameters(prm);
      time_stepping.add_parameters(prm);
      narrow_band.add_parameters(prm);
      reinit.add_parameters(prm);
      normal_vector.add_parameters(prm);
      curvature.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);
    }

    void
    post(const std::string &parameter_filename) final
    {
      reinit.post(base.fe);
      normal_vector.post(base.verbosity_level);
      curvature.post(base.verbosity_level);
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
    /// Normal vector and curvature of the continuous level set on the cut cells (only the filter
    /// parameters, the linear solvers, the verbosity levels and 'curvature/enable' are used).
    NormalVectorData<number>         normal_vector;
    CurvatureData<number>            curvature;
    OutputData<number>               output;
    Profiling::ProfilingData<number> profiling;
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

    /**
     * @brief Optional initial level-set function, which is evaluated with the cell and is
     * interpolated onto the cut cells instead of the initial condition "level_set" (see
     * NarrowBandLevelSetOperation::set_initial_condition()). By default, it is empty.
     */
    virtual typename NarrowBandLevelSetOperation<dim, number>::CellwiseFunction
    get_cellwise_initial_condition() const
    {
      return {};
    }
  };
} // namespace MeltPoolDG::LevelSet
