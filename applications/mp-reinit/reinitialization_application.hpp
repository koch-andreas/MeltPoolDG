#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/level_set/reinitialization_operation_base.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>

#include <memory>
#include <string>

#include "reinitialization_case.hpp"


namespace MeltPoolDG
{
  namespace LevelSet
  {
    template <int dim, typename number>
    class ReinitializationApplication
    {
    private:
      using CaseType       = ReinitializationCase<dim, number>;
      using VectorType     = dealii::LinearAlgebra::distributed::Vector<number>;
      using DoFHandlerType = dealii::DoFHandler<dim>;

    public:
      ReinitializationApplication(std::unique_ptr<CaseType> simulation_case_)
        : simulation_case(std::move(simulation_case_))
        , param(simulation_case->parameters)
      {}

      void
      run();

    private:
      std::unique_ptr<CaseType>                            simulation_case;
      const ReinitializationCaseParameters<number>        &param;
      DoFHandlerType                                       dof_handler;
      dealii::AffineConstraints<number>                    constraints;
      dealii::AffineConstraints<number>                    normal_no_bc_constraints;
      dealii::AffineConstraints<number>                    normal_dirichlet_x_constraints;
      dealii::AffineConstraints<number>                    normal_dirichlet_y_constraints;
      dealii::AffineConstraints<number>                    normal_dirichlet_z_constraints;
      std::unique_ptr<Profiling::ProfilingMonitor<number>> profiling_monitor;

      std::shared_ptr<ScratchData<dim, dim, number>>              scratch_data;
      std::unique_ptr<TimeIntegration::TimeIterator<number>>      time_iterator;
      std::unique_ptr<ReinitializationOperationBase<dim, number>> reinit_operation;
      unsigned int                                                reinit_dof_idx             = -1;
      unsigned int                                                normal_no_bc_dof_idx       = -1;
      unsigned int                                                normal_dirichlet_x_dof_idx = -1;
      unsigned int                                                normal_dirichlet_y_dof_idx = -1;
      unsigned int                                                normal_dirichlet_z_dof_idx = -1;
      unsigned int                                                reinit_quad_idx            = -1;

      std::unique_ptr<Postprocessor<dim, number>> post_processor;

      void
      initialize();

      void
      setup_dof_system();

      /**
       * Distribute the DoFs for a level-set field, which is only defined on the cells intersected
       * by the zero level-set iso-surface of the initial condition: FE_DGQ on the intersected
       * cells and FE_Nothing on all other cells (cell categories according to
       * CutUtil::CellCategory).
       */
      void
      distribute_dofs_on_intersected_cells();

      void
      refine_mesh();

      void
      output_results(const unsigned int time_step, const number current_time);
    };
  } // namespace LevelSet
} // namespace MeltPoolDG
