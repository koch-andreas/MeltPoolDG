#pragma once
// for distributed vectors/matrices
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
// solvers
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h> // only for symmetric matrices
#include <deal.II/lac/solver_gmres.h>

// preconditioner
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG
{
  class LinearSolver
  {
  public:
    template <typename VectorType,
              typename OperatorType       = dealii::TrilinosWrappers::SparseMatrix,
              typename PreconditionerType = dealii::PreconditionIdentity,
              typename number             = double>
    static int
    solve(const OperatorType             &system_matrix,
          VectorType                     &solution,
          const VectorType               &rhs,
          const LinearSolverData<number> &data,
          const PreconditionerType       &preconditioner = dealii::PreconditionIdentity(),
          const std::string               identifier     = "")
    {
      const bool monitor_history = data.monitor_type != LinearSolverMonitorType::none;

      dealii::ReductionControl solver_control(data.max_iterations,
                                              data.abs_tolerance,
                                              data.rel_tolerance);

      if (monitor_history)
        solver_control.enable_history_data();

      const auto finalize = [&](const bool failed_step = false) {
        // TODO: introduce get_mpi_communicator() in BlockVector in deal.II
        std::unique_ptr<dealii::ConditionalOStream> pcout;
        if constexpr (dealii::internal::is_block_vector<VectorType>)
          pcout = std::make_unique<dealii::ConditionalOStream>(
            std::cout,
            dealii::Utilities::MPI::this_mpi_process(solution.block(0).get_mpi_communicator()) ==
              0);
        else
          pcout =
            std::make_unique<dealii::ConditionalOStream>(std::cout,
                                                         dealii::Utilities::MPI::this_mpi_process(
                                                           solution.get_mpi_communicator()) == 0);

        if (monitor_history)
          {
            const auto &history_data = solver_control.get_history_data();

            Journal::print_decoration_line(*pcout);
            {
              std::ostringstream str;
              str << std::setw(5) << "# iter" << std::setw(20) << std::scientific
                  << std::setprecision(5) << "|Res|    ";
              Journal::print_line(*pcout, str.str(), "linear solver");
            }
            Journal::print_decoration_line(*pcout);

            const auto print_value = [&](const unsigned int iteration, const number value) {
              std::ostringstream str;
              str << std::setw(5) << iteration << std::setw(20) << std::scientific
                  << std::setprecision(5) << value;
              Journal::print_line(*pcout, str.str(), "linear solver");
            };

            if (data.monitor_type == LinearSolverMonitorType::all)
              {
                for (unsigned int i = 0; i < history_data.size(); ++i)
                  print_value(i, history_data[i]);
              }
            else if (data.monitor_type == LinearSolverMonitorType::reduced)
              {
                print_value(0, history_data[0]);
                if (history_data.size() >= 2)
                  print_value(history_data.size() - 1, history_data.back());
              }
            else
              {
                AssertThrow(false, dealii::ExcNotImplemented());
              }
            Journal::print_decoration_line(*pcout);
          }

        if (!identifier.empty() && failed_step)
          Journal::print_line(*pcout,
                              "Exception with ID >> " + identifier + " << occured.",
                              "linear solver");
      };


      try
        {
          switch (data.solver_type)
            {
                case (LinearSolverType::CG): {
                  dealii::SolverCG<VectorType> solver(solver_control);

                  solver.connect_condition_number_slot([&](const double condition_number) {
                    Journal::print_line(std::cout,
                                        "Condition number: " + std::to_string(condition_number),
                                        "linear solver");
                  });

                  solver.solve(system_matrix, solution, rhs, preconditioner);
                  break;
                }
                case (LinearSolverType::GMRES): {
                  typename dealii::SolverGMRES<VectorType>::AdditionalData additional_data;
                  additional_data.right_preconditioning = true;

                  dealii::SolverGMRES<VectorType> solver(solver_control, additional_data);
                  solver.solve(system_matrix, solution, rhs, preconditioner);
                  break;
                }
              default:
                AssertThrow(false, dealii::ExcNotImplemented());
            }
        }
      catch (const dealii::SolverControl::NoConvergence &e)
        {
          finalize(true /*failed_step*/);

          AssertThrow(false, e);
        }

      finalize();

      return solver_control.last_step();
    }
  };
} // namespace MeltPoolDG
