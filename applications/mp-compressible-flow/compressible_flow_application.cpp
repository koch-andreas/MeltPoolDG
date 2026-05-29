#include "compressible_flow_application.hpp"

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/compressible_flow/cutdg_operation.hpp>
#include <meltpooldg/compressible_flow/dg_operation.hpp>
#include <meltpooldg/compressible_flow/operation_type_erasure.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/cell_monitor.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "compressible_flow_case.hpp"

namespace MeltPoolDG::CompressibleFlow
{
  using namespace dealii;

  template <int dim, typename number>
  void
  Application<dim, number>::run()
  {
    initialize();

    // Print startup info on the solver and the output the initial condition if output is requested.
    const auto print_solver_startup_info = [&]() {
      // helpers
      auto &pcout = scratch_data->get_pcout(1);

      auto print_decoration = [&]() { Journal::print_decoration_line(pcout); };

      auto print_line = [&](const std::string &text = "") { Journal::print_line(pcout, text); };

      auto print_line_with_decoration = [&](const std::string &text = "") {
        Journal::print_line_with_decoration(pcout, text);
      };

      auto print_kv = [&](const std::string &key, const std::string &value) {
        std::ostringstream oss;
        oss << " " << std::left << std::setw(16) << key << ": " << value;
        print_line(oss.str());
      };

      // header
      std::ostringstream title;
      title << std::string(30, ' ') << "MeltPoolDG: Compressible Flow Solver";
      print_line_with_decoration(title.str());

      // case info
      print_line();
      print_kv("Case", simulation_case->parameters.base.case_name);
      print_kv("Dimension", std::to_string(dim) + "D");

      const bool is_cut = (simulation_case->parameters.flow.domain_representation_type == "cut");
      print_kv("Method", is_cut ? "cut Discontinuous Galerkin" : "Discontinuous Galerkin");

      print_kv("MPI Processes",
               std::to_string(
                 dealii::Utilities::MPI::n_mpi_processes(scratch_data->get_mpi_comm())));

      print_line();
      print_decoration();

      // initial solution
      if (simulation_case->parameters.output.do_user_defined_postprocessing)
        {
          print_decoration();
          print_line(" Initialization Summary");
          print_decoration();
          output_results(time_iterator->get_current_time_step_number(),
                         time_iterator->get_current_time(),
                         true);
          print_decoration();
        }
      pcout << std::endl;
    };

    print_solver_startup_info();

    Journal::print_line_with_decoration(scratch_data->get_pcout(1), " Starting simulation...");

    while (not time_iterator->is_finished())
      {
        // update level-set for cutDG
        if (level_set_field_function)
          compute_level_set();

        // use CFL condition to compute time step size if required
        if (simulation_case->parameters.flow.do_cfl_time_stepping)
          time_iterator->set_current_time_increment(comp_flow_operation.compute_time_step_size(),
                                                    std::numeric_limits<number>::max());

        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout(1));

        comp_flow_operation.solve(time_iterator->get_current_time(),
                                  time_iterator->get_current_time_increment());

        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time(),
                       time_iterator->is_finished());

        if (profiling_monitor and profiling_monitor->now())
          {
            profiling_monitor->print(scratch_data->get_pcout(1),
                                     scratch_data->get_timer(),
                                     scratch_data->get_mpi_comm());
          }
      }

    // always print timing statistics
    if (profiling_monitor)
      {
        profiling_monitor->print(scratch_data->get_pcout(1),
                                 scratch_data->get_timer(),
                                 scratch_data->get_mpi_comm());
      }
    Journal::print_end(scratch_data->get_pcout(1));
  }

  template <int dim, typename number>
  void
  Application<dim, number>::compute_level_set()
  {
    // set the current time to the field function
    level_set_field_function->set_time(time_iterator->get_current_time());

    // interpolate the values of the given field function to the dof vector values
    level_set.zero_out_ghost_values();
    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     scratch_data->get_dof_handler(level_set_dof_idx),
                                     *level_set_field_function,
                                     level_set);
    level_set.update_ghost_values();
  }

  template <int dim, typename number>
  void
  Application<dim, number>::setup_dof_system()
  {
    // distribute DoFs
    comp_flow_operation.distribute_dofs(dof_handler);

    scratch_data->create_partitioning();

    simulation_case->set_time_boundary_conditions(
      simulation_case->parameters.time_stepping.start_time);

    // create the matrix-free object
    scratch_data->build(true,
                        true,
                        simulation_case->parameters.flow.domain_representation_type == "cut",
                        simulation_case->parameters.flow.fe.degree == 2 and
                          simulation_case->parameters.flow.domain_representation_type == "cut");

    if (simulation_case->parameters.flow.domain_representation_type == "cut")
      {
        // Currently, only homogeneous Cartesian grids without mesh refinements are enabled for
        // cutDG
        AssertThrow(
          std::abs(scratch_data->get_min_cell_size() - scratch_data->get_max_cell_size()) < 1e-10,
          dealii::ExcMessage(
            "Only homogeneous Cartesian grids without local grid refinements are supported!"));
      }

    // print mesh information
    CellMonitor<number>::add_info("compFlow::cells",
                                  scratch_data->get_triangulation().n_global_active_cells(),
                                  scratch_data->get_min_cell_size(),
                                  scratch_data->get_max_cell_size());

    comp_flow_operation.reinit();
  }

  template <int dim, typename number>
  void
  Application<dim, number>::initialize()
  {
    // setup DoFHandler
    dof_handler.reinit(*simulation_case->triangulation);
    if (simulation_case->parameters.flow.domain_representation_type == "cut")
      dof_handler_level_set.reinit(*simulation_case->triangulation);

    // setup scratch data
    {
      scratch_data = std::make_shared<ScratchData<dim, dim, number>>(
        simulation_case->mpi_communicator, simulation_case->parameters.base.verbosity_level, true);
      // set up mapping
      scratch_data->set_mapping(
        FiniteElementUtils::create_mapping<dim>(simulation_case->parameters.flow.fe));

      // create quadrature rule
      comp_flow_quad_idx = scratch_data->attach_quadrature(
        FiniteElementUtils::create_quadrature<dim>(simulation_case->parameters.flow.fe));

      comp_flow_dof_idx = scratch_data->attach_dof_handler(dof_handler);
      scratch_data->attach_constraint_matrix(constraints);
      if (simulation_case->parameters.flow.domain_representation_type == "cut")
        {
          level_set_dof_idx = scratch_data->attach_dof_handler(dof_handler_level_set);
          scratch_data->attach_constraint_matrix(constraints_level_set);
        }
    }

    // set analytical level-set field function
    if (simulation_case->parameters.flow.domain_representation_type == "cut")
      level_set_field_function = simulation_case->get_field_function("level_set",
                                                                     "compressible_flow",
                                                                     false /* is_optional */);

    // initialize the time iterator
    time_iterator = std::make_shared<TimeIntegration::TimeIterator<number>>(
      simulation_case->parameters.time_stepping);

    // initialize compressible flow operation. Choose between domain representation type "fitted"
    // and "cut".
    if (simulation_case->parameters.flow.domain_representation_type == "fitted")
      {
        const auto create_dg_operator = [&]<int n_species>() {
          std::unique_ptr<DGOperation<dim, number, n_species>> operation =
            std::make_unique<DGOperation<dim, number, n_species>>(
              *scratch_data,
              simulation_case->parameters.flow,
              simulation_case->parameters.material,
              comp_flow_dof_idx,
              comp_flow_quad_idx);
          comp_flow_operation = OperationTypeErasure<dim, number>(std::move(operation));
        };

        switch (this->simulation_case->parameters.material.number_of_species)
          {
            case 1:
              create_dg_operator.template operator()<1>();
              break;
            case 2:
              create_dg_operator.template operator()<2>();
              break;
            default:
              AssertThrow(false,
                          dealii::ExcMessage(
                            "Only up to two species are supported in the current implementation!"));
          }
      }
    else if (simulation_case->parameters.flow.domain_representation_type == "cut")
      {
        std::unique_ptr<CutDGOperation<dim, number>> operation =
          std::make_unique<CutDGOperation<dim, number>>(
            *scratch_data,
            simulation_case->parameters.flow,
            simulation_case->parameters.material,
            simulation_case->parameters.cut,
            *time_iterator,
            [this]() { this->setup_dof_system(); },
            level_set,
            comp_flow_dof_idx,
            level_set_dof_idx,
            comp_flow_quad_idx);

        // prescribe a function for the object velocity in the case of a moving immersed (rigid)
        // object
        std::shared_ptr<Function<dim>> unfitted_object_velocity_function;
        unfitted_object_velocity_function =
          simulation_case->get_field_function("unfitted_object_velocity",
                                              "compressible_flow",
                                              true /* is_optional */);
        if (unfitted_object_velocity_function)
          operation->set_unfitted_object_velocity(unfitted_object_velocity_function);

        // set inflow function in the case of an unfitted inflow boundary
        std::shared_ptr<Function<dim>> unfitted_inflow_function;
        unfitted_inflow_function = simulation_case->get_field_function(
          "unfitted_inflow",
          "compressible_flow",
          not(simulation_case->parameters.cut.unfitted_flow_boundary_condition ==
              "inflow") /* is_optional */);
        if (unfitted_inflow_function)
          operation->set_inflow_field_unfitted_boundary(unfitted_inflow_function);

        comp_flow_operation = OperationTypeErasure<dim, number>(std::move(operation));
      }
    else
      DEAL_II_NOT_IMPLEMENTED();

    // set up level-set for cutDG
    if (simulation_case->parameters.flow.domain_representation_type == "cut")
      {
        // currently, we use a continuous level-set field with same element degree as the flow field
        const FE_Q<dim> fe_level_set(simulation_case->parameters.flow.fe.degree);
        dof_handler_level_set.distribute_dofs(fe_level_set);

        Assert(level_set_field_function != nullptr, ExcInternalError());

        level_set.reinit(dof_handler_level_set.locally_owned_dofs(),
                         DoFTools::extract_locally_relevant_dofs(dof_handler_level_set),
                         dof_handler_level_set.get_mpi_communicator());
        compute_level_set();
      }

    setup_dof_system();

    // set boundary conditions
    const std::string operation_name = "compressible_flow";
    comp_flow_operation.set_boundary_conditions(simulation_case, operation_name);

    // set initial condition for the flow field
    comp_flow_operation.set_initial_condition(
      *simulation_case->get_initial_condition("compressible_flow"));

    // set body force
    if (dim > 1 and simulation_case->parameters.flow.gravity_constant > 0.)
      {
        std::unique_ptr<dealii::Functions::ConstantFunction<dim>> body_force =
          std::make_unique<dealii::Functions::ConstantFunction<dim>>(
            dim > 2 ?
              std::vector<number>({0., 0., -simulation_case->parameters.flow.gravity_constant}) :
              std::vector<number>({0., -simulation_case->parameters.flow.gravity_constant}));
        comp_flow_operation.set_body_force(std::move(body_force));
      }

    // initialize postprocessor
    post_processor =
      std::make_unique<Postprocessor<dim, number>>(scratch_data->get_mpi_comm(comp_flow_dof_idx),
                                                   simulation_case->parameters.output,
                                                   simulation_case->parameters.time_stepping,
                                                   scratch_data->get_mapping(),
                                                   scratch_data->get_triangulation(
                                                     comp_flow_dof_idx),
                                                   scratch_data->get_pcout(2));

    // initialize profiling
    if (simulation_case->parameters.profiling.enable)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<number>>(simulation_case->parameters.profiling,
                                                              *time_iterator);
  }

  template <int dim, typename number>
  void
  Application<dim, number>::output_results(const unsigned int time_step,
                                           const number       current_time,
                                           const bool         force_output)
  {
    if (not post_processor->is_output_timestep(time_step, current_time) and
        not simulation_case->parameters.output.do_user_defined_postprocessing and not force_output)
      return;

    if (simulation_case->parameters.flow.eigenvalues_data.do_output or
        simulation_case->parameters.flow.eigenvalues_data.print_summary)
      {
        const auto eigenvalues = comp_flow_operation.estimate_jacobian_eigenvalues(
          time_iterator->get_current_time_increment());

        if (simulation_case->parameters.flow.eigenvalues_data.do_output)
          {
            post_processor->output_complex_valued_vector_to_csv(
              time_step,
              current_time,
              eigenvalues,
              simulation_case->parameters.flow.eigenvalues_data.output_filename,
              force_output);
          };
      }

    const auto attach_output_vectors = [&](GenericDataOut<dim, number> &data_out) {
      comp_flow_operation.attach_output_vectors(data_out);
    };

    GenericDataOut<dim, number> generic_data_out(
      scratch_data->get_mapping(),
      current_time,
      simulation_case->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    if (level_set_field_function)
      generic_data_out.add_data_vector(dof_handler_level_set, level_set, "level_set");

    // user-defined postprocessing
    if (simulation_case->parameters.output.do_user_defined_postprocessing and
        (post_processor->is_output_timestep(time_step, current_time) or force_output))
      simulation_case->do_postprocessing(generic_data_out);

    // postprocessing
    post_processor->process(time_step, generic_data_out, current_time, force_output);
  }

  template class Application<1, double>;
  template class Application<2, double>;
  template class Application<3, double>;

} // namespace MeltPoolDG::CompressibleFlow

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);
  MeltPoolDG::default_main<MeltPoolDG::CompressibleFlow::CaseParameters<double>,
                           MeltPoolDG::CompressibleFlow::Case,
                           MeltPoolDG::CompressibleFlow::Application>(argc, argv, mpi_comm);
  return 0;
}
