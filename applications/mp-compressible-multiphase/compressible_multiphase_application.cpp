#include "compressible_multiphase_application.hpp"

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/compressible_flow/multiphase_operation.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/cell_monitor.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>

#include <string>

#include "compressible_multiphase_case.hpp"

namespace MeltPoolDG::Multiphase
{
  using namespace dealii;

  template <int dim, typename number>
  void
  CompressibleMultiphaseApplication<dim, number>::run()
  {
    initialize();

    // output initial condition
    output_results(time_iterator->get_current_time_step_number(),
                   time_iterator->get_current_time());

    while (not time_iterator->is_finished())
      {
        // use CFL condition to compute time step size if required
        if (simulation_case->parameters.flow.do_cfl_time_stepping)
          time_iterator->set_current_time_increment(
            comp_multiphase_operation.compute_time_step_size(), std::numeric_limits<number>::max());

        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout(1));

        comp_multiphase_operation.solve(time_iterator->get_current_time(),
                                        time_iterator->get_current_time_increment());

        // do level-set advection and reinitialization
        update_level_set(time_iterator->get_current_time_increment());

        // do output if requested
        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time());

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
  CompressibleMultiphaseApplication<dim, number>::interpolate_initial_level_set()
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
  CompressibleMultiphaseApplication<dim, number>::update_level_set(const number &time_step)
  {
    // TODO: introduce dof_handler for level-set advection velocity field, compute projection of
    // interface velocity in normal direction of the interface to reduce the distortion of the
    // level-set field

    // TODO: use advection_DG_operation and reinitialization_..._operation for dim>2 cases with
    // distorted level-set.

    // move level-set analytically (currently, only 1D simulations are supported)
    /*AssertThrow(dim == 1,
                dealii::ExcNotImplemented(
                  "Currently, interface movement is only allowed for 1D simulations."));

    level_set_advection_operator.move_level_set(level_set,
                                                time_step,
                                                simulation_case->parameters.base.case_name,
                                                scratch_data,
                                                level_set_dof_idx);*/
  }

  template <int dim, typename number>
  void
  CompressibleMultiphaseApplication<dim, number>::setup_dof_system()
  {
    comp_multiphase_operation.distribute_dofs(dof_handler);

    scratch_data->create_partitioning();

    simulation_case->set_time_boundary_conditions(
      simulation_case->parameters.time_stepping.start_time);

    // create the matrix-free object
    scratch_data->build(true, true, true, simulation_case->parameters.flow.fe.degree == 2);

    // Currently, only homogeneous Cartesian grids without mesh refinements are enabled for
    // cutDG
    AssertThrow(
      std::abs(scratch_data->get_min_cell_size() - scratch_data->get_max_cell_size()) < 1e-10,
      dealii::ExcMessage(
        "Only homogeneous Cartesian grids without local grid refinements are supported!"));

    // print mesh information
    CellMonitor<number>::add_info("compFlow::cells",
                                  scratch_data->get_triangulation().n_global_active_cells(),
                                  scratch_data->get_min_cell_size(),
                                  scratch_data->get_max_cell_size());

    comp_multiphase_operation.reinit();
  }

  template <int dim, typename number>
  void
  CompressibleMultiphaseApplication<dim, number>::initialize()
  {
    // setup DoFHandler
    dof_handler.reinit(*simulation_case->triangulation);
    dof_handler_level_set.reinit(*simulation_case->triangulation);

    // setup scratch data
    {
      scratch_data = std::make_shared<ScratchData<dim, dim, number>>(
        simulation_case->mpi_communicator, simulation_case->parameters.base.verbosity_level, true);
      // set up mapping
      scratch_data->set_mapping(
        FiniteElementUtils::create_mapping<dim>(simulation_case->parameters.flow.fe));

      // create quadrature rule
      comp_multiphase_quad_idx = scratch_data->attach_quadrature(
        FiniteElementUtils::create_quadrature<dim>(simulation_case->parameters.flow.fe));

      comp_multiphase_dof_idx = scratch_data->attach_dof_handler(dof_handler);
      scratch_data->attach_constraint_matrix(constraints);

      level_set_dof_idx = scratch_data->attach_dof_handler(dof_handler_level_set);
      scratch_data->attach_constraint_matrix(constraints_level_set);
    }

    // set analytical level-set field function
    level_set_field_function = simulation_case->get_field_function("level_set",
                                                                   "compressible_multiphase_flow",
                                                                   false /* is_optional */);

    // initialize the time iterator
    time_iterator = std::make_shared<TimeIntegration::TimeIterator<number>>(
      simulation_case->parameters.time_stepping);

    // initialize compressible multiphase operation
    std::unique_ptr<CompressibleMultiphaseOperation<dim, number>> operation =
      std::make_unique<CompressibleMultiphaseOperation<dim, number>>(
        *scratch_data,
        simulation_case->parameters.flow,
        simulation_case->parameters.material_gas,
        simulation_case->parameters.material_liquid,
        simulation_case->parameters.phase_change,
        simulation_case->parameters.cut,
        simulation_case->parameters.phase_coupling,
        simulation_case->parameters.darcy_damping,
        *time_iterator,
        [this]() { this->setup_dof_system(); },
        level_set,
        comp_multiphase_dof_idx,
        level_set_dof_idx,
        comp_multiphase_quad_idx);

    comp_multiphase_operation =
      MeltPoolDG::CompressibleFlow::OperationTypeErasure<dim, number>(std::move(operation));

    // set up level-set for cutDG
    // currently, we use a continuous level-set field with same element degree as the flow field
    const FE_Q<dim> fe_level_set(simulation_case->parameters.flow.fe.degree);
    dof_handler_level_set.distribute_dofs(fe_level_set);

    level_set.reinit(dof_handler_level_set.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler_level_set),
                     dof_handler_level_set.get_mpi_communicator());
    interpolate_initial_level_set();

    setup_dof_system();

    // set boundary conditions
    const std::string operation_name = "compressible_multiphase_flow";
    comp_multiphase_operation.set_boundary_conditions(simulation_case, operation_name);

    // set initial condition for the flow field
    comp_multiphase_operation.set_initial_condition(
      *simulation_case->get_initial_condition("compressible_multiphase_flow"));

    // set body force
    if (dim > 1 and simulation_case->parameters.flow.gravity_constant > 0.)
      {
        std::unique_ptr<Functions::ConstantFunction<dim>> body_force =
          std::make_unique<Functions::ConstantFunction<dim>>(
            dim > 2 ?
              std::vector<number>({0., 0., -simulation_case->parameters.flow.gravity_constant}) :
              std::vector<number>({0., -simulation_case->parameters.flow.gravity_constant}));
        comp_multiphase_operation.set_body_force(std::move(body_force));
      }

    // initialize postprocessor
    post_processor = std::make_unique<Postprocessor<dim, number>>(
      scratch_data->get_mpi_comm(comp_multiphase_dof_idx),
      simulation_case->parameters.output,
      simulation_case->parameters.time_stepping,
      scratch_data->get_mapping(),
      scratch_data->get_triangulation(comp_multiphase_dof_idx),
      scratch_data->get_pcout(2));

    // initialize profiling
    if (simulation_case->parameters.profiling.enable)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<number>>(simulation_case->parameters.profiling,
                                                              *time_iterator);
  }

  template <int dim, typename number>
  void
  CompressibleMultiphaseApplication<dim, number>::output_results(const unsigned int time_step,
                                                                 const number      &current_time)
  {
    if (not post_processor->is_output_timestep(time_step, current_time) and
        not simulation_case->parameters.output.do_user_defined_postprocessing)
      return;

    const auto attach_output_vectors = [&](GenericDataOut<dim, number> &data_out) {
      comp_multiphase_operation.attach_output_vectors(data_out);
    };

    GenericDataOut<dim, number> generic_data_out(
      scratch_data->get_mapping(),
      current_time,
      simulation_case->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    generic_data_out.add_data_vector(dof_handler_level_set, level_set, "level_set");

    // user-defined postprocessing
    if (simulation_case->parameters.output.do_user_defined_postprocessing and
        post_processor->is_output_timestep(time_step, current_time))
      simulation_case->do_postprocessing(generic_data_out);

    // postprocessing
    post_processor->process(time_step, generic_data_out, current_time);
  }

  template class CompressibleMultiphaseApplication<1, double>;
  template class CompressibleMultiphaseApplication<2, double>;
  template class CompressibleMultiphaseApplication<3, double>;

} // namespace MeltPoolDG::Multiphase

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);
  MeltPoolDG::default_main<MeltPoolDG::Multiphase::CompressibleMultiphaseCaseParameters<double>,
                           MeltPoolDG::Multiphase::CompressibleMultiphaseCase,
                           MeltPoolDG::Multiphase::CompressibleMultiphaseApplication>(argc,
                                                                                      argv,
                                                                                      mpi_comm);
  return 0;
}
