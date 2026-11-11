#include "narrow_band_application.hpp"

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/numerics/data_out.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  void
  NarrowBandApplication<dim, number>::run()
  {
    initialize();

    ScopedName                 sc("run");
    dealii::TimerOutput::Scope scope(scratch_data->get_timer(), sc);

    while (not time_iterator->is_finished())
      {
        const number time_step = time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout(1));

        // time at the beginning of the time step
        const number time = time_iterator->get_current_time() - time_step;

        const number cfl_number = narrow_band_operation->compute_cfl_number(time, time_step);
        {
          std::ostringstream str;
          str << "CFL number (2p+1) max|u| dt / h: " << std::scientific << std::setprecision(4)
              << cfl_number;
          Journal::print_line(scratch_data->get_pcout(1), str.str(), "narrow_band");
        }
        if (cfl_number > 1.0)
          {
            std::ostringstream str;
            str << "Warning: CFL number " << std::scientific << std::setprecision(4) << cfl_number
                << " > 1. The interface might leave the narrow band.";
            Journal::print_line(scratch_data->get_pcout(0), str.str(), "narrow_band");
          }

        // output of the stages of the advection of the time steps, which start in the time window
        if (param.stage_output.enable)
          {
            const unsigned int time_step_number = time_iterator->get_current_time_step_number();
            const number       tolerance        = 1e-10 * time_step;
            if (time >= param.stage_output.start_time - tolerance and
                time <= param.stage_output.end_time + tolerance)
              narrow_band_operation->set_stage_output_function(
                [this, time_step_number](const unsigned int stage,
                                         const number       stage_time,
                                         const number       stage_time_step,
                                         const auto        &level_set,
                                         const auto        &level_set_band) {
                  output_stage(time_step_number,
                               stage,
                               stage_time,
                               stage_time_step,
                               level_set,
                               level_set_band);
                });
            else
              narrow_band_operation->set_stage_output_function({});
          }

        narrow_band_operation->advance_time_step(time,
                                                 time_step,
                                                 time_iterator->get_current_time_step_number());

        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time(),
                       time_iterator->is_finished());

        if (profiling_monitor and profiling_monitor->now())
          profiling_monitor->print(scratch_data->get_pcout(1),
                                   scratch_data->get_timer(),
                                   scratch_data->get_mpi_comm());
      }

    if (profiling_monitor)
      profiling_monitor->print(scratch_data->get_pcout(1),
                               scratch_data->get_timer(),
                               scratch_data->get_mpi_comm());

    Journal::print_end(scratch_data->get_pcout(1));
  }

  template <int dim, typename number>
  void
  NarrowBandApplication<dim, number>::initialize()
  {
    scratch_data =
      std::make_shared<ScratchData<dim, dim, number>>(simulation_case->mpi_communicator,
                                                      param.base.verbosity_level,
                                                      true /*do_matrix_free*/);

    ScopedName                 sc("initialize");
    dealii::TimerOutput::Scope scope(scratch_data->get_timer(), sc);

    scratch_data->set_mapping(FiniteElementUtils::create_mapping<dim>(param.base.fe));

    time_iterator = std::make_unique<TimeIntegration::TimeIterator<number>>(param.time_stepping);

    // The narrow-band operation attaches its DoFHandlers and its quadrature rule to the scratch
    // data.
    narrow_band_operation = std::make_unique<NarrowBandLevelSetOperation<dim, number>>(
      *scratch_data,
      *simulation_case->triangulation,
      param.narrow_band,
      param.reinit,
      param.base.fe.degree,
      simulation_case->get_field_function("prescribed_velocity", "level_set"),
      &param.normal_vector,
      &param.curvature);

    const auto &initial_condition = *simulation_case->get_initial_condition("level_set");
    narrow_band_operation->set_initial_condition(initial_condition,
                                                 false /*do_reinitialize*/,
                                                 simulation_case->get_cellwise_initial_condition());

    post_processor = std::make_unique<Postprocessor<dim, number>>(scratch_data->get_mpi_comm(),
                                                                  param.output,
                                                                  param.time_stepping,
                                                                  scratch_data->get_mapping(),
                                                                  *simulation_case->triangulation,
                                                                  scratch_data->get_pcout(2));

    // output of the initial state
    output_results(0, param.time_stepping.start_time, false);

    if (param.profiling.verbosity != Profiling::ProfilingVerbosity::none)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<number>>(param.profiling, *time_iterator);
  }

  template <int dim, typename number>
  void
  NarrowBandApplication<dim, number>::output_results(const unsigned int time_step,
                                                     const number       time,
                                                     const bool         force_output)
  {
    if (not post_processor->is_output_timestep(time_step, time) and not force_output)
      return;

    GenericDataOut<dim, number> generic_data_out(scratch_data->get_mapping(),
                                                 time,
                                                 param.output.output_variables);
    narrow_band_operation->attach_output_vectors(generic_data_out);

    if (param.output.do_user_defined_postprocessing)
      simulation_case->do_postprocessing(generic_data_out);

    post_processor->process(time_step, generic_data_out, time);
  }

  template <int dim, typename number>
  void
  NarrowBandApplication<dim, number>::output_stage(
    const unsigned int                                                   time_step_number,
    const unsigned int                                                   stage,
    const number                                                         time,
    const number                                                         time_step,
    const typename NarrowBandLevelSetOperation<dim, number>::VectorType &level_set,
    const typename NarrowBandLevelSetOperation<dim, number>::VectorType &level_set_band)
  {
    namespace fs         = std::filesystem;
    const auto &paraview = param.output.paraview;

    dealii::DataOut<dim> data_out;
    data_out.add_data_vector(narrow_band_operation->get_dof_handler_cut(), level_set, "level_set");
    data_out.add_data_vector(narrow_band_operation->get_dof_handler_band(),
                             level_set_band,
                             "level_set_band");

    dealii::DataOutBase::VtkFlags flags;
    if (dim > 1)
      flags.write_higher_order_cells = paraview.write_higher_order_cells;
    data_out.set_flags(flags);

    data_out.build_patches(scratch_data->get_mapping(),
                           paraview.n_patches > 0 ? paraview.n_patches : param.base.fe.degree);

    const std::string filename =
      paraview.filename + "_stages_" +
      dealii::Utilities::int_to_string(time_step_number, paraview.n_digits_timestep) + "_" +
      std::to_string(stage) + ".vtu";
    data_out.write_vtu_in_parallel((fs::path(param.output.directory) / filename).string(),
                                   scratch_data->get_mpi_comm());

    // pvd record: stage s of the time step from t^n to t^n + dt at t^n + s / (s + 1) dt
    stage_times_and_names.emplace_back(time + time_step * stage / (stage + 1.), filename);
    if (dealii::Utilities::MPI::this_mpi_process(scratch_data->get_mpi_comm()) == 0)
      {
        std::ofstream pvd_output(fs::path(param.output.directory) /
                                 (paraview.filename + "_stages.pvd"));
        dealii::DataOutBase::write_pvd_record(pvd_output, stage_times_and_names);
      }

    Journal::print_line(scratch_data->get_pcout(2),
                        "write stage output " + filename,
                        "narrow_band");
  }

  template class NarrowBandApplication<1, double>;
  template class NarrowBandApplication<2, double>;
  template class NarrowBandApplication<3, double>;
} // namespace MeltPoolDG::LevelSet

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);
  MeltPoolDG::default_main<MeltPoolDG::LevelSet::NarrowBandCaseParameters<double>,
                           MeltPoolDG::LevelSet::NarrowBandCase,
                           MeltPoolDG::LevelSet::NarrowBandApplication>(argc, argv, mpi_comm);
  return 0;
}
