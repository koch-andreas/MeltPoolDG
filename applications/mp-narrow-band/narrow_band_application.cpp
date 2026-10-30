#include "narrow_band_application.hpp"

#include <deal.II/base/mpi.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

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
      simulation_case->get_field_function("prescribed_velocity", "level_set"));

    narrow_band_operation->set_initial_condition(
      *simulation_case->get_initial_condition("level_set"));

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
