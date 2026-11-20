#include "narrow_band_application.hpp"

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/numerics/data_out.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

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

    bool stop = false;
    while (not time_iterator->is_finished() and not stop)
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

        // restart: the decisions of the root process (wall-clock times) are taken by all
        // processes
        if (restart_monitor and param.restart.save >= 0)
          {
            const MPI_Comm mpi_comm = scratch_data->get_mpi_comm();
            bool           save     = restart_monitor->do_save();
            bool           wall_time_reached =
              param.max_wall_time > 0. and elapsed_wall_time() >= param.max_wall_time;
            save              = dealii::Utilities::MPI::broadcast(mpi_comm, save, 0);
            wall_time_reached = dealii::Utilities::MPI::broadcast(mpi_comm, wall_time_reached, 0);

            if (save or wall_time_reached or time_iterator->is_finished())
              save_restart();

            if (wall_time_reached and not time_iterator->is_finished())
              {
                std::ostringstream str;
                str << "maximum wall time of " << param.max_wall_time
                    << " s reached: the simulation stops at t = " << std::scientific
                    << std::setprecision(4) << time_iterator->get_current_time()
                    << " (continue with restart/load = 0)";
                Journal::print_line(scratch_data->get_pcout(0), str.str(), "restart");
                stop = true;
              }
          }

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

    // restart: mesh of the saved state (before the DoFHandlers are created)
    const bool do_load = param.restart.load >= 0;
    if (do_load)
      load_triangulation();

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

    post_processor = std::make_unique<Postprocessor<dim, number>>(scratch_data->get_mpi_comm(),
                                                                  param.output,
                                                                  param.time_stepping,
                                                                  scratch_data->get_mapping(),
                                                                  *simulation_case->triangulation,
                                                                  scratch_data->get_pcout(2));

    if (do_load)
      load_state();
    else
      {
        const auto &initial_condition = *simulation_case->get_initial_condition("level_set");
        narrow_band_operation->set_initial_condition(
          initial_condition,
          simulation_case->reinitialize_initial_condition(),
          simulation_case->get_cellwise_initial_condition());

        // output of the initial state
        output_results(0, param.time_stepping.start_time, false);
      }

    // restart: the intervals of the restart files start at the initial or the restored time
    if (param.restart.save >= 0 or do_load)
      restart_monitor =
        std::make_unique<Restart::RestartMonitor<number>>(param.restart, *time_iterator);

    if (param.profiling.verbosity != Profiling::ProfilingVerbosity::none)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<number>>(param.profiling, *time_iterator);
  }

  template <int dim, typename number>
  void
  NarrowBandApplication<dim, number>::save_restart()
  {
    ScopedName                 sc("save_restart");
    dealii::TimerOutput::Scope scope(scratch_data->get_timer(), sc);

    namespace fs            = std::filesystem;
    const MPI_Comm mpi_comm = scratch_data->get_mpi_comm();
    const bool     is_root  = dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0;

    // keep the previous states: the files i are copied to i + 1 (root process)
    restart_monitor->prepare_save();
    MPI_Barrier(mpi_comm);

    const std::string prefix = param.restart.prefix + "_0";

    // state, which is the same on all processes (written to a temporary file, which replaces
    // the file of the previous state once all files have been written)
    if (is_root)
      {
        std::ofstream file(prefix + "_problem.restart.tmp", std::ios::binary);
        AssertThrow(file,
                    dealii::ExcMessage("The restart file " + prefix +
                                       "_problem.restart.tmp could not be opened."));
        boost::archive::binary_oarchive archive(file);
        archive << static_cast<const TimeIntegration::TimeIterator<number> &>(*time_iterator);
        archive << static_cast<const Postprocessor<dim, number> &>(*post_processor);
        archive << static_cast<const std::vector<std::pair<double, std::string>> &>(
          stage_times_and_names);
        archive << static_cast<const NarrowBandLevelSetOperation<dim, number> &>(
          *narrow_band_operation);
      }

    // mesh and cell-wise state of the narrow band (all processes)
    narrow_band_operation->prepare_for_serialization();
    simulation_case->triangulation->save(prefix + "_tria");

    MPI_Barrier(mpi_comm);
    if (is_root)
      fs::rename(prefix + "_problem.restart.tmp", prefix + "_problem.restart");
    MPI_Barrier(mpi_comm);

    std::ostringstream str;
    str << "write restart files " << prefix << "_* (t = " << std::scientific << std::setprecision(4)
        << time_iterator->get_current_time() << ", time step "
        << time_iterator->get_current_time_step_number() << ")";
    Journal::print_line(scratch_data->get_pcout(0), str.str(), "restart");
  }

  template <int dim, typename number>
  void
  NarrowBandApplication<dim, number>::load_triangulation()
  {
    ScopedName                 sc("load_restart");
    dealii::TimerOutput::Scope scope(scratch_data->get_timer(), sc);

    namespace fs             = std::filesystem;
    const std::string prefix = param.restart.prefix + "_" + std::to_string(param.restart.load);
    AssertThrow(fs::exists(prefix + "_problem.restart") and fs::exists(prefix + "_tria"),
                dealii::ExcMessage("The restart files " + prefix + "_* do not exist."));

    // Triangulation::load() requires the coarse mesh: the refinement of the case is removed
    auto &triangulation = *simulation_case->triangulation;
    for (unsigned int n = 0; triangulation.n_global_levels() > 1; ++n)
      {
        AssertThrow(n < 64,
                    dealii::ExcMessage("The mesh of the case could not be coarsened to its "
                                       "coarse mesh before loading the restart files."));
        for (const auto &cell : triangulation.active_cell_iterators())
          if (cell->is_locally_owned() and cell->level() > 0)
            cell->set_coarsen_flag();
        triangulation.execute_coarsening_and_refinement();
      }

    triangulation.load(prefix + "_tria");

    Journal::print_line(scratch_data->get_pcout(0),
                        "load restart files " + prefix + "_*",
                        "restart");
  }

  template <int dim, typename number>
  void
  NarrowBandApplication<dim, number>::load_state()
  {
    ScopedName                 sc("load_restart");
    dealii::TimerOutput::Scope scope(scratch_data->get_timer(), sc);

    const std::string prefix = param.restart.prefix + "_" + std::to_string(param.restart.load);

    // state, which is the same on all processes (read by all processes)
    {
      std::ifstream file(prefix + "_problem.restart", std::ios::binary);
      AssertThrow(file,
                  dealii::ExcMessage("The restart file " + prefix +
                                     "_problem.restart could not be opened."));
      boost::archive::binary_iarchive archive(file);
      archive >> *time_iterator;
      archive >> *post_processor;
      archive >> stage_times_and_names;
      archive >> *narrow_band_operation;
    }

    // cell-wise state of the narrow band
    narrow_band_operation->deserialize();

    std::ostringstream str;
    str << "restart at t = " << std::scientific << std::setprecision(4)
        << time_iterator->get_current_time() << " (time step "
        << time_iterator->get_current_time_step_number() << ")";
    Journal::print_line(scratch_data->get_pcout(0), str.str(), "restart");
  }

  template <int dim, typename number>
  double
  NarrowBandApplication<dim, number>::elapsed_wall_time() const
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_time_start)
      .count();
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

    // The postprocessor checks the output interval again; thus, the forced output (after the last
    // time step) has to be passed on. The paraview files are only written if enabled.
    post_processor->process(time_step, generic_data_out, time, force_output);
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
