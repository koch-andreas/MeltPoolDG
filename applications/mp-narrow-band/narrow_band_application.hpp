#pragma once

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/narrow_band_level_set_operation.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>
#include <meltpooldg/utilities/restart.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "narrow_band_case.hpp"

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Application for the narrow-band DG level-set method with a prescribed advection
   * velocity (see NarrowBandLevelSetOperation).
   *
   * Restart (subsection restart, see Restart::RestartData): with 'save' >= 0, the state of the
   * simulation is written every 'write time step size' (wall-clock seconds for 'time type' = real,
   * simulation time for 'time type' = simulation), after the last time step and when the
   * 'max wall time' is exceeded (the simulation then stops). The files
   * <prefix>_<i>_problem.restart (time, output records, state of the narrow band, which is the
   * same on all processes, e.g., the advected interface points) and <prefix>_<i>_tria* (mesh and
   * cell-wise state of the narrow band) are written to the restart directory (default: the
   * output directory); i = 0 is the latest state, older states are kept up to i = 'save' - 1.
   * With 'load' = i >= 0, the simulation continues from the state i. The restarted simulation
   * continues exactly as the simulation, which wrote the state, if the same number of processes
   * and the same parameters are used; the end time, the time step size, the output and the
   * restart parameters may be changed.
   */
  template <int dim, typename number>
  class NarrowBandApplication
  {
  private:
    using CaseType = NarrowBandCase<dim, number>;

  public:
    NarrowBandApplication(std::unique_ptr<CaseType> simulation_case_in)
      : simulation_case(std::move(simulation_case_in))
      , param(simulation_case->parameters)
      , wall_time_start(std::chrono::steady_clock::now())
    {}

    void
    run();

  private:
    void
    initialize();

    /**
     * @brief Write the output (paraview, if enabled) and do the user-defined postprocessing at the
     * output times. With @p force_output, the output is written regardless of the output interval;
     * this is used after the last time step.
     */
    void
    output_results(const unsigned int time_step, const number time, const bool force_output);

    /**
     * @brief Restart: write the state of the simulation to the restart files with the number 0
     * (the files of the previous states are renamed, see Restart::RestartMonitor::prepare_save()).
     */
    void
    save_restart();

    /**
     * @brief Restart: load the triangulation from the restart files with the number 'load'.
     * Called before the narrow-band operation is created.
     */
    void
    load_triangulation();

    /**
     * @brief Restart: read the time, the output records and the state of the narrow band from the
     * restart files with the number 'load' (instead of setting the initial condition).
     */
    void
    load_state();

    /**
     * @brief Elapsed wall-clock time in seconds since the creation of the application (root
     * process).
     */
    double
    elapsed_wall_time() const;

    /**
     * @brief Write the level set at a stage of the advection of the time step @p time_step_number
     * (see NarrowBandStageOutputData).
     */
    void
    output_stage(
      const unsigned int                                                   time_step_number,
      const unsigned int                                                   stage,
      const number                                                         time,
      const number                                                         time_step,
      const typename NarrowBandLevelSetOperation<dim, number>::VectorType &level_set,
      const typename NarrowBandLevelSetOperation<dim, number>::VectorType &level_set_band);

    std::unique_ptr<CaseType>                                 simulation_case;
    const NarrowBandCaseParameters<number>                   &param;
    std::shared_ptr<ScratchData<dim, dim, number>>            scratch_data;
    std::unique_ptr<NarrowBandLevelSetOperation<dim, number>> narrow_band_operation;
    std::unique_ptr<TimeIntegration::TimeIterator<number>>    time_iterator;
    std::unique_ptr<Postprocessor<dim, number>>               post_processor;
    std::unique_ptr<Profiling::ProfilingMonitor<number>>      profiling_monitor;
    std::unique_ptr<Restart::RestartMonitor<number>>          restart_monitor;

    /// Wall-clock time at the creation of the application (maximum wall time of the restart).
    const std::chrono::steady_clock::time_point wall_time_start;

    /// Times and filenames of the written stages of the advection (pvd record).
    std::vector<std::pair<double, std::string>> stage_times_and_names;
  };
} // namespace MeltPoolDG::LevelSet
