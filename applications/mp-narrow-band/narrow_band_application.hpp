#pragma once

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/narrow_band_level_set_operation.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>

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
    {}

    void
    run();

  private:
    void
    initialize();

    void
    output_results(const unsigned int time_step, const number time, const bool force_output);

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

    /// Times and filenames of the written stages of the advection (pvd record).
    std::vector<std::pair<double, std::string>> stage_times_and_names;
  };
} // namespace MeltPoolDG::LevelSet
