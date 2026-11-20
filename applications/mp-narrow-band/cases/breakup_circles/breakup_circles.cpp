#include "breakup_circles.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::BreakupCircles
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::NarrowBandCase,
                           SimulationBreakupCircles,
                           "breakup_circles",
                           2,
                           double);
} // namespace MeltPoolDG::Simulation::BreakupCircles
