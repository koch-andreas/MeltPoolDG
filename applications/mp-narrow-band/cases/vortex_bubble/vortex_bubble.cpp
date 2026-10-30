#include "vortex_bubble.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::NarrowBandVortexBubble
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::NarrowBandCase,
                           SimulationVortexBubble,
                           "vortex_bubble",
                           2,
                           double);
} // namespace MeltPoolDG::Simulation::NarrowBandVortexBubble
