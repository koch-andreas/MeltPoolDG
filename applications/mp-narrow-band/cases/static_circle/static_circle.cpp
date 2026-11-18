#include "static_circle.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::StaticCircle
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::NarrowBandCase,
                           SimulationStaticCircle,
                           "static_circle",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::NarrowBandCase,
                           SimulationStaticCircle,
                           "static_circle",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::StaticCircle
