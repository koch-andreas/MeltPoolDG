#include "rotating_circle.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::RotatingCircle
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::NarrowBandCase,
                           SimulationRotatingCircle,
                           "rotating_circle",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::NarrowBandCase,
                           SimulationRotatingCircle,
                           "rotating_circle",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::RotatingCircle
