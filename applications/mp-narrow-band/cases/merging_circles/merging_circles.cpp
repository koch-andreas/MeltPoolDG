#include "merging_circles.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::MergingCircles
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::NarrowBandCase,
                           SimulationMergingCircles,
                           "merging_circles",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::NarrowBandCase,
                           SimulationMergingCircles,
                           "merging_circles",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::MergingCircles
