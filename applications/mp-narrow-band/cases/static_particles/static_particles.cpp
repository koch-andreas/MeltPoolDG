#include "static_particles.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::NarrowBandStaticParticles
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::NarrowBandCase,
                           SimulationStaticParticles,
                           "static_particles",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::NarrowBandCase,
                           SimulationStaticParticles,
                           "static_particles",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::NarrowBandStaticParticles
