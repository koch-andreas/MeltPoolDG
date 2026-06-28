#include "evaporating_droplet_circular_domain.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::CompressibleMultiphase
{
  MELTPOOLDG_REGISTER_CASE(Multiphase::CompressibleMultiphaseCase,
                           SimulationEvaporatingDropletCircularDomain,
                           "evaporating_droplet_circular_domain",
                           2,
                           double);

  MELTPOOLDG_REGISTER_CASE(Multiphase::CompressibleMultiphaseCase,
                           SimulationEvaporatingDropletCircularDomain,
                           "evaporating_droplet_circular_domain",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::CompressibleMultiphase
