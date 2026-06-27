#include "evaporating_droplet.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::CompressibleMultiphase
{
  MELTPOOLDG_REGISTER_CASE(Multiphase::CompressibleMultiphaseCase,
                           SimulationEvaporatingDroplet,
                           "evaporating_droplet",
                           1,
                           double);

  MELTPOOLDG_REGISTER_CASE(Multiphase::CompressibleMultiphaseCase,
                           SimulationEvaporatingDroplet,
                           "evaporating_droplet",
                           2,
                           double);

  MELTPOOLDG_REGISTER_CASE(Multiphase::CompressibleMultiphaseCase,
                           SimulationEvaporatingDroplet,
                           "evaporating_droplet",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::CompressibleMultiphase
