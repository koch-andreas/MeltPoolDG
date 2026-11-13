#include "zalesak_disk.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::NarrowBandZalesakDisk
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::NarrowBandCase,
                           SimulationZalesakDisk,
                           "zalesak_disk",
                           2,
                           double);
} // namespace MeltPoolDG::Simulation::NarrowBandZalesakDisk
