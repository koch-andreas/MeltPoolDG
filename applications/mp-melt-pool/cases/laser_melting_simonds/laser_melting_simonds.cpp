#include <meltpooldg/core/case_registration.hpp>

#include "../../melt_pool_case.hpp"
#include "laser_melting_simonds.templates.hpp"

namespace MeltPoolDG::Simulation::LaserMeltingSimonds
{
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolCase,
                                     SimulationLaserMeltingSimonds,
                                     "laser_melting_simonds",
                                     1,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolCase,
                                     SimulationLaserMeltingSimonds,
                                     "laser_melting_simonds",
                                     2,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolCase,
                                     SimulationLaserMeltingSimonds,
                                     "laser_melting_simonds",
                                     3,
                                     double);
} // namespace MeltPoolDG::Simulation::LaserMeltingSimonds
