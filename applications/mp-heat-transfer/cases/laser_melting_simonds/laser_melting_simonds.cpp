#include <meltpooldg/core/case_registration.hpp>

#include "../../../mp-melt-pool/cases/laser_melting_simonds/laser_melting_simonds.templates.hpp"
#include "../../heat_transfer_case.hpp"

namespace MeltPoolDG::Simulation::LaserMeltingSimonds
{
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(Heat::HeatTransferCase,
                                     SimulationLaserMeltingSimonds,
                                     "laser_melting_simonds",
                                     1,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(Heat::HeatTransferCase,
                                     SimulationLaserMeltingSimonds,
                                     "laser_melting_simonds",
                                     2,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(Heat::HeatTransferCase,
                                     SimulationLaserMeltingSimonds,
                                     "laser_melting_simonds",
                                     3,
                                     double);
} // namespace MeltPoolDG::Simulation::LaserMeltingSimonds
