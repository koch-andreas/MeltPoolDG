#  include "../../../mp-radiative-transport/cases/radiative_transport.templates.hpp"
//
#  include <meltpooldg/core/case_registration.hpp>

#  include "../../heat_transfer_case.hpp"

namespace MeltPoolDG::Simulation::RadiativeTransport
{
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolDG::Heat::HeatTransferCase,
                                     SimulationRadTrans,
                                     "radiative_transport",
                                     1,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolDG::Heat::HeatTransferCase,
                                     SimulationRadTrans,
                                     "radiative_transport",
                                     2,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(MeltPoolDG::Heat::HeatTransferCase,
                                     SimulationRadTrans,
                                     "radiative_transport",
                                     3,
                                     double);
} // namespace MeltPoolDG::Simulation::RadiativeTransport
