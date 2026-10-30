#include <meltpooldg/core/parameters_base.hpp>

#include <iostream>

#include "../narrow_band_case.hpp"

int
main()
{
  dealii::ParameterHandler                               prm;
  MeltPoolDG::LevelSet::NarrowBandCaseParameters<double> parameters;
  parameters.print_parameters(prm, std::cout, true /*print_details*/);
  return 0;
}
