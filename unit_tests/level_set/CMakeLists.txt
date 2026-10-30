mpdg_setup_serial_unit_tests(
    TEST_SOURCE_FILES
    reinitialization_elliptic_operator.cpp
)

mpdg_setup_mpi_unit_tests(
    N_MPI_PROCESSES
    3
    TEST_SOURCE_FILES
    narrow_band_level_set.cpp
)
