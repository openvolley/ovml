library(testthat)
library(ovml)

torch::install_torch()
test_check("ovml")
