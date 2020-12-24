#' Example images provided as part of the ovideo package
#'
#' @param choices integer: which image files to return?
#' - 1 - an image from a match between GKS Katowice and MKS Bedzin during the 2018/19 Polish Plus Liga
#' - 2 - the standard YOLO dog image
#'
#' @return Path to the image files
#'
#' @export
ovml_example_image <- function(choices = 1) {
    assert_that(is.numeric(choices))
    if (!all(choices %in% c(1, 2))) stop("unrecognized choices values: ", setdiff(choices, c(1, 2)))
    out <- rep(NA_character_, length(choices))
    out[choices == 1] <- system.file("extdata/images/2019_03_01-KATS-BEDS-frame.jpg", package = "ovml")
    out[choices == 2] <- system.file("extdata/images/dog.jpg", package = "ovml")
    out
}
