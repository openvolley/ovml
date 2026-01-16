#' \pkg{ovml}
#'
#' Image and video machine learning tools, for application to volleyball analytics.
#'
#' @name ovml
#' @import ovml.common
#' @importFrom assertthat assert_that is.flag is.string
#' @importFrom magick image_data image_info image_read
#' @importFrom magrittr `%>%`
#' @importFrom stats setNames
#' @importFrom torch cuda_is_available nnf_interpolate nnf_pad nnf_softplus nn_batch_norm2d nn_conv2d nn_leaky_relu nn_max_pool2d nn_module nn_module_list nn_sequential torch_arange torch_cat torch_device torch_meshgrid torch_split torch_tanh torch_tensor
#' @importFrom utils download.file tail
"_PACKAGE"
