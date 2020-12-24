#' \pkg{ovml}
#'
#' Image and video machine learning tools, for application to volleyball analytics.
#'
#' @name ovml
#' @docType package
#' @importFrom assertthat assert_that is.flag is.string
#' @importFrom graphics lines rasterImage text
#' @importFrom magick geometry_size_pixels image_data image_extent image_info image_read image_scale
#' @importFrom magrittr `%>%`
#' @importFrom stats setNames
#' @importFrom torch cuda_is_available nnf_interpolate nnf_pad nnf_softplus nn_batch_norm2d nn_conv2d nn_leaky_relu nn_max_pool2d nn_module nn_module_list nn_sequential torch_arange torch_cat torch_device torch_meshgrid torch_tanh torch_tensor
#' @importFrom utils download.file tail

NULL
