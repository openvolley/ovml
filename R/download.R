#' Path to the cache directory used for model weight files and other data
#'
#' @return The path as a string
#'
#' @export
ovml_cache_dir <- function() {
    out <- rappdirs::user_cache_dir("ovml", "openvolley")
    if (!dir.exists(out)) dir.create(out, recursive = TRUE)
    out
}


ovml_download_if <- function(url, dest) {
    ## dest is basename of destination file
    weights_file <- file.path(ovml_cache_dir(), dest)
    if (!dir.exists(dirname(weights_file))) dir.create(dirname(weights_file), recursive = TRUE)
    if (!file.exists(weights_file)) {
        message("downloading weights from ", url, " to ", weights_file)
        download.file(url, destfile = weights_file, mode = "wb")
    }
    weights_file
}
