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


ovml_download_if <- function(url, dest, expected_sha1 = NULL) {
    if (length(url) < 1 || is.na(url)) stop("no download url provided")
    ## dest is basename of destination file
    weights_file <- file.path(ovml_cache_dir(), dest)
    if (!dir.exists(dirname(weights_file))) dir.create(dirname(weights_file), recursive = TRUE)
    if (!file.exists(weights_file)) {
        message("downloading weights from ", url, " to ", weights_file)
        curl::curl_download(url, destfile = weights_file, quiet = !interactive())
        if (!check_sha1(weights_file, expected_sha1)) {
            try(unlink(weights_file), silent = TRUE)
            stop("SHA1 of downloaded weights file does not match expected, deleting")
        }
    } else {
        if (!check_sha1(weights_file, expected_sha1)) warning("SHA1 of weights file does not match expected, perhaps you need to re-download it?")
    }
    weights_file
}

check_sha1 <- function(filename, expected = NULL) {
    if (!file.exists(filename)) {
        warning("weights file does not exist")
        FALSE
    } else {
        if (!is.null(expected)) {
            digest::digest(filename, algo = "sha1", file = TRUE) %eq% expected
        } else {
            TRUE
        }
    }
}
