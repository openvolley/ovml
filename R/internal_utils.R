str_trim <- function(x) {
    gsub("^[[:space:]]+", "", gsub("[[:space:]]+$", "", x))
}


image_wh <- function(im) {
    if (is.character(im)) im <- image_read(im)
    as.numeric(image_info(im)[, c("width", "height")])
}

image_resz <- function(im, sz, preserve_aspect = TRUE) {
    if (preserve_aspect) {
        out <- image_scale(im, geometry_size_pixels(width = sz, height = sz, preserve_aspect = TRUE))
        image_extent(out, geometry_size_pixels(width = sz, height = sz), color = "#808080")
    } else {
        image_scale(im, geometry_size_pixels(width = sz, height = sz, preserve_aspect = FALSE))
    }
}

