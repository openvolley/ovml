#' Preview plot of detections over image using base graphics
#'
#' @param img string or image: filename of jpg image, or image as read by [[jpeg::readJPEG]]
#' @param detections data.frame: as returned by [ovml_yolo_detect]
#' @param line_args list: parameters passed to [lines] (for [ovml_plot]) or [[ggplot2::geom_rect]] (for [ovml_ggplot])
#' @param label_args list: parameters passed to [text]
#' @param label_geom string: for [ovml_ggplot], the geom function to use for labels. Either "text" (use [[ggplot2::geom_text]]) or "label" ([[ggplot2::geom_label]])
#'
#' @examples
#'
#' ## define some demo data
#' dets <- data.frame(class = rep("person", 3),
#'                    score = rep(0.99, 3),
#'                    xmin = c(829, 611, 736),
#'                    xmax = c(960, 733, 836),
#'                    ymin = c(88, 258, 213),
#'                    ymax = c(278, 444, 385),
#'                    stringsAsFactors = FALSE)
#' img <- system.file("extdata/images/2019_03_01-KATS-BEDS-frame.jpg", package = "ovml")
#' ovml_plot(img, dets, line_args = list(col = "red", lwd = 2))
#' ovml_ggplot(img, dets) + ggplot2::theme_void()
#'
#' @export
ovml_plot <- function(img, detections, line_args = list(col = "blue", lwd = 1), label_args = list(col = "white", cex = 0.75)) {
    if (is.character(img)) img <- jpeg::readJPEG(img)
    plot(c(0, dim(img)[2]), c(0, dim(img)[1]), type = "n", axes = FALSE, xlab = "", ylab = "", asp = 1)
    rasterImage(img, 0, 0, dim(img)[2], dim(img)[1])
    for (i in seq_len(nrow(detections))) {
        do.call(lines, c(list(x = c(detections$xmin[i], rep(detections$xmax[i], 2), rep(detections$xmin[i], 2)), y = c(rep(detections$ymin[i], 2), rep(detections$ymax[i], 2), detections$ymin[i])), line_args))
    }
    do.call(text, c(list(x = (detections$xmin + detections$xmax)/2, y = detections$ymin, labels = detections$class), label_args))
}

#' @export
#' @rdname ovml_plot
ovml_ggplot <- function(img, detections, line_args = list(col = "blue", size = 0.75, fill = NA), label_args = list(col = "white", size = 2.5, fill = "blue"), label_geom = "label") {
    assert_that(is.string(label_geom))
    label_geom <- tolower(label_geom)
    label_geom <- match.arg(label_geom, c("label", "text"))
    if (is.character(img)) img <- jpeg::readJPEG(img, native = TRUE)
    iwh <- dim(img)[c(2, 1)]
    detections$xmid <- (detections$xmin + detections$xmax)/2
    if (label_geom == "text") label_args$fill <- NULL
    ggplot2::ggplot(detections) +
        ggplot2::annotation_custom(grid::rasterGrob(img), xmin = 0, xmax = iwh[1], ymin = 0, ymax = iwh[2]) +
        do.call(ggplot2::geom_rect, c(list(ggplot2::aes_string(xmin = "xmin", xmax = "xmax", ymin = "ymin", ymax = "ymax")),  line_args)) +
        do.call(ifelse(label_geom == "label", ggplot2::geom_label, ggplot2::geom_text), c(list(ggplot2::aes_string(x = "xmid", y = "ymin", label = "class")), label_args)) +
        ggplot2::coord_fixed() + ggplot2::xlim(c(0, iwh[1])) + ggplot2::ylim(c(0, iwh[2]))
}
