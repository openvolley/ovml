#' Construct YOLO network
#'
#' @references https://github.com/pjreddie/darknet
#' @param version integer or string: one of
#' - 3 : YOLO v3
#' - 4 : YOLO v4
#' - "4-tiny" : YOLO v4-tiny
#' - "4-mvb" : an experimental network trained specifically to detect (only) volleyballs
#'
#' @param device string: "cpu" or "cuda"
#' @param weights_file string: either the path to the weights file that already exists on your system or "auto". If "auto", the weights file will be downloaded if necessary and stored in the directory given by [ovml_cache_dir()]
#' @param class_labels character: the class labels used for network training. If missing or NULL, these default to `ovml_class_labels("coco")` for all models except "4-mvb", which uses `ovml_class_labels("mvb")`
#'
#' @return A YOLO network object
#'
#' @examples
#' \dontrun{
#'   dn <- ovml_yolo()
#'   img <- ovml_example_image()
#'   res <- ovml_yolo_detect(dn, img)
#'   ovml_ggplot(img, res)
#' }
#'
#' @export
ovml_yolo <- function(version = 4, device = "cpu", weights_file = "auto", class_labels) {
    if (is.numeric(version)) version <- as.character(version)
    assert_that(version %in% c("3", "4", "4-tiny", "4-mvb"))
    assert_that(is.string(device))
    device <- tolower(device)
    device <- match.arg(device, c("cpu", "cuda"))
    if (device == "cuda" && !cuda_is_available()) {
        warning("'cuda' device not available, using 'cpu'")
        device <- "cpu"
    }
    if (version == "3") {
        if (missing(class_labels) || length(class_labels) < 1 || is.na(class_labels)) class_labels <- ovml_class_labels("coco")
        dn <- yolo3_darknet(system.file(paste0("extdata/yolo/yolov", version, ".cfg"), package = "ovml"), device = device)
        w_url <- "https://pjreddie.com/media/files/yolov3.weights"
    } else if (version %in% c("4", "4-tiny", "4-mvb")) {
        dn <- yolo4_darknet(system.file(paste0("extdata/yolo/yolov", version, ".cfg"), package = "ovml"), device = device)
        if (version == "4") {
            if (missing(class_labels) || length(class_labels) < 1 || is.na(class_labels)) class_labels <- ovml_class_labels("coco")
            w_url <- "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
        } else if (version == "4-tiny") {
            if (missing(class_labels) || length(class_labels) < 1 || is.na(class_labels)) class_labels <- ovml_class_labels("coco")
            w_url <- "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
        } else {
            if (missing(class_labels) || length(class_labels) < 1 || is.na(class_labels)) class_labels <- ovml_class_labels("mvb")
            w_url <- "https://github.com/openvolley/ovml/releases/download/latest/yolov4-mvb.weights"
        }
    }
    dn$class_labels <- class_labels
    if (length(weights_file) && nzchar(weights_file) && !is.na(weights_file)) {
        if (identical(tolower(weights_file), "auto")) {
            weights_file <- ovml_download_if(w_url, dest = paste0("yolov", version, ".weights"))
        }
        if (file.exists(weights_file)) {
            dn$load_weights(weights_file)
        }
    }
    if (device == "cuda") dn$to(torch_device(device))
    dn$eval() ## set to inference mode
    dn
}

#' Detect objects in image using a YOLO network
#'
#' @param net yolo: as returned by [ovml_yolo()]
#' @param image_file character: path to one or more image files
#' @param conf scalar: confidence level
#' @param nms_conf scalar: non-max suppression confidence level
#'
#' @return A data.frame with columns "image_number", "class", "score", "xmin", "xmax", "ymin", "ymax"
#'
#' @seealso [ovml_yolo()]
#'
#' @examples
#' \dontrun{
#'   dn <- ovml_yolo()
#'   img <- ovml_example_image()
#'   res <- ovml_yolo_detect(dn, img)
#'   ovml_ggplot(img, res)
#' }
#' @export
ovml_yolo_detect <- function(net, image_file, conf = 0.6, nms_conf = 0.4) {
    input_image_size <- as.integer(net$blocks[[1]]$height)
    if (length(input_image_size) < 1 || is.na(input_image_size) || input_image_size <= 0) stop("invalid input_image_size: ", input_image_size)
    if (length(net$num_classes) < 1 || is.na(net$num_classes)) stop("invalid number of classes")
    if (length(net$class_labels) != net$num_classes) stop("length of class_labels does not match the number of classes")
    imgs <- lapply(image_file, function(im) {
        image <- image_read(im) ## h x w x rgb
        resized_image <- as.numeric(image_data(image_resz(image, input_image_size, preserve_aspect = YOLO3_LETTERBOXING), "rgb"))
        list(tensor = torch_tensor(aperm(array(resized_image, dim = c(1, dim(resized_image))), c(1, 4, 2, 3))), original_wh = image_wh(image))
    })
    img_tensor <- torch_cat(lapply(imgs, function(z) z$tensor), dim = 1)
    output <- as.array(net$forward(img_tensor))
    owh <- do.call(rbind, lapply(imgs, function(z) z$original_wh))
    write_results(output, num_classes = net$num_classes, confidence = conf, nms_conf = nms_conf, original_wh = owh, input_image_size = input_image_size, class_labels = net$class_labels)
}

#' Class labels
#'
#' @param dataset string: which dataset? One of
#' - "coco" (used with the 3, 4, and "4-tiny" models)
#' - "mvb" (used with the yolov4-mvb model)
#' @return A character vector of class labels
#'
#' @export
ovml_class_labels <- function(dataset = "coco") {
    assert_that(is.string(dataset))
    dataset <- tolower(dataset)
    dataset <- match.arg(dataset, c("coco", "mvb"))
    switch(dataset,
           "coco" = {
               ## old, with missing elements. This might be needed with the original C darknet implementation?
               ##c("person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", NA_character_, "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", NA_character_, "backpack", "umbrella", NA_character_, NA_character_, "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", NA_character_, "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", NA_character_, "diningtable", NA_character_, NA_character_, "toilet", NA_character_, "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", NA_character_, "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
               c("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
           },
           "mvb" = c("volleyball"),
           stop("unexpected dataset: ", dataset)
           )
}
